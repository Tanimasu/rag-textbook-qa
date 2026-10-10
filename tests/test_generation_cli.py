import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_textbook_qa.cli import main

SECRET = "secret-key-for-test"


def workspace(root: Path) -> Path:
    (root / "src" / "rag_textbook_qa").mkdir(parents=True)
    (root / "project").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (root / "project" / ".env").write_text("", encoding="utf-8")
    source = {"citation_id": 1, "content": "进程是系统进行资源分配的单位。"}
    variant = {"context": "内容:\n" + source["content"], "sources": [source]}
    cases = {"cases": [{"id": "01", "question": "什么是进程？", "variants": {"baseline": variant}}]}
    path = root / "cases.json"
    path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    return path


def fake_client(model: str) -> MagicMock:
    llm = MagicMock()
    llm.default_model = model
    llm.base_url = "https://api.example.com/v1/"
    return llm


class GenerationCliTests(unittest.TestCase):
    def test_ambiguous_frozen_json_stops_before_clients_or_outputs(self):
        for fragment, ambiguous in (
            ('"question":', '"question":"先出现的问题","question":'),
            ('"cases":', '"ignored":NaN,"cases":'),
            ('"cases":', '"ignored":Infinity,"cases":'),
            ('"cases":', '"ignored":1e400,"cases":'),
        ):
            with self.subTest(ambiguous=ambiguous), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = workspace(root)
                original = path.read_text().replace(fragment, ambiguous, 1)
                path.write_text(original)
                with (
                    patch("rag_textbook_qa.evaluation.generation_runner.llm_pair_from_env") as pair,
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    main(["--workspace", str(root), "evaluate-generation", "--cases", str(path),
                          "--arm", "hot=baseline@0.7", "--output-dir", str(root / "output")])
                pair.assert_not_called()
                self.assertFalse((root / "output").exists())
                self.assertEqual(path.read_text(), original)

    def test_invalid_attempt_cap_stops_before_clients_or_output_creation(self):
        for value in ("0", "-1", "1.5"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = workspace(root)
                with (
                    patch("rag_textbook_qa.evaluation.generation_runner.llm_pair_from_env") as pair,
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    main(["--workspace", str(root), "evaluate-generation", "--cases", str(path),
                          "--arm", "hot=baseline@0.7", "--max-http-attempts", value,
                          "--output-dir", str(root / "output")])
                pair.assert_not_called()
                self.assertFalse((root / "output").exists())

    def test_cli_shares_cap_preserves_partial_report_and_closes_clients(self):
        generator, judge = fake_client("generator"), fake_client("judge")
        generator.client.with_options.return_value.chat.completions.create.return_value = (
            SimpleNamespace(usage=None, choices=[SimpleNamespace(
                message=SimpleNamespace(content="回答"), finish_reason="stop")]))
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = workspace(root)
            with (
                patch("rag_textbook_qa.evaluation.generation_runner.llm_pair_from_env",
                      return_value=(generator, judge, {})),
                contextlib.redirect_stdout(stdout),
            ):
                status = main(["--workspace", str(root), "evaluate-generation", "--cases", str(path),
                    "--arm", "hot=baseline@0.7", "--samples", "1", "--concurrency", "1",
                    "--max-http-attempts", "1", "--output-dir", str(root / "output")])
            report = json.loads((root / "output" / "report.json").read_text())
            self.assertEqual((report["generated"], report["judged"]), (1, 0))
            self.assertEqual(report["http_attempt_budget"]["attempts_started"], 1)
            self.assertEqual(len((root / "output" / "calls.jsonl").read_text().splitlines()), 1)
        self.assertEqual(status, 2)
        self.assertIn("续跑将重新计数", stdout.getvalue())
        generator.client.with_options.return_value.chat.completions.create.assert_called_once()
        judge.client.with_options.return_value.chat.completions.create.assert_not_called()
        generator.close.assert_called_once_with()
        judge.close.assert_called_once_with()

    def test_protocol_hash_describes_loaded_snapshot_if_file_changes_during_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = workspace(root)
            original = path.read_bytes()
            payload = json.loads(original)

            def create_pair():
                payload["cases"][0]["question"] = "客户端初始化期间换了一道题"
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return fake_client("generator"), fake_client("judge"), {}

            report = {"planned": 0, "generated": 0, "completed": 0, "incomplete": 0,
                      "truncated": 0, "empty_answers": 0, "judged": 0,
                      "reference_arm": "hot", "arms": {}, "comparisons": {}}
            with (
                patch("rag_textbook_qa.evaluation.generation_runner.llm_pair_from_env",
                      side_effect=create_pair),
                patch("rag_textbook_qa.evaluation.generation_runner.run_generation_experiment",
                      return_value=report) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(["--workspace", str(root), "evaluate-generation", "--cases", str(path),
                      "--arm", "hot=baseline@0.7", "--output-dir", str(root / "output")])
            self.assertEqual(run.call_args.args[0][0].question, "什么是进程？")
            self.assertEqual(run.call_args.kwargs["protocol"]["cases_sha256"],
                             hashlib.sha256(original).hexdigest())

    def test_dry_run_counts_generated_and_stored_samples_without_clients_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = workspace(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cases"][0]["variants"]["baseline"]["answers"] = ["存档一", "存档二"]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            stdout = io.StringIO()
            destination = root / "intended-run"
            with (
                patch.dict(os.environ, {"LLM_API_KEY": SECRET, "LLM_MODEL": "shared",
                                        "RAG_MODEL": "generator", "RAGAS_MODEL": "judge"}, clear=True),
                patch("rag_textbook_qa.evaluation.generation_runner.llm_pair_from_env") as pair,
                patch("rag_textbook_qa.evaluation.generation_runner.run_generation_experiment") as run,
                contextlib.redirect_stdout(stdout),
            ):
                main(["--workspace", str(root), "evaluate-generation", "--cases", str(path),
                      "--arm", "hot=baseline@0.7", "--arm", "kept=baseline@stored",
                      "--samples", "3", "--output-dir", str(destination),
                      "--max-http-attempts", "17", "--dry-run"])
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["models"], {"generator": "generator", "judge": "judge"})
            self.assertEqual(plan["planned_samples"], 5)
            self.assertEqual(plan["new_generation_samples"], 3)
            self.assertEqual(plan["stored_samples"], 2)
            self.assertEqual(plan["nominal_judge_calls_if_all_complete"], 10)
            self.assertEqual(plan["max_generation_http_attempts"], 12)
            self.assertEqual(plan["max_judge_http_attempts"], 120)
            self.assertEqual(plan["model_calls"], 0)
            self.assertEqual(plan["http_attempt_budget"], {"scope": "invocation", "limit": 17,
                                                         "attempts_started": 0, "blocked": False})
            self.assertIsNone(plan["cost_estimate"])
            self.assertNotIn(SECRET, stdout.getvalue())
            pair.assert_not_called()
            run.assert_not_called()
            self.assertFalse(destination.exists())

    def test_invalid_frozen_answers_are_rejected_before_creating_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = workspace(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cases"][0]["variants"]["baseline"]["answers"] = "保存过的回答"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with (
                patch("rag_textbook_qa.evaluation.generation_runner.llm_pair_from_env") as pair,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(["--workspace", str(root), "evaluate-generation", "--cases", str(path),
                      "--arm", "kept=baseline@stored", "--output-dir", str(root / "output")])
            pair.assert_not_called()
            self.assertFalse((root / "output").exists())

    def run_cli(self, environment: dict[str, str], *, report: dict | None = None,
                stdout: io.StringIO | None = None,
                runner_error: Exception | None = None) -> tuple[MagicMock, Path]:
        self.clients = []

        def client_factory(**kwargs):
            llm = fake_client(kwargs["model"] or "gen-model")
            self.clients.append(llm)
            return llm

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = workspace(root)
            output_path = root / "generation"
            report = report or {
                "planned": 2, "generated": 2, "completed": 2, "incomplete": 0,
                "truncated": 0, "empty_answers": 0, "judged": 2, "reference_arm": "hot",
                "arms": {}, "comparisons": {},
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch(
                    "rag_textbook_qa.llm.client.create_llm_client",
                    side_effect=client_factory,
                ),
                patch(
                    "rag_textbook_qa.evaluation.generation_runner.run_generation_experiment",
                    return_value=report,
                    side_effect=runner_error,
                ) as run,
                contextlib.redirect_stdout(stdout if stdout is not None else io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                main([
                    "--workspace", str(root), "evaluate-generation",
                    "--cases", str(cases_path),
                    "--arm", "hot=baseline@0.7", "--arm", "cool=baseline@0.2",
                    "--samples", "1", "--output-dir", str(output_path),
                ])
        return run, output_path

    def test_freezes_both_models_without_calling_any_api(self):
        run, output_path = self.run_cli(
            {"LLM_API_KEY": SECRET, "LLM_MODEL": "gen-model", "RAGAS_MODEL": "judge-model"}
        )

        arms = run.call_args.args[1]
        protocol = run.call_args.kwargs["protocol"]
        self.assertEqual([arm.temperature for arm in arms], [0.7, 0.2])
        self.assertEqual(run.call_args.kwargs["output_dir"], output_path)
        self.assertEqual(protocol["judge"]["model"], "judge-model")
        self.assertEqual(protocol["generator"]["host"], "api.example.com")
        self.assertNotIn(SECRET, json.dumps(protocol))
        for client in self.clients:
            client.close.assert_called_once_with()

    def test_refuses_to_let_the_generator_judge_itself(self):
        with self.assertRaises(SystemExit):
            self.run_cli({"LLM_API_KEY": SECRET, "LLM_MODEL": "gen-model"})
        for client in self.clients:
            client.close.assert_called_once_with()

    def test_failed_experiment_closes_both_clients(self):
        with self.assertRaises(SystemExit):
            self.run_cli(
                {"LLM_API_KEY": SECRET, "LLM_MODEL": "gen-model", "RAGAS_MODEL": "judge-model"},
                runner_error=RuntimeError("fixture failure"),
            )
        self.assertEqual(len(self.clients), 2)
        for client in self.clients:
            client.close.assert_called_once_with()

    def test_reports_incomplete_answers_and_the_quality_denominator(self):
        stdout = io.StringIO()
        self.run_cli(
            {"LLM_API_KEY": SECRET, "LLM_MODEL": "gen-model", "RAGAS_MODEL": "judge-model"},
            report={
                "planned": 3, "generated": 3, "completed": 1, "incomplete": 2,
                "truncated": 1, "empty_answers": 1, "judged": 1, "reference_arm": "hot",
                "arms": {"hot": {
                    "completed": 1, "samples": 3, "incomplete": 2, "judged": 1,
                    "problem_claims": 0.0, "problem_rate": 0.0, "strict_rate": 0.0,
                    "coverage": 1.0, "overlap": None,
                }},
                "comparisons": {},
            },
            stdout=stdout,
        )

        self.assertIn("完整 1，未完成 2（截断 1，空回答 1），已评判 1", stdout.getvalue())
        self.assertIn("质量均分与配对比较仅使用完整回答", stdout.getvalue())
        self.assertIn("续跑不会重新生成", stdout.getvalue())
        self.assertIn("hot: 完整 1/3，未完成 2，已评判 1", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
