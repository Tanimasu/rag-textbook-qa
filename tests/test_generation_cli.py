import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
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
    def run_cli(self, environment: dict[str, str], *, report: dict | None = None,
                stdout: io.StringIO | None = None) -> tuple[MagicMock, Path]:
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
                    side_effect=lambda **kwargs: fake_client(kwargs["model"] or "gen-model"),
                ),
                patch(
                    "rag_textbook_qa.evaluation.generation_runner.run_generation_experiment",
                    return_value=report,
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

    def test_refuses_to_let_the_generator_judge_itself(self):
        with self.assertRaises(SystemExit):
            self.run_cli({"LLM_API_KEY": SECRET, "LLM_MODEL": "gen-model"})

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
