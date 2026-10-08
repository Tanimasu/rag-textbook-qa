import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from filelock import FileLock

from rag_textbook_qa.evaluation.generation import Arm, ContextVariant, GenerationCase
from rag_textbook_qa.evaluation.generation_runner import (
    JsonlLog,
    freeze_protocol,
    generation_request,
    llm_pair_from_env,
    openai_generator,
    run_generation_experiment,
    sample_keys,
    with_retries,
)
from rag_textbook_qa.llm.client import LLMClient

SOURCE = {"citation_id": 1, "content": "进程是系统进行资源分配和调度的一个独立单位。"}
CASE = GenerationCase(
    "01",
    "什么是进程？",
    ("进程的含义",),
    {"baseline": ContextVariant("【参考资料 1】\n" + SOURCE["content"], (SOURCE,), ("存档回答",))},
)


def fake_judge(prompt: str) -> str:
    if "任务一" in prompt:
        payload = {
            "claims": [
                {"id": 1, "type": "fact", "text": "进程是独立单位"},
                {"id": 2, "type": "fact", "text": "进程由操作系统创建"},
            ],
            "coverage": [{"id": 1, "addressed": True}],
        }
    else:
        quote = {"source_id": 1, "quote": "资源分配和调度的一个独立单位"}
        payload = {
            "verdicts": [
                {"id": 1, "label": "supported", "evidence": [quote]},
                {"id": 2, "label": "unsupported", "evidence": [], "reason": "片段未提"},
            ]
        }
    return json.dumps(payload, ensure_ascii=False)


def fake_answer(text: str) -> dict:
    return {
        "answer": text,
        "finish_reason": "stop",
        "reasoning_chars": 0,
        "tokens": {"prompt": 1, "completion": 1},
        "seconds": 0.1,
        "attempts": 1,
    }


def run(output: Path, arms: list[Arm], **overrides) -> dict:
    options = {
        "generator": lambda prompt, temperature: fake_answer("回答"),
        "judge": fake_judge,
        "samples": 2,
        "seed": 0,
        "concurrency": 2,
        "protocol": {"generator": "fake"},
        "prompt_builder": lambda question, context: question + context,
        "log": lambda line: None,
    }
    return run_generation_experiment([CASE], arms, output_dir=output, **{**options, **overrides})


class ClientTests(unittest.TestCase):
    def test_invalid_direct_cases_stop_before_generator_judge_or_protocol_write(self):
        invalid_cases = [[], [CASE, CASE], [replace(CASE, question=" ")],
                         [replace(CASE, variants={"baseline": replace(CASE.variants["baseline"], answers="答案")})]]
        for cases in invalid_cases:
            with self.subTest(cases=cases), tempfile.TemporaryDirectory() as directory:
                generator, judge = MagicMock(), MagicMock()
                output = Path(directory)
                with self.assertRaises(ValueError):
                    run_generation_experiment(cases, [Arm("hot", "baseline", 0.7)],
                        output_dir=output, generator=generator, judge=judge, samples=1,
                        seed=0, concurrency=1, protocol={})
                generator.assert_not_called()
                judge.assert_not_called()
                self.assertFalse((output / "protocol.json").exists())

    def test_invalid_arms_stop_before_model_calls(self):
        invalid_arms = [[], [Arm("x", "baseline", 0.7)] * 2,
                        [Arm("x", "baseline", float("inf"))],
                        [Arm("x", "baseline", -1)], [Arm(" ", "baseline", 0.7)]]
        for arms in invalid_arms:
            with self.subTest(arms=arms), tempfile.TemporaryDirectory() as directory:
                generator, judge = MagicMock(), MagicMock()
                with self.assertRaises(ValueError):
                    run(Path(directory), arms, generator=generator, judge=judge)
                generator.assert_not_called()
                judge.assert_not_called()

    def test_generation_request_matches_the_product_call(self):
        sdk = MagicMock()
        message = SimpleNamespace(content="答")
        sdk.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None
        )
        client = LLMClient("key", "https://example.com/v1", "model-x", verbose=False, sdk_client=sdk)
        client.generate_answer("提示词", temperature=0.2, max_tokens=2000, retry=0)

        self.assertEqual(
            sdk.chat.completions.create.call_args.kwargs,
            generation_request("model-x", "提示词", 0.2, 2000),
        )

    def test_generator_keeps_usage_but_not_reasoning_text(self):
        sdk = MagicMock()
        message = SimpleNamespace(content="答", reasoning_content="想了很久")
        sdk.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        result = openai_generator(sdk, "model-x", max_tokens=100)("提示词", 0.7)

        sdk.with_options.assert_called_once_with(timeout=180.0, max_retries=0)
        self.assertEqual((result["answer"], result["reasoning_chars"]), ("答", 4))
        self.assertEqual(result["tokens"], {"prompt": 10, "completion": 5})
        self.assertNotIn("想了很久", json.dumps(result, ensure_ascii=False))

    def test_retries_transient_errors_only(self):
        class RateLimitError(Exception):
            pass

        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise RateLimitError
            return "ok"

        self.assertEqual(with_retries(flaky, sleep=lambda seconds: None), ("ok", 3))
        with self.assertRaises(KeyError):
            with_retries(lambda: {}["missing"], sleep=lambda seconds: None)


class PairLifecycleTests(unittest.TestCase):
    def test_partial_pair_setup_closes_every_client_already_created(self):
        for failed_at in ("judge", "options"):
            with self.subTest(failed_at=failed_at):
                generator, judge = MagicMock(), MagicMock()
                generator.default_model, judge.default_model = "generator", "judge"
                with (
                    patch("rag_textbook_qa.llm.client.create_llm_client", side_effect=[
                        generator, RuntimeError("judge failed") if failed_at == "judge" else judge,
                    ]),
                    patch("rag_textbook_qa.evaluation.ragas.judge_model_kwargs",
                          side_effect=RuntimeError("options failed")),
                    self.assertRaises(RuntimeError),
                ):
                    llm_pair_from_env()
                generator.close.assert_called_once_with()
                if failed_at == "options":
                    judge.close.assert_called_once_with()
                else:
                    judge.close.assert_not_called()


class ProtocolPersistenceTests(unittest.TestCase):
    def test_changed_requirements_cannot_reuse_generated_answer_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arms = [Arm("hot", "baseline", 0.7)]
            run(output, arms)
            generator, judge = MagicMock(), MagicMock()
            with self.assertRaisesRegex(ValueError, "冻结"):
                run_generation_experiment(
                    [replace(CASE, requirements=("新的答案要点",))], arms, output_dir=output,
                    generator=generator, judge=judge, samples=2, seed=0, concurrency=2,
                    protocol={"generator": "fake"},
                    prompt_builder=lambda question, context: question + context,
                    log=lambda line: None,
                )
            generator.assert_not_called()
            judge.assert_not_called()

    def test_legacy_protocol_without_case_binding_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arms = [Arm("kept", "baseline", None)]
            run(output, arms)
            path = output / "protocol.json"
            protocol = json.loads(path.read_text(encoding="utf-8"))
            del protocol["settings"]["frozen_cases_sha256"]
            path.write_text(json.dumps(protocol, ensure_ascii=False), encoding="utf-8")
            originals = {p: p.read_bytes() for p in output.iterdir() if p.is_file()}
            generator, judge = MagicMock(), MagicMock()
            with self.assertRaisesRegex(ValueError, "冻结"):
                run(output, arms, generator=generator, judge=judge)
            generator.assert_not_called()
            judge.assert_not_called()
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)

    def test_changed_frozen_inputs_cannot_reuse_stored_generations_or_judgments(self):
        baseline = CASE.variants["baseline"]
        changed_variants = (
            replace(baseline, answers=("另一份存档回答",)),
            replace(baseline, context=baseline.context + "\n补充资料。"),
            replace(baseline, sources=({**SOURCE, "content": "资源分配和调度的一个独立单位"},)),
        )
        changed_cases = (
            replace(CASE, question="进程的含义是什么？"),
            replace(CASE, requirements=("回答必须解释调度",)),
            *(replace(CASE, variants={"baseline": variant}) for variant in changed_variants),
        )
        for changed in changed_cases:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                arms = [Arm("kept", "baseline", None)]
                run(output, arms)
                originals = {p: p.read_bytes() for p in output.iterdir() if p.is_file()}
                generator, judge = MagicMock(), MagicMock()
                with self.assertRaisesRegex(ValueError, "冻结"):
                    run_generation_experiment(
                        [changed], arms, output_dir=output, generator=generator, judge=judge,
                        samples=2, seed=0, concurrency=2, protocol={"generator": "fake"},
                        prompt_builder=lambda question, context: question + context,
                        log=lambda line: None,
                    )
                generator.assert_not_called()
                judge.assert_not_called()
                for path, original in originals.items():
                    self.assertEqual(path.read_bytes(), original)

    def test_prior_judge_version_cannot_resume_or_rewrite_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arms = [Arm("hot", "baseline", 0.7)]
            run(output, arms)
            protocol = output / "protocol.json"
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["settings"]["judge_version"] = 3
            protocol.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            originals = {p: p.read_bytes() for p in output.iterdir() if p.is_file()}
            generator, judge = MagicMock(), MagicMock()
            with self.assertRaisesRegex(ValueError, "冻结"):
                run(output, arms, generator=generator, judge=judge)
            generator.assert_not_called()
            judge.assert_not_called()
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)

    def test_failed_protocol_write_leaves_no_frozen_partial_file(self):
        original_write = Path.write_text

        def interrupted(path, text, **kwargs):
            original_write(path, text[:10], **kwargs)
            raise OSError("fixture protocol write interrupted")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            settings = {"generator": {"model": "fixture"}}
            with patch.object(Path, "write_text", interrupted), self.assertRaises(OSError):
                freeze_protocol(path, settings)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
            freeze_protocol(path, settings)
            original = path.read_bytes()
            freeze_protocol(path, settings)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                freeze_protocol(path, {"generator": {"model": "changed"}})
            self.assertEqual(path.read_bytes(), original)


class ExperimentTests(unittest.TestCase):
    def test_interrupt_cancels_queued_judgments_and_resume_keeps_finished_score(self):
        started, release = threading.Event(), threading.Event()
        original_shutdown = ThreadPoolExecutor.shutdown
        stages = []
        calls = []

        def judge(prompt):
            calls.append(prompt)
            started.set()
            if not release.wait(3):
                raise RuntimeError("fixture release timed out")
            return fake_judge(prompt)

        def interrupt_judging(futures):
            stages.append(1)
            if len(stages) == 1:
                return as_completed(futures)
            self.assertTrue(started.wait(2))
            raise KeyboardInterrupt

        def release_during_shutdown(pool, **kwargs):
            if started.is_set():
                release.set()
            return original_shutdown(pool, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arms = [Arm(f"kept-{i}", "baseline", None) for i in range(20)]
            generator = MagicMock()
            with (
                patch("rag_textbook_qa.evaluation.generation_runner.as_completed", interrupt_judging),
                patch.object(ThreadPoolExecutor, "shutdown", release_during_shutdown),
                self.assertRaises(KeyboardInterrupt),
            ):
                run(output, arms, generator=generator, judge=judge, concurrency=1)
            self.assertEqual(len(calls), 2)  # extract + verify for the one in-flight sample
            self.assertEqual(len(JsonlLog(output / "judgments.jsonl").records), 1)
            report = run(output, arms, generator=generator, judge=judge, concurrency=1)
            self.assertEqual(len(calls), 40)
            self.assertEqual((report["generated"], report["judged"]), (20, 20))
            generator.assert_not_called()

    def test_interrupt_cancels_queued_samples_and_resume_keeps_finished_output(self):
        started, release = threading.Event(), threading.Event()
        calls = []
        original_shutdown = ThreadPoolExecutor.shutdown

        def generate(prompt, temperature):
            calls.append(prompt)
            started.set()
            if not release.wait(3):
                raise RuntimeError("fixture release timed out")
            return fake_answer("回答")

        def interrupted(futures):
            self.assertTrue(started.wait(2))
            raise KeyboardInterrupt

        def release_during_shutdown(pool, **kwargs):
            # The request finishes only after shutdown begins, without sleeps
            # or a scheduling race between the interrupt and the test timer.
            release.set()
            return original_shutdown(pool, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arms = [Arm("hot", "baseline", 0.7)]
            judge = MagicMock(side_effect=fake_judge)
            with (
                patch("rag_textbook_qa.evaluation.generation_runner.as_completed", interrupted),
                patch.object(ThreadPoolExecutor, "shutdown", release_during_shutdown),
                self.assertRaises(KeyboardInterrupt),
            ):
                run(output, arms, generator=generate, judge=judge, samples=20, concurrency=1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(JsonlLog(output / "generations.jsonl").records), 1)
            judge.assert_not_called()
            report = run(output, arms, generator=generate, judge=judge, samples=20, concurrency=1)
            self.assertEqual(len(calls), 20)
            self.assertEqual((report["generated"], report["judged"]), (20, 20))

    def test_failed_report_update_preserves_the_previous_complete_report(self):
        original_write = Path.write_text

        def interrupted(path, text, **kwargs):
            original_write(path, text[:10], **kwargs)
            raise OSError("fixture report write interrupted")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arms = [Arm("hot", "baseline", 0.7)]
            generator = MagicMock(return_value=fake_answer("回答"))
            judge = MagicMock(side_effect=fake_judge)
            run(output, arms, generator=generator, judge=judge)
            original = (output / "report.json").read_bytes()
            previous_files = set(output.iterdir())
            generator.reset_mock()
            judge.reset_mock()
            with patch.object(Path, "write_text", interrupted), self.assertRaises(OSError):
                run(output, arms, generator=generator, judge=judge)
            self.assertEqual((output / "report.json").read_bytes(), original)
            self.assertEqual(set(output.iterdir()), previous_files)
            generator.assert_not_called()
            judge.assert_not_called()
            self.assertEqual(run(output, arms, generator=generator, judge=judge)["judged"], 2)
            generator.assert_not_called()
            judge.assert_not_called()

    def test_directory_lock_refuses_another_runner_before_paid_calls(self):
        generator, judge = MagicMock(), MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with FileLock(output / ".run.lock", timeout=0), self.assertRaisesRegex(ValueError, "正在运行"):
                run(output, [Arm("hot", "baseline", 0.7)], generator=generator, judge=judge)
            generator.assert_not_called()
            judge.assert_not_called()
            self.assertFalse((output / "protocol.json").exists())
            self.assertEqual(run(output, [Arm("hot", "baseline", 0.7)])["judged"], 2)

    def test_directory_lock_releases_when_prompt_preparation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "cannot render"):
                run(output, [Arm("hot", "baseline", 0.7)],
                    prompt_builder=MagicMock(side_effect=RuntimeError("cannot render")))
            self.assertEqual(run(output, [Arm("hot", "baseline", 0.7)])["judged"], 2)

    def test_sample_order_is_seeded_and_stored_arms_use_their_answers(self):
        arms = [Arm("hot", "baseline", 0.7), Arm("kept", "baseline", None)]
        keys = sample_keys([CASE], arms, 3, seed=1)

        self.assertEqual(
            sorted(keys),
            [("01", "hot", 0), ("01", "hot", 1), ("01", "hot", 2), ("01", "kept", 0)],
        )
        self.assertEqual(keys, sample_keys([CASE], arms, 3, seed=1))

    def test_resumes_without_new_calls_and_refuses_a_changed_protocol(self):
        calls = []

        def generator(prompt, temperature):
            calls.append(temperature)
            return fake_answer(f"回答{len(calls)}")

        arms = [Arm("hot", "baseline", 0.7), Arm("cool", "baseline", 0.2)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run(output, arms, generator=generator)
            again = run(output, arms, generator=generator)
            with self.assertRaises(ValueError):
                run(output, arms, generator=generator, samples=3)
            with self.assertRaises(ValueError):
                run(
                    output,
                    arms,
                    generator=generator,
                    prompt_builder=lambda question, context: "已修改：" + question + context,
                )

        self.assertEqual((report["planned"], report["generated"], report["judged"]), (4, 4, 4))
        self.assertAlmostEqual(report["arms"]["hot"]["problem_rate"], 0.5)
        self.assertAlmostEqual(report["arms"]["hot"]["problem_claims"], 1.0)
        self.assertEqual(sorted(calls), [0.2, 0.2, 0.7, 0.7])
        self.assertEqual(again["judged"], 4)

    def test_jsonl_resume_separates_a_new_record_from_a_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generations.jsonl"
            path.write_text('{"case_id":', encoding="utf-8")
            record = {"case_id": "01", "arm": "hot", "index": 0, "answer": "回答"}

            log = JsonlLog(path)
            log.append(record)
            reloaded = JsonlLog(path)

        self.assertEqual(log.records[("01", "hot", 0)], record)
        self.assertEqual(reloaded.records[("01", "hot", 0)], record)

    def test_jsonl_recovers_good_records_around_invalid_utf8_without_rewriting_them(self):
        first = {"case_id": "01", "arm": "hot", "index": 0, "answer": "原始中文回答"}
        following = {"case_id": "01", "arm": "hot", "index": 2, "answer": "后续回答"}
        for damaged in (
            b'{"case_id":"01","arm":"hot","index":1,"answer":"' + "中".encode()[:2],
            b'{"case_id":"01","arm":"hot","index":1,"answer":"\xff"}\n',
        ):
            with self.subTest(damaged=damaged), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "generations.jsonl"
                original = json.dumps(first, ensure_ascii=False).encode() + b"\n" + damaged
                path.write_bytes(original)
                log = JsonlLog(path)
                self.assertEqual(log.lines, [first])
                log.append(following)
                reloaded = JsonlLog(path)
                self.assertEqual(reloaded.lines, [first, following])
                self.assertTrue(path.read_bytes().startswith(original))

    def test_failed_partial_append_is_separated_from_the_next_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generations.jsonl"
            log = JsonlLog(path)
            failed = {"case_id": "01", "arm": "hot", "index": 0, "answer": "未完全保存"}
            following = {"case_id": "01", "arm": "hot", "index": 1, "answer": "已保存"}
            underlying = path.open("a", encoding="utf-8")

            class PartialWriter:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    underlying.close()

                def write(self, value):
                    underlying.write(value[:len(value) // 2])
                    underlying.flush()
                    raise OSError("fixture disk write failed")

            with patch.object(Path, "open", return_value=PartialWriter()), self.assertRaises(OSError):
                log.append(failed)
            self.assertEqual(log.records, {})
            log.append(following)
            self.assertEqual(JsonlLog(path).lines, [following])

    def test_failed_close_does_not_register_an_unsaved_result_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            log = JsonlLog(Path(directory) / "generations.jsonl")
            record = {"case_id": "01", "arm": "hot", "index": 0, "answer": "回答"}

            class FailedFlush:
                def __enter__(self):
                    return self

                def write(self, value):
                    return len(value)

                def __exit__(self, *args):
                    raise OSError("fixture buffered flush failed")

            with patch.object(Path, "open", return_value=FailedFlush()), self.assertRaises(OSError):
                log.append(record)
            self.assertEqual(log.lines, [])
            self.assertEqual(log.records, {})

    def test_failed_samples_are_logged_without_messages_and_redone(self):
        calls = []

        def generator(prompt, temperature):
            calls.append(temperature)
            if len(calls) == 1:
                raise ConnectionError("network down")
            return fake_answer("回答")

        arms = [Arm("hot", "baseline", 0.7)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = run(output, arms, generator=generator, concurrency=1)
            failure = json.loads((output / "failures.jsonl").read_text(encoding="utf-8"))
            second = run(output, arms, generator=generator, concurrency=1)

        self.assertEqual((first["generated"], first["failures_logged"]), (1, 1))
        self.assertEqual(failure["error_type"], "ConnectionError")
        self.assertNotIn("network down", json.dumps(failure))
        self.assertEqual(second["generated"], 2)

    def test_stored_answers_are_judged_without_generating(self):
        def generator(prompt, temperature):
            raise AssertionError("stored arms must not call the generator")

        with tempfile.TemporaryDirectory() as directory:
            report = run(Path(directory), [Arm("kept", "baseline", None)], generator=generator)

        self.assertEqual((report["planned"], report["judged"]), (1, 1))
        self.assertEqual((report["completed"], report["incomplete"]), (1, 0))
        self.assertEqual(report["usage"]["kept"]["prompt_tokens"], 0)

    def test_incomplete_generations_are_saved_without_judging_or_regenerating(self):
        for reason in ("length", "content_filter", "tool_calls", None, "missing"):
            with self.subTest(finish_reason=reason), tempfile.TemporaryDirectory() as directory:
                answer = fake_answer("未完成的正文")
                if reason == "missing":
                    answer.pop("finish_reason")
                else:
                    answer["finish_reason"] = reason
                generator = MagicMock(return_value=answer)
                judge = MagicMock(side_effect=fake_judge)
                output = Path(directory)
                arms = [Arm("hot", "baseline", 0.7)]

                report = run(output, arms, generator=generator, judge=judge, samples=1)
                again = run(output, arms, generator=generator, judge=judge, samples=1)
                saved = JsonlLog(output / "generations.jsonl")

                self.assertEqual(generator.call_count, 1)
                judge.assert_not_called()
                self.assertEqual(len(saved.lines), 1)
                self.assertEqual(saved.lines[0]["answer"], "未完成的正文")
                self.assertEqual(report["usage"]["hot"]["completion_tokens"], 1)
                self.assertEqual((report["generated"], report["completed"], report["incomplete"],
                                  report["judged"], report["failures_logged"]), (1, 0, 1, 0, 0))
                self.assertEqual(report["truncated"], int(reason == "length"))
                self.assertIsNone(report["arms"]["hot"]["problem_claims"])
                self.assertEqual(again["incomplete"], 1)
                self.assertFalse((output / "judgments.jsonl").exists())

    def test_empty_stopped_answers_are_saved_without_judging(self):
        for text in ("", " \n\t"):
            with self.subTest(answer=text), tempfile.TemporaryDirectory() as directory:
                generator = MagicMock(return_value=fake_answer(text))
                judge = MagicMock(side_effect=fake_judge)
                report = run(Path(directory), [Arm("hot", "baseline", 0.7)],
                             generator=generator, judge=judge, samples=1)

                judge.assert_not_called()
                self.assertEqual((report["generated"], report["completed"], report["incomplete"],
                                  report["empty_answers"], report["judged"]), (1, 0, 1, 1, 0))
                self.assertEqual(report["usage"]["hot"]["completion_tokens"], 1)

    def test_resume_excludes_legacy_incomplete_judgments_and_preserves_paid_results(self):
        arms = [Arm("hot", "baseline", 0.7), Arm("cool", "baseline", 0.2)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run(output, arms, samples=1)
            path = output / "generations.jsonl"
            records = JsonlLog(path).lines
            for record in records:
                if record["arm"] == "hot":
                    record["finish_reason"] = "length"
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            original_judgments = (output / "judgments.jsonl").read_bytes()
            generator = MagicMock()
            judge = MagicMock()

            report = run(output, arms, generator=generator, judge=judge, samples=1)

            generator.assert_not_called()
            judge.assert_not_called()
            self.assertEqual((output / "judgments.jsonl").read_bytes(), original_judgments)
            self.assertEqual((report["generated"], report["judged"], report["incomplete"]), (2, 1, 1))
            self.assertEqual(report["usage"]["hot"]["completion_tokens"], 1)
            self.assertEqual(report["summary_version"], 2)
            self.assertEqual(report["quality_sample_policy"], "completed_answers_only")
            self.assertEqual(report["settings"]["judge_version"], 4)
            self.assertIsNone(report["arms"]["hot"]["problem_claims"])
            self.assertTrue(all(value is None for value in report["comparisons"]["cool"].values()))

    def test_malformed_judge_output_fails_only_that_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run(output, [Arm("kept", "baseline", None)], judge=lambda prompt: "不是 JSON")
            failure = json.loads((output / "failures.jsonl").read_text(encoding="utf-8"))

        self.assertEqual((report["generated"], report["judged"]), (1, 0))
        self.assertTrue(failure["detail"].startswith("judge_output_invalid"))


if __name__ == "__main__":
    unittest.main()
