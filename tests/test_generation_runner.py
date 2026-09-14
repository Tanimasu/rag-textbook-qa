import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_textbook_qa.evaluation.generation import Arm, ContextVariant, GenerationCase
from rag_textbook_qa.evaluation.generation_runner import (
    generation_request,
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


class ExperimentTests(unittest.TestCase):
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

        self.assertEqual((report["planned"], report["generated"], report["judged"]), (4, 4, 4))
        self.assertAlmostEqual(report["arms"]["hot"]["problem_rate"], 0.5)
        self.assertAlmostEqual(report["arms"]["hot"]["problem_claims"], 1.0)
        self.assertEqual(sorted(calls), [0.2, 0.2, 0.7, 0.7])
        self.assertEqual(again["judged"], 4)

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
        self.assertEqual(report["usage"]["kept"]["prompt_tokens"], 0)

    def test_malformed_judge_output_fails_only_that_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run(output, [Arm("kept", "baseline", None)], judge=lambda prompt: "不是 JSON")
            failure = json.loads((output / "failures.jsonl").read_text(encoding="utf-8"))

        self.assertEqual((report["generated"], report["judged"]), (1, 0))
        self.assertTrue(failure["detail"].startswith("judge_output_invalid"))


if __name__ == "__main__":
    unittest.main()
