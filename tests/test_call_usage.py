import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_textbook_qa.evaluation.call_usage import (
    CallAttemptBudget,
    CallAttemptLimitReached,
    CallUsageLog,
    CallUsageRecordingError,
)
from rag_textbook_qa.evaluation.generation_runner import (
    openai_generator,
    openai_judge,
    with_retries,
)
from rag_textbook_qa.llm.client import LLMGenerationIncompleteError


def response(finish_reason="stop", usage=None):
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content='{"verdicts":[]}', reasoning_content="private reasoning"),
    )])


class CallUsageTests(unittest.TestCase):
    def test_concurrent_attempts_share_a_strict_cap_with_or_without_usage_observer(self):
        for observed in (False, True):
            with self.subTest(observed=observed):
                sdk = MagicMock()
                sdk.with_options.return_value.chat.completions.create.return_value = response()
                records = []
                budget = CallAttemptBudget(7)
                generator = openai_generator(sdk, "model", max_tokens=20,
                    on_call=records.append if observed else None, call_budget=budget)

                def request(_, generate=generator):
                    try:
                        generate("private prompt", .7)
                        return True
                    except CallAttemptLimitReached:
                        return False

                with ThreadPoolExecutor(max_workers=8) as pool:
                    sent = list(pool.map(request, range(40)))
                self.assertEqual(sum(sent), 7)
                self.assertEqual(sdk.with_options.return_value.chat.completions.create.call_count, 7)
                self.assertEqual(len(records), 7 if observed else 0)
                self.assertEqual(budget.snapshot(), {"scope": "invocation", "limit": 7,
                                                    "attempts_started": 7, "blocked": True})

    def test_generator_retry_and_judge_consume_the_same_attempt_budget(self):
        class RateLimitError(Exception):
            pass

        sdk = MagicMock()
        sdk.with_options.return_value.chat.completions.create.side_effect = [
            RateLimitError("private failure"), response(), response(),
        ]
        records = []
        budget = CallAttemptBudget(3)
        generator = openai_generator(sdk, "generator", max_tokens=20,
                                     on_call=records.append, call_budget=budget)
        judge = openai_judge(sdk, "judge", extra={}, on_call=records.append, call_budget=budget)
        with patch("rag_textbook_qa.evaluation.generation_runner.with_retries",
                   side_effect=lambda call: with_retries(call, sleep=lambda _: None)):
            generator("prompt", .7)
            judge("prompt")
            with self.assertRaises(CallAttemptLimitReached):
                judge("prompt")
        self.assertEqual([row["role"] for row in records], ["generator", "generator", "judge"])
        self.assertEqual([row["status"] for row in records], ["error", "response", "response"])
        self.assertEqual(sdk.with_options.return_value.chat.completions.create.call_count, 3)

    def test_invalid_attempt_limit_is_rejected(self):
        for value in (True, 0, -1, 1.5, "2", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CallAttemptBudget(value)

    def test_invalid_numeric_metadata_never_poison_existing_usage_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            recorder = CallUsageLog(path)
            recorder({"call_id": "previous", "tokens": None})
            original = path.read_bytes()
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    recorder({"call_id": "bad", "seconds": value})
                self.assertEqual(path.read_bytes(), original)

    def test_both_adapters_log_billable_usage_without_content_or_credentials(self):
        sdk = MagicMock()
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=30, total_tokens=130,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=20),
            prompt_tokens_details=SimpleNamespace(cached_tokens=40))
        sdk.with_options.return_value.chat.completions.create.return_value = response(usage=usage)
        records = []
        generator = openai_generator(sdk, "generator", max_tokens=50, on_call=records.append)
        judge = openai_judge(sdk, "judge", extra={}, on_call=records.append)
        generator("private prompt", 0.7)
        judge("private prompt")
        self.assertEqual([row["role"] for row in records], ["generator", "judge"])
        self.assertEqual(records[1]["tokens"], {"prompt":100,"completion":30,"total":130,
                                               "reasoning":20,"cached_prompt":40})
        self.assertEqual(len({row["call_id"] for row in records}), 2)
        serialized = json.dumps(records)
        self.assertNotIn("private", serialized)
        self.assertNotIn("verdicts", serialized)

    def test_transient_retry_preserves_failed_attempt_and_unknown_cost(self):
        class RateLimitError(Exception):
            status_code = 429
        sdk = MagicMock()
        sdk.with_options.return_value.chat.completions.create.side_effect = [
            RateLimitError("private credential in provider message"), response(),
        ]
        records = []
        with patch("rag_textbook_qa.evaluation.generation_runner.with_retries",
                   side_effect=lambda call: with_retries(call, sleep=lambda _: None)):
            generated = openai_generator(sdk, "model", max_tokens=20, on_call=records.append)("prompt", 0.7)
        self.assertEqual([row["status"] for row in records], ["error", "response"])
        self.assertEqual(records[0]["http_status"], 429)
        self.assertIsNone(records[0]["tokens"])
        self.assertIsNone(records[1]["tokens"])
        self.assertIsNone(generated["tokens"])
        self.assertEqual(generated["usage_record_version"], 1)
        self.assertNotIn("private", json.dumps(records))

    def test_partial_or_invalid_provider_counts_remain_unknown_in_both_outputs(self):
        for prompt, completion in ((10, None), (True, -1), ("10", 4), (0, 0)):
            with self.subTest(prompt=prompt, completion=completion):
                sdk = MagicMock()
                sdk.with_options.return_value.chat.completions.create.return_value = response(
                    usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
                )
                records = []
                generated = openai_generator(sdk, "model", max_tokens=20,
                                             on_call=records.append)("prompt", .7)
                expected = {"prompt": prompt if type(prompt) is int and prompt >= 0 else None,
                            "completion": completion if type(completion) is int and completion >= 0 else None}
                self.assertEqual(generated["tokens"], expected)
                self.assertEqual({key:records[0]["tokens"][key] for key in expected}, expected)

    def test_truncated_json_is_not_judged_but_its_cost_is_recorded(self):
        sdk = MagicMock()
        sdk.with_options.return_value.chat.completions.create.return_value = response("length")
        records = []
        with self.assertRaises(LLMGenerationIncompleteError):
            openai_judge(sdk, "judge", extra={}, on_call=records.append)("prompt")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["finish_reason"], "length")
        self.assertEqual(sdk.with_options.return_value.chat.completions.create.call_count, 1)

    def test_accounting_failure_does_not_retry_a_completed_paid_response(self):
        sdk = MagicMock()
        sdk.with_options.return_value.chat.completions.create.return_value = response()
        observer = MagicMock(side_effect=OSError("disk full"))
        with self.assertRaises(CallUsageRecordingError):
            openai_generator(sdk, "model", max_tokens=20, on_call=observer)("prompt", 0.7)
        self.assertEqual(sdk.with_options.return_value.chat.completions.create.call_count, 1)

    def test_concurrent_append_retains_existing_records_and_damaged_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"calls.jsonl"
            original = b'{"call_id":"old"}\n{"damaged": "\xe4'
            path.write_bytes(original)
            log = CallUsageLog(path)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(log, ({"call_id":str(i)} for i in range(30))))
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(original+b"\n"))
            rows = [json.loads(line) for line in raw.splitlines() if b"damaged" not in line]
            self.assertEqual({row["call_id"] for row in rows}, {"old",*(str(i) for i in range(30))})
