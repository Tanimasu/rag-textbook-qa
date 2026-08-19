import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag_textbook_qa.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMSettings,
    create_llm_client,
)


def completion_response(content="answer"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=4,
            total_tokens=7,
        ),
    )


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeSDKClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


class LLMClientTests(unittest.TestCase):
    def test_settings_are_resolved_explicitly_at_factory_call_time(self):
        environment = {
            "LLM_API_KEY": "test-key",
            "LLM_API_BASE": "https://llm.example/v1",
            "LLM_MODEL": "test-model",
        }
        sdk = FakeSDKClient([completion_response()])

        client = create_llm_client(
            environ=environment,
            sdk_client=sdk,
            verbose=False,
        )

        self.assertEqual(client.base_url, "https://llm.example/v1/")
        self.assertEqual(client.default_model, "test-model")
        self.assertNotIn("test-key", repr(client.__dict__))
        self.assertNotIn("test-key", repr(LLMSettings.from_env(environment)))

    def test_invalid_configuration_fails_before_sdk_creation(self):
        with self.assertRaisesRegex(LLMConfigurationError, "LLM_API_KEY"):
            create_llm_client(environ={}, verbose=False)
        with self.assertRaisesRegex(LLMConfigurationError, "首尾空白"):
            LLMClient(
                api_key=" key ",
                base_url="https://llm.example/v1",
                sdk_client=FakeSDKClient([]),
                verbose=False,
            )
        with self.assertRaisesRegex(LLMConfigurationError, "凭据"):
            LLMClient(
                api_key="key",
                base_url="https://user:pass@llm.example/v1",
                sdk_client=FakeSDKClient([]),
                verbose=False,
            )

    def test_generate_answer_preserves_contract_and_retries_sdk_errors(self):
        sdk = FakeSDKClient([RuntimeError("temporary"), completion_response("回答")])
        client = LLMClient(
            api_key="key",
            base_url="https://llm.example/v1",
            model="test-model",
            sdk_client=sdk,
            verbose=False,
        )

        with (
            patch("rag_textbook_qa.llm.client.time.sleep") as sleep,
            patch(
                "rag_textbook_qa.llm.client.time.monotonic",
                side_effect=[1.0, 2.0, 2.25],
            ),
        ):
            result = client.generate_answer("问题", retry=1)

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "回答")
        self.assertEqual(result["tokens"]["total"], 7)
        self.assertEqual(result["time"], 0.25)
        self.assertEqual(len(sdk.completions.calls), 2)
        self.assertFalse(sdk.completions.calls[-1]["stream"])
        sleep.assert_called_once_with(1)

    def test_stream_skips_empty_deltas_without_network(self):
        chunks = [
            SimpleNamespace(choices=[]),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="A"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="B"))]
            ),
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            client = LLMClient(
                api_key="key",
                base_url="https://llm.example/v1",
                sdk_client=FakeSDKClient([chunks]),
                verbose=True,
            )
            result = list(client.stream_answer("问题"))

        self.assertEqual(result, ["A", "B"])
        self.assertNotIn("key", output.getvalue())


if __name__ == "__main__":
    unittest.main()
