import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag_textbook_qa.llm import (
    GenerationCancelled,
    LLMClient,
    LLMConfigurationError,
    LLMGenerationIncompleteError,
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


class FakeStream:
    """An SDK stream that records how far it was read and whether it was closed."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.pulled = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.pulled += 1
            yield chunk

    def close(self):
        self.closed = True


def reasoning_chunk():
    delta = SimpleNamespace(content=None, reasoning_content="思考")
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])


def answer_chunk(text, finish_reason=None):
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


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
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")]
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

    def test_generate_answer_marks_a_length_limited_answer_as_incomplete(self):
        sdk = FakeSDKClient([completion_response("未完成")])
        sdk.completions.outcomes[0].choices[0].finish_reason = "length"
        client = LLMClient(
            api_key="key",
            base_url="https://llm.example/v1",
            sdk_client=sdk,
            verbose=False,
        )

        result = client.generate_answer("问题", retry=0)

        self.assertFalse(result["success"])
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["answer"], "未完成")
        self.assertIn("长度上限", result["error"])

    def test_stream_rejects_a_length_limited_terminal_chunk(self):
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="未完成"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="length")]
            ),
        ]
        client = LLMClient(
            api_key="key",
            base_url="https://llm.example/v1",
            sdk_client=FakeSDKClient([chunks]),
            verbose=False,
        )

        with self.assertRaisesRegex(LLMGenerationIncompleteError, "长度上限"):
            list(client.stream_answer("问题", raise_on_error=True))

    def test_a_stop_during_hidden_reasoning_ends_and_closes_the_request(self):
        stream = FakeStream(
            [reasoning_chunk() for _ in range(20)] + [answer_chunk("答案", "stop")]
        )
        client = LLMClient(
            api_key="key",
            base_url="https://llm.example/v1",
            sdk_client=FakeSDKClient([stream]),
            verbose=False,
        )

        yielded = []
        # raise_on_error stays False: a stop must not come back as an error message.
        with self.assertRaises(GenerationCancelled) as caught:
            yielded.extend(client.stream_answer("问题", should_stop=lambda: stream.pulled >= 3))

        self.assertTrue(caught.exception.request_sent)
        self.assertEqual(yielded, [])
        self.assertEqual(stream.pulled, 3)
        self.assertTrue(stream.closed)

    def test_a_stop_before_the_request_sends_nothing(self):
        sdk = FakeSDKClient([])
        client = LLMClient(
            api_key="key",
            base_url="https://llm.example/v1",
            sdk_client=sdk,
            verbose=False,
        )

        with self.assertRaises(GenerationCancelled) as caught:
            list(client.stream_answer("问题", should_stop=lambda: True))

        self.assertFalse(caught.exception.request_sent)
        self.assertEqual(sdk.completions.calls, [])

    def test_a_finished_stream_is_closed(self):
        stream = FakeStream([answer_chunk("A"), answer_chunk("B", "stop")])
        client = LLMClient(
            api_key="key",
            base_url="https://llm.example/v1",
            sdk_client=FakeSDKClient([stream]),
            verbose=False,
        )

        self.assertEqual(list(client.stream_answer("问题", should_stop=lambda: False)), ["A", "B"])
        self.assertTrue(stream.closed)

    def test_stream_can_raise_errors_for_engine_handling(self):
        client = LLMClient(
            api_key="key",
            base_url="https://llm.example/v1",
            sdk_client=FakeSDKClient([RuntimeError("stream failed")]),
            verbose=False,
        )

        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            list(client.stream_answer("问题", raise_on_error=True))


if __name__ == "__main__":
    unittest.main()
