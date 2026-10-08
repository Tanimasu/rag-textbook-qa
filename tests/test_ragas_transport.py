"""Exercise installed eval dependencies with isolated HTTP, without model calls."""

import asyncio
import contextlib
import importlib.util
import io
import json
import os
import unittest
from unittest.mock import patch

EVAL_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in (
    "ragas", "datasets", "langchain_openai",
))


@unittest.skipUnless(EVAL_AVAILABLE, "RAGAS eval extra is not installed")
class RagasTransportTests(unittest.TestCase):
    def assert_failed_metric_attempts(self, status, expected, *, relevancy_samples=1):
        self.enterContext(patch.dict(os.environ, {
            "RAGAS_DO_NOT_TRACK": "true", "LANGCHAIN_TRACING_V2": "false",
            "RAGAS_RELEVANCY_SAMPLES": str(relevancy_samples), "RAGAS_DISABLE_THINKING": "false",
            "RAGAS_EMBEDDING_MODEL": "isolated-embedding",
        }))
        import httpx
        import langchain_openai
        import ragas.run_config
        import tenacity
        from datasets import Dataset

        from rag_textbook_qa.evaluation.ragas import RAGASEvaluator

        requests = []

        def unavailable(request):
            self.assertEqual(request.url.host, "isolated.invalid")
            requests.append(json.loads(request.content))
            return httpx.Response(status, json={"error": {
                "message": "isolated provider failure", "type": "test_error", "code": "test_error",
            }})

        async_http = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
        sync_http = httpx.Client(transport=httpx.MockTransport(unavailable))
        original = langchain_openai.ChatOpenAI

        def isolated_chat(**kwargs):
            model = original(**kwargs, http_async_client=async_http, http_client=sync_http)
            # Keep real SDK retry handling but remove sleeping from the test.
            model.root_async_client._calculate_retry_timeout = lambda *args: 0
            model.root_client._calculate_retry_timeout = lambda *args: 0
            return model

        evaluator = None
        try:
            with (
                patch.object(langchain_openai, "ChatOpenAI", side_effect=isolated_chat),
                patch.object(ragas.run_config, "wait_random_exponential",
                             side_effect=lambda **kwargs: tenacity.wait_none()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                evaluator = RAGASEvaluator(api_key="isolated-test-key",
                                           base_url="https://isolated.invalid/v1", model="isolated-model")
                dataset = Dataset.from_dict({"question": ["isolated question"],
                                             "answer": ["isolated answer"],
                                             "contexts": [["isolated evidence"]]})
                metric = evaluator._faithfulness if relevancy_samples == 1 else evaluator._answer_relevancy
                result = evaluator.evaluate(dataset, metrics=[metric])
            self.assertTrue(result.to_pandas()[metric.name].isna().all())
            self.assertEqual(len(requests), expected)
            self.assertTrue(all(request["model"] == "isolated-model" for request in requests))
        finally:
            if evaluator is not None:
                evaluator.llm.root_client.close()
                # Embeddings are constructed but unused in this faithfulness job.
                evaluator.embeddings.client._client.close()
                asyncio.run(evaluator.embeddings.async_client._client.close())
            asyncio.run(async_http.aclose())
            sync_http.close()

    def test_rate_limit_has_one_retry_owner(self):
        # Three SDK retries plus the first request; RAGAS must not multiply them.
        self.assert_failed_metric_attempts(429, 4)

    def test_server_failure_still_uses_sdk_recovery(self):
        self.assert_failed_metric_attempts(503, 4)

    def test_relevancy_repeats_keep_the_same_retry_budget(self):
        self.assert_failed_metric_attempts(429, 12, relevancy_samples=3)

    def test_invalid_credentials_are_not_retried(self):
        self.assert_failed_metric_attempts(401, 1)


if __name__ == "__main__":
    unittest.main()
