"""Exercise installed eval dependencies with isolated HTTP, without model calls."""

import asyncio
import contextlib
import importlib.util
import io
import json
import os
import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

EVAL_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in (
    "ragas", "datasets", "langchain_openai",
))


@unittest.skipUnless(EVAL_AVAILABLE, "RAGAS eval extra is not installed")
class RagasTransportTests(unittest.TestCase):
    def test_interrupted_caller_cancels_pending_work_before_close(self):
        self.enterContext(patch.dict(os.environ, {
            "RAGAS_DO_NOT_TRACK": "true", "LANGCHAIN_TRACING_V2": "false",
        }))
        from rag_textbook_qa.evaluation.ragas import RAGASEvaluator

        started, cancelled, release = threading.Event(), threading.Event(), threading.Event()

        async def pending():
            started.set()
            try:
                while not release.is_set():
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        def interrupt(*args, **kwargs):
            self.assertTrue(started.wait(5))
            raise KeyboardInterrupt

        with contextlib.redirect_stdout(io.StringIO()):
            evaluator = RAGASEvaluator(api_key="isolated-test-key",
                                       base_url="https://isolated.invalid/v1", model="isolated-model")
        try:
            with patch.object(Future, "result", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                evaluator._run_async(pending)
            self.assertTrue(cancelled.wait(5), "interrupted scoring remains active")
        finally:
            release.set()
            evaluator.close()

    def test_successful_relevancy_repeats_use_chat_and_embeddings_then_close(self):
        self.enterContext(patch.dict(os.environ, {
            "RAGAS_DO_NOT_TRACK": "true", "LANGCHAIN_TRACING_V2": "false",
            "RAGAS_RELEVANCY_SAMPLES": "3", "RAGAS_DISABLE_THINKING": "false",
            "RAGAS_EMBEDDING_MODEL": "isolated-embedding",
        }))
        import httpx
        from datasets import Dataset

        from rag_textbook_qa.evaluation.ragas import RAGASEvaluator

        requests = []

        def response(request):
            self.assertEqual(request.url.host, "isolated.invalid")
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(200, json={"object": "list", "model": body["model"],
                    "data": [{"object": "embedding", "index": index, "embedding": [1.0, 0.0]}
                             for index, _ in enumerate(body["input"])],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1}})
            return httpx.Response(200, json={"id": "isolated", "object": "chat.completion",
                "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": json.dumps({
                        "question": "isolated question", "noncommittal": 0,
                    })}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

        sync_type, async_type = httpx.Client, httpx.AsyncClient
        with (
            patch("rag_textbook_qa.evaluation.ragas.httpx", SimpleNamespace(
                Client=lambda: sync_type(transport=httpx.MockTransport(response)),
                AsyncClient=lambda: async_type(transport=httpx.MockTransport(response)),
            )),
            contextlib.redirect_stdout(io.StringIO()),
            RAGASEvaluator(api_key="isolated-test-key", base_url="https://isolated.invalid/v1",
                           model="isolated-model") as evaluator,
        ):
            dataset = Dataset.from_dict({"question": ["isolated question"],
                                         "answer": ["isolated answer"],
                                         "contexts": [["isolated evidence"]]})
            result = evaluator.evaluate(dataset, metrics=[evaluator._answer_relevancy])
            self.assertAlmostEqual(result.to_pandas()["answer_relevancy"][0], 1.0)
        chat = [body for path, body in requests if path.endswith("/chat/completions")]
        embeddings = [body for path, body in requests if path.endswith("/embeddings")]
        self.assertEqual(len(chat), 3)
        self.assertEqual(len(embeddings), 6)
        self.assertTrue(all(body["model"] == "isolated-model" for body in chat))
        self.assertTrue(all(body["model"] == "isolated-embedding" for body in embeddings))
        self.assertTrue(evaluator._http_client.is_closed)
        self.assertTrue(evaluator._http_async_client.is_closed)

    def test_repeated_scoring_reuses_connections_and_closes_them_on_their_loop(self):
        self.enterContext(patch.dict(os.environ, {
            "RAGAS_DO_NOT_TRACK": "true", "LANGCHAIN_TRACING_V2": "false",
            "RAGAS_RELEVANCY_SAMPLES": "1", "RAGAS_DISABLE_THINKING": "false",
        }))
        from datasets import Dataset

        from rag_textbook_qa.evaluation.ragas import RAGASEvaluator

        requests = []
        disconnected = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                requests.append((json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                                 self.client_address[1]))
                body = b'{"error":{"message":"isolated credentials","type":"test_error"}}'
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def finish(self):
                super().finish()
                disconnected.set()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with RAGASEvaluator(api_key="isolated-test-key",
                                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                    model="isolated-model") as evaluator:
                    evaluator.llm.root_async_client._calculate_retry_timeout = lambda *args: 0
                    dataset = Dataset.from_dict({"question": ["isolated question"],
                                                 "answer": ["isolated answer"],
                                                 "contexts": [["isolated evidence"]]})

                    def score():
                        result = evaluator.evaluate(dataset, metrics=[evaluator._faithfulness])
                        self.assertTrue(result.to_pandas()["faithfulness"].isna().all())

                    async def score_inside_async_caller():
                        score()
                        score()

                    score()
                    asyncio.run(score_inside_async_caller())
                    score()
                    score()
                    self.assertEqual(len(requests), 5)
                    self.assertEqual(len({port for _, port in requests}), 1)
                    self.assertTrue(all(body["model"] == "isolated-model" for body, _ in requests))
                self.assertTrue(disconnected.wait(5), "judge connection remains open after close")
                self.assertTrue(evaluator._http_client.is_closed)
                self.assertTrue(evaluator._http_async_client.is_closed)
                evaluator.close()
                with self.assertRaisesRegex(RuntimeError, "已关闭"):
                    score()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def assert_parallel_clients(self, metric_attribute):
        self.enterContext(patch.dict(os.environ, {
            "RAGAS_DO_NOT_TRACK": "true", "LANGCHAIN_TRACING_V2": "false",
            "RAGAS_RELEVANCY_SAMPLES": "1", "RAGAS_DISABLE_THINKING": "false",
        }))
        import httpx
        import langchain_openai
        from datasets import Dataset

        from rag_textbook_qa.evaluation.ragas import RAGASEvaluator

        first_started = threading.Event()
        second_reached = threading.Event()
        requests = []
        clients = []
        evaluators = []
        original = langchain_openai.ChatOpenAI

        def unavailable(request):
            requests.append((threading.get_ident(), request.url.host,
                             json.loads(request.content)["model"]))
            if threading.get_ident() == requests[0][0]:
                first_started.set()
                if not second_reached.wait(10):
                    raise AssertionError("second evaluation did not reach its judge")
            else:
                second_reached.set()
            return httpx.Response(401, json={"error": {
                "message": "isolated credentials", "type": "test_error", "code": "test_error",
            }})

        def isolated_chat(**kwargs):
            sync_http = httpx.Client(transport=httpx.MockTransport(unavailable))
            async_http = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
            clients.append((sync_http, async_http))
            kwargs.update(http_async_client=async_http, http_client=sync_http)
            return original(**kwargs)

        def evaluate(evaluator):
            dataset = Dataset.from_dict({"question": ["isolated question"],
                                         "answer": ["isolated answer"],
                                         "contexts": [["isolated evidence"]],
                                         "ground_truth": ["isolated reference"]})
            return evaluator.evaluate(dataset, metrics=[getattr(evaluator, metric_attribute)])

        try:
            with (
                patch.object(langchain_openai, "ChatOpenAI", side_effect=isolated_chat),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                for name in ("first", "second"):
                    evaluators.append(RAGASEvaluator(api_key=f"isolated-{name}-key",
                                                     base_url=f"https://{name}.invalid/v1",
                                                     model=f"{name}-model"))
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="isolated") as pool:
                    first = pool.submit(evaluate, evaluators[0])
                    self.assertTrue(first_started.wait(10))
                    second = pool.submit(evaluate, evaluators[1])
                    results = [first.result(timeout=15), second.result(timeout=15)]
            metric_name = getattr(evaluators[0], metric_attribute).name
            self.assertTrue(all(result.to_pandas()[metric_name].isna().all() for result in results))
            self.assertEqual(sorted((host, model) for _, host, model in requests), [
                ("first.invalid", "first-model"),
                ("second.invalid", "second-model"),
            ])
        finally:
            second_reached.set()
            for evaluator in evaluators:
                evaluator.close()
            for sync_http, async_http in clients:
                sync_http.close()
                asyncio.run(async_http.aclose())

    def test_parallel_faithfulness_uses_each_runs_judge(self):
        self.assert_parallel_clients("_faithfulness")

    def test_parallel_context_precision_uses_each_runs_judge(self):
        self.assert_parallel_clients("_context_precision")

    def test_parallel_context_recall_uses_each_runs_judge(self):
        self.assert_parallel_clients("_context_recall")

    def assert_failed_metric_attempts(self, status, expected, *, relevancy_samples=1,
                                      default_metrics=False):
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
            kwargs.update(http_async_client=async_http, http_client=sync_http)
            model = original(**kwargs)
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
                result = evaluator.evaluate(dataset, metrics=None if default_metrics else [metric])
            self.assertTrue(result.to_pandas()[metric.name].isna().all())
            self.assertEqual(len(requests), expected)
            self.assertTrue(all(request["model"] == "isolated-model" for request in requests))
        finally:
            if evaluator is not None:
                evaluator.close()
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

    def test_default_metrics_accept_questions_without_reference(self):
        self.assert_failed_metric_attempts(401, 2, default_metrics=True)


if __name__ == "__main__":
    unittest.main()
