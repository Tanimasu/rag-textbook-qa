import platform
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`.*")
from fastapi.testclient import TestClient

from rag_textbook_qa.providers import (
    AuthenticationError,
    MissingOptionalDependencyError,
    ModelIdentity,
    ModelMismatchError,
)
from rag_textbook_qa.providers.base import DEFAULT_QUERY_INSTRUCTION
from rag_textbook_qa.worker import (
    WorkerRuntime,
    create_worker_app,
    run_worker_server,
    validate_worker_bind,
)


class FakeEmbeddingProvider:
    identity = ModelIdentity(
        task="embedding",
        model="embed-model",
        normalized=True,
        query_instruction=DEFAULT_QUERY_INSTRUCTION,
    )

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def embed_queries(self, texts):
        self.query_calls.append(list(texts))
        return [[0.0, 1.0] for _ in texts]


class FakeRerankerProvider:
    identity = ModelIdentity(task="reranker", model="rerank-model")

    def __init__(self):
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return [float(len(document)) for document in documents]


class WorkerRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = WorkerRuntime(
            FakeEmbeddingProvider(),
            FakeRerankerProvider(),
            token="worker-secret",
            device="cuda",
        )

    def test_non_loopback_bind_requires_token(self):
        with self.assertRaises(AuthenticationError):
            validate_worker_bind("100.64.0.10", None)
        validate_worker_bind("127.0.0.1", None)
        validate_worker_bind("100.64.0.10", "token")

    def test_authorization_uses_bearer_token(self):
        with self.assertRaises(AuthenticationError):
            self.runtime.authorize(None)
        with self.assertRaises(AuthenticationError):
            self.runtime.authorize("Bearer wrong")
        self.runtime.authorize("Bearer worker-secret")

    def test_non_ascii_or_whitespace_tokens_fail_without_leaking_values(self):
        invalid_tokens = ("中文-token", " leading", "trailing ", "two words")
        for token in invalid_tokens:
            with self.subTest(token=token), self.assertRaises(
                AuthenticationError
            ) as raised:
                validate_worker_bind("100.64.0.10", token)
            self.assertNotIn(token, str(raised.exception))

        with self.assertRaises(AuthenticationError) as raised:
            self.runtime.authorize("Bearer 中文-token")
        self.assertNotIn("中文-token", str(raised.exception))

        with self.assertRaises(AuthenticationError):
            WorkerRuntime(
                FakeEmbeddingProvider(),
                FakeRerankerProvider(),
                token="中文-token",
                device="cuda",
            )

    def test_embedding_and_rerank_contracts(self):
        embedded = self.runtime.embeddings(
            {"model": "embed-model", "input_type": "query", "texts": ["问题"]}
        )
        reranked = self.runtime.rerank(
            {"model": "rerank-model", "query": "q", "documents": ["a", "long"]}
        )

        self.assertEqual(embedded["embeddings"], [[0.0, 1.0]])
        self.assertEqual(reranked["scores"], [1.0, 4.0])
        self.assertEqual(embedded["fingerprint"], FakeEmbeddingProvider.identity.fingerprint)

    def test_warmup_runs_one_minimal_inference_per_provider(self):
        timings = self.runtime.warmup()

        self.assertEqual(self.runtime.embedding_provider.query_calls, [["模型预热"]])
        self.assertEqual(
            self.runtime.reranker_provider.calls,
            [("模型预热", ["模型预热"])],
        )
        self.assertGreaterEqual(timings["embedding_seconds"], 0)
        self.assertGreaterEqual(timings["reranker_seconds"], 0)

    def test_model_mismatch_is_rejected_before_inference(self):
        with self.assertRaises(ModelMismatchError):
            self.runtime.embeddings({"model": "wrong", "input_type": "query", "texts": ["问题"]})

    def test_http_routes_enforce_auth_and_expose_the_contract(self):
        client = TestClient(create_worker_app(self.runtime))

        self.assertEqual(client.get("/health").status_code, 401)
        malformed = client.get(
            "/health",
            headers=[(b"authorization", b"Bearer \xff")],
        )
        headers = {"Authorization": "Bearer worker-secret"}
        health = client.get("/health", headers=headers)
        embedded = client.post(
            "/v1/embeddings",
            headers=headers,
            json={"model": "embed-model", "input_type": "document", "texts": ["正文"]},
        )
        mismatch = client.post(
            "/v1/rerank",
            headers=headers,
            json={"model": "wrong", "query": "q", "documents": ["doc"]},
        )

        self.assertEqual(malformed.status_code, 401)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["device"], "cuda")
        self.assertEqual(health.json()["platform"], platform.system())
        self.assertEqual(embedded.json()["embeddings"], [[1.0, 0.0]])
        self.assertEqual(mismatch.status_code, 409)

        with patch.object(
            self.runtime.embedding_provider,
            "embed_documents",
            side_effect=MissingOptionalDependencyError("install local-models"),
        ):
            missing_runtime = client.post(
                "/v1/embeddings",
                headers=headers,
                json={
                    "model": "embed-model",
                    "input_type": "document",
                    "texts": ["正文"],
                },
            )
        self.assertEqual(missing_runtime.status_code, 424)

    def test_server_warmup_is_opt_in(self):
        fake_uvicorn = SimpleNamespace(run=unittest.mock.Mock())
        with (
            patch.dict(sys.modules, {"uvicorn": fake_uvicorn}),
            patch("rag_textbook_qa.worker.WorkerRuntime") as runtime_class,
        ):
            runtime_class.return_value.warmup.return_value = {
                "embedding_seconds": 1.0,
                "reranker_seconds": 2.0,
            }
            run_worker_server(
                host="100.64.0.10",
                port=8765,
                embedding_model="embed-model",
                reranker_model="rerank-model",
                device="cuda",
                token="worker-secret",
                warmup=True,
            )

        runtime_class.return_value.warmup.assert_called_once_with()
        fake_uvicorn.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
