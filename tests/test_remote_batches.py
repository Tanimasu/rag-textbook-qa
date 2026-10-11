import unittest
import warnings
from unittest.mock import MagicMock, patch

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`.*")
from fastapi.testclient import TestClient
from test_multibook_retrieval import engine_for, row

from rag_textbook_qa.providers import (
    ModelIdentity,
    ModelMismatchError,
    ProviderProtocolError,
    TransientProviderError,
)
from rag_textbook_qa.providers.base import DEFAULT_QUERY_INSTRUCTION
from rag_textbook_qa.providers.remote import (
    FallbackRerankerProvider,
    RemoteEmbeddingProvider,
    RemoteRerankerProvider,
)
from rag_textbook_qa.worker import WorkerRuntime, create_worker_app


class WorkerClient:
    """Exercise the actual worker routes and validation with model-free providers."""

    def __init__(self):
        self.embedding = MagicMock()
        self.embedding.identity = ModelIdentity(
            task="embedding", model="embedding-model", normalized=True,
            query_instruction=DEFAULT_QUERY_INSTRUCTION,
        )
        self.embedding.embed_documents.side_effect = lambda texts: [[float(len(text)), 1.] for text in texts]
        self.embedding.embed_queries.side_effect = self.embedding.embed_documents.side_effect
        self.reranker = MagicMock()
        self.reranker.identity = ModelIdentity(task="reranker", model="rerank-model")
        self.reranker.rerank.side_effect = lambda query, documents: [
            float(document.removeprefix("正文")) if document.startswith("正文") else float(len(document))
            for document in documents
        ]
        self.http = TestClient(create_worker_app(WorkerRuntime(
            self.embedding, self.reranker, token=None, device="cpu",
        )))
        self.requests = []

    def request(self, path, *, method="GET", payload=None):
        self.requests.append((path, method, payload))
        response = self.http.request(method, path, json=payload)
        if response.status_code != 200:
            raise ProviderProtocolError(f"worker returned {response.status_code}")
        return response.json()


class RemoteBatchTests(unittest.TestCase):
    def client(self):
        client = WorkerClient()
        self.addCleanup(client.http.close)
        return client

    def test_150_scores_are_merged_in_the_original_document_order(self):
        client = self.client()
        provider = RemoteRerankerProvider(client, "rerank-model")
        documents = ["d" * (index + 1) for index in range(150)]
        self.assertEqual(provider.rerank("question", documents), list(range(1, 151)))
        batches = [request[2]["documents"] for request in client.requests if request[0] == "/v1/rerank"]
        self.assertEqual([len(batch) for batch in batches], [128, 22])
        self.assertEqual([document for batch in batches for document in batch], documents)
        self.assertEqual(len(provider.telemetry.since(0)), 1)
        self.assertEqual([request[0] for request in client.requests].count("/health"), 1)

    def test_five_book_global_top_ten_fits_the_worker_contract(self):
        client = self.client()
        engine = engine_for(books=("a", "b", "c", "d", "e"))
        data = {book: [row(book, str(index), 150 - book_index * 30 - index)
                       for index in range(30)]
                for book_index, book in enumerate(("a", "b", "c", "d", "e"))}
        engine.search_embedding.side_effect = lambda book, query, k, **kwargs: data[book][:k]
        engine.search_bm25.side_effect = lambda book, query, k: data[book][:k]
        engine.reranker = RemoteRerankerProvider(client, "rerank-model")
        results, _, _ = engine._retrieve("question", {"status": "disabled"}, None, 10, False)
        self.assertEqual([(result["book_name"], result["chunk_id"]) for result in results],
                         [("a", str(index)) for index in range(10)])
        self.assertEqual(client.reranker.rerank.call_count, 2)

    def test_embedding_batches_preserve_both_order_and_input_type(self):
        client = self.client()
        provider = RemoteEmbeddingProvider(client, "embedding-model")
        texts = ["q" * (index + 1) for index in range(257)]
        expected = [[float(len(text)), 1.] for text in texts]
        self.assertEqual(provider.embed_queries(texts), expected)
        posts = [request[2] for request in client.requests if request[0] == "/v1/embeddings"]
        self.assertEqual([len(post["texts"]) for post in posts], [128, 128, 1])
        self.assertTrue(all(post["input_type"] == "query" for post in posts))
        self.assertEqual(provider.embed_documents(["document"]), [[8., 1.]])
        self.assertEqual(client.requests[-1][2]["input_type"], "document")

    def test_character_limit_splits_whole_documents(self):
        client = self.client()
        documents = ["a" * 130_000, "b" * 130_000]
        provider = RemoteRerankerProvider(client, "rerank-model")
        self.assertEqual(provider.rerank("question", documents), [130_000., 130_000.])
        self.assertEqual([call.args[1] for call in client.reranker.rerank.call_args_list],
                         [[documents[0]], [documents[1]]])

    def test_oversized_single_input_is_rejected_before_any_worker_request(self):
        client = self.client()
        for provider, call in (
            (RemoteRerankerProvider(client, "rerank-model"), lambda provider, values: provider.rerank("q", values)),
            (RemoteEmbeddingProvider(client, "embedding-model"), lambda provider, values: provider.embed_documents(values)),
        ):
            with self.subTest(task=provider.identity.task), self.assertRaises(ProviderProtocolError):
                call(provider, ["valid", "x" * 250_001])
        self.assertEqual(client.requests, [])

    def test_different_embedding_dimensions_across_batches_fail_the_logical_call(self):
        client = self.client()
        client.embedding.embed_documents.side_effect = lambda texts: [
            [1., 2.] if len(texts) > 1 else [1., 2., 3.] for _ in texts
        ]
        provider = RemoteEmbeddingProvider(client, "embedding-model")
        with self.assertRaisesRegex(ProviderProtocolError, "维度不一致"):
            provider.embed_documents(["document"] * 129)
        self.assertFalse(provider.telemetry.since(0)[-1].success)

    def test_later_batch_failure_falls_back_for_the_whole_input(self):
        client = self.client()
        primary = RemoteRerankerProvider(client, "rerank-model")
        fallback = MagicMock()
        fallback.identity = primary.identity
        fallback.rerank.return_value = [7.] * 150
        provider = FallbackRerankerProvider(primary, fallback)
        request = client.request

        def fail_second(path, **kwargs):
            if path == "/v1/rerank" and client.reranker.rerank.call_count:
                raise TransientProviderError("offline")
            return request(path, **kwargs)

        documents = ["document"] * 150
        with patch.object(client, "request", side_effect=fail_second):
            self.assertEqual(provider.rerank("q", documents), [7.] * 150)
        fallback.rerank.assert_called_once_with("q", documents)
        self.assertFalse(primary.telemetry.since(0)[-1].success)
        self.assertTrue(provider.telemetry.since(0)[-1].fallback_used)

    def test_model_change_in_a_later_batch_fails_without_local_fallback(self):
        client = self.client()
        primary = RemoteRerankerProvider(client, "rerank-model")
        fallback = MagicMock()
        fallback.identity = primary.identity
        provider = FallbackRerankerProvider(primary, fallback)
        request = client.request

        def change_second(path, **kwargs):
            response = request(path, **kwargs)
            if path == "/v1/rerank" and client.reranker.rerank.call_count == 2:
                response["fingerprint"] = "another-model"
            return response

        with patch.object(client, "request", side_effect=change_second), self.assertRaises(ModelMismatchError):
            provider.rerank("q", ["document"] * 150)
        fallback.rerank.assert_not_called()
