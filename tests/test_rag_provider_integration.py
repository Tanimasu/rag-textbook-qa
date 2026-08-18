import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.providers import ModelIdentity
from rag_textbook_qa.providers.base import DEFAULT_QUERY_INSTRUCTION

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))
from rag_engine import RAGEngine


class FakeEmbeddingProvider:
    identity = ModelIdentity(
        task="embedding",
        model="fake-embedding",
        normalized=True,
        query_instruction=DEFAULT_QUERY_INSTRUCTION,
    )

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_queries(self, texts):
        return [[0.0, 1.0] for _ in texts]


class FakeRerankerProvider:
    identity = ModelIdentity(task="reranker", model="fake-reranker")

    def rerank(self, query, documents):
        return [float(index) for index, _ in enumerate(documents)]


class RagProviderIntegrationTests(unittest.TestCase):
    def test_engine_accepts_injected_providers_without_model_runtime(self):
        before = {"sentence_transformers", "torch"}.intersection(sys.modules)
        with (
            patch.dict("os.environ", {"RAG_QA_COMPUTE_BACKEND": "invalid"}),
            tempfile.TemporaryDirectory() as temporary_directory,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            engine = RAGEngine(
                db_path=temporary_directory,
                enable_llm=False,
                embedding_provider=FakeEmbeddingProvider(),
                reranker_provider=FakeRerankerProvider(),
            )
            results = engine._rerank(
                "query",
                [{"content": "first"}, {"content": "second"}],
                top_k=1,
            )

        after = {"sentence_transformers", "torch"}.intersection(sys.modules)
        self.assertEqual(after, before)
        self.assertEqual(results[0]["content"], "second")
        self.assertEqual(results[0]["rerank_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
