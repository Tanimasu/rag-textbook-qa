import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_textbook_qa.api.app import public_retrieval
from rag_textbook_qa.llm import GenerationCancelled
from rag_textbook_qa.rag.engine import RAGEngine


class RetrievalInspectionTests(unittest.TestCase):
    def engine(self):
        engine = object.__new__(RAGEngine)
        engine.verbose = False
        engine.vectorizer = SimpleNamespace()
        engine.llm = MagicMock()
        engine.fusion_weights = {"embedding": 1, "bm25": .5}
        engine.reranker = SimpleNamespace(rerank=MagicMock(return_value=[.1, .9]))
        engine.context_budget = 4000
        a = {"chunk_id": "a_p1", "book_name": "os", "content": "进程是程序的一次执行过程。", "rank": 1}
        b = {"chunk_id": "b_p2", "book_name": "os", "content": "程序是静态代码，进程是动态执行。", "rank": 2}
        engine.search_embedding = MagicMock(return_value=[a, b])
        engine.search_bm25 = MagicMock(return_value=[{**b, "rank": 1}, {**a, "rank": 2}])
        return engine

    def test_stages_record_rank_changes_and_actual_context_without_generation(self):
        engine = self.engine()
        result = engine.inspect_retrieval("什么是进程", "os", top_k=2)
        stages = result["stages"]
        self.assertEqual([row["chunk_id"] for row in stages["embedding"]], ["a_p1", "b_p2"])
        self.assertEqual([row["chunk_id"] for row in stages["keyword"]], ["b_p2", "a_p1"])
        self.assertEqual([row["chunk_id"] for row in stages["fusion"]], ["a_p1", "b_p2"])
        self.assertEqual([row["chunk_id"] for row in stages["rerank"]], ["b_p2", "a_p1"])
        self.assertEqual([row["chunk_id"] for row in stages["context"]], ["b_p2", "a_p1"])
        self.assertNotIn("rerank_score", stages["fusion"][0])
        engine.search_embedding.assert_called_once_with("os", "什么是进程", 18, use_hyde=False)
        engine.llm.assert_not_called()
        self.assertFalse(engine.llm.mock_calls)

    def test_disabled_reranker_and_cancellation_are_explicit(self):
        engine = self.engine()
        engine.reranker = None
        result = engine.inspect_retrieval("进程", "os", top_k=2)
        self.assertFalse(result["reranker_enabled"])
        self.assertEqual(result["stages"]["fusion"], result["stages"]["rerank"])
        engine.search_embedding.reset_mock()
        with self.assertRaises(GenerationCancelled):
            engine.inspect_retrieval("进程", "os", should_stop=lambda: True)
        engine.search_embedding.assert_not_called()

    def test_public_stages_bound_payload_and_remove_internal_and_nonfinite_fields(self):
        row = {"book_name": "os", "chunk_id": "a_p1", "content": "x" * 13000,
               "rerank_score": float("nan"), "rrf_score": .5, "prompt": "secret",
               "source_ranks": {"bm25": 2, "embedding": True, "secret": "secret"}}
        payload = public_retrieval({"stages": {"keyword": [row] * 40}, "elapsed_seconds": float("inf")})
        stage = payload["stages"][0]
        self.assertEqual(stage["count"], 40)
        self.assertEqual(len(stage["items"]), 30)
        self.assertEqual(len(stage["items"][0]["excerpt"]), 12000)
        self.assertEqual(stage["items"][0]["scores"], {"rrf_score": .5})
        self.assertEqual(stage["items"][0]["source_ranks"], {"bm25": 2})
        self.assertIsNone(payload["elapsed_seconds"])
        self.assertNotIn("secret", json.dumps(payload, allow_nan=False))
