import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_textbook_qa.api.app import public_retrieval
from rag_textbook_qa.llm import GenerationCancelled
from rag_textbook_qa.rag.engine import RAGEngine


class RetrievalInspectionTests(unittest.TestCase):
    def multibook_engine(self, books=5, reranker=True):
        engine = self.engine()
        names = [f"book{index}" for index in range(books)]
        provider = SimpleNamespace(embed_queries=MagicMock(return_value=[[1., 0.]]),
                                   embed_documents=MagicMock())
        engine.vectorizer = SimpleNamespace(
            embedding_provider=provider,
            client=SimpleNamespace(list_collections=lambda: [SimpleNamespace(name="textbook_" + b)
                                                             for b in reversed(names)]),
            validate_collection_embedding=MagicMock(),
        )
        engine.enable_llm = engine.enable_hyde = True
        data = {book: [{"book_name": book, "chunk_id": f"shared_p{i}", "rank": i + 1,
                        "content": f"正文{book}|{100 - i if book == 'book0' else 20 - i}"}
                       for i in range(20)] for book in names}
        engine.search_embedding = MagicMock(side_effect=lambda b, q, k, **kw:
                                             [dict(row) for row in data[b]][:k])
        engine.search_bm25 = MagicMock(side_effect=lambda b, q, k: [dict(row) for row in data[b]][:k])
        engine.reranker = SimpleNamespace(rerank=MagicMock(
            side_effect=lambda q, docs: [float(doc.split("|")[-1]) for doc in docs]
        )) if reranker else None
        return engine

    def test_all_books_observes_the_product_route_and_reranks_once_globally(self):
        engine = self.multibook_engine()
        engine._retrieve = MagicMock(wraps=engine._retrieve)
        result = engine.inspect_retrieval("问题")
        engine._retrieve.assert_called_once()
        self.assertEqual(engine._retrieve.call_args.args, ("问题", {"status": "disabled"}, None, 5, False))
        self.assertEqual(result["scope"], "all")
        stages = result["stages"]
        self.assertEqual(len(stages["fusion"]), 75)
        self.assertEqual({row["book_name"] for row in stages["fusion"]},
                         {f"book{i}" for i in range(5)})
        self.assertEqual([row["book_name"] for row in stages["embedding"][:5]],
                         [f"book{i}" for i in range(5)])
        self.assertTrue(all(row["book_name"] == "book0" for row in stages["rerank"]))
        self.assertEqual(len(stages["context"]), 5)
        self.assertFalse(any("rerank_score" in row for row in stages["fusion"]))
        engine.vectorizer.embedding_provider.embed_queries.assert_called_once_with(["问题"])
        engine.vectorizer.embedding_provider.embed_documents.assert_not_called()
        self.assertEqual(engine.vectorizer.validate_collection_embedding.call_count, 5)
        self.assertEqual(engine.search_embedding.call_count, 5)
        self.assertEqual(engine.search_bm25.call_count, 5)
        self.assertTrue(all(call.args[2] == 45 and call.kwargs["use_hyde"] is False
                            for call in engine.search_embedding.call_args_list))
        self.assertTrue(all(call.args[2] == 45 for call in engine.search_bm25.call_args_list))
        engine.reranker.rerank.assert_called_once()
        self.assertEqual(len(engine.reranker.rerank.call_args.args[1]), 75)
        self.assertFalse(engine.llm.mock_calls)
        ordinary = self.multibook_engine()
        rows, _, _ = ordinary._retrieve("问题", {"status": "disabled"}, None, 5, False)
        self.assertEqual(rows, stages["rerank"])

    def test_all_books_without_reranker_or_any_books_remains_bounded(self):
        engine = self.multibook_engine(reranker=False)
        result = engine.inspect_retrieval("问题")
        self.assertFalse(result["reranker_enabled"])
        self.assertEqual(len(result["stages"]["fusion"]), 25)
        self.assertEqual(result["stages"]["rerank"], result["stages"]["fusion"][:5])
        self.assertTrue(all(call.args[2] == 15 for call in engine.search_embedding.call_args_list))
        empty = self.multibook_engine(books=0)
        result = empty.inspect_retrieval("问题")
        self.assertEqual(set(result["stages"]), {"keyword", "embedding", "fusion", "rerank", "context"})
        self.assertTrue(all(not rows for rows in result["stages"].values()))
        empty.vectorizer.embedding_provider.embed_queries.assert_not_called()
        empty.reranker.rerank.assert_not_called()

    def test_inspection_rejects_blank_books_and_mismatched_indexes_before_models(self):
        for book in ("", " ", True):
            with self.subTest(book=book), self.assertRaisesRegex(ValueError, "教材编号"):
                self.multibook_engine().inspect_retrieval("问题", book)
        engine = self.multibook_engine()
        engine.vectorizer.validate_collection_embedding.side_effect = ValueError("fingerprint")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            engine.inspect_retrieval("问题")
        engine.vectorizer.embedding_provider.embed_queries.assert_not_called()
        engine.reranker.rerank.assert_not_called()

    def test_all_books_checks_cancellation_after_shared_embedding_and_between_books(self):
        for stage in ("shared_embedding", "book_embedding", "keyword"):
            with self.subTest(stage=stage):
                engine = self.multibook_engine()
                stopped = [False]
                if stage == "shared_embedding":
                    target = engine.vectorizer.embedding_provider.embed_queries
                    original = lambda *args, **kwargs: [[1., 0.]]
                else:
                    target = engine.search_embedding if stage == "book_embedding" else engine.search_bm25
                    original = target.side_effect
                def stop(*args, _original=original, _stopped=stopped, **kwargs):
                    rows = _original(*args, **kwargs)
                    _stopped[0] = True
                    return rows
                target.side_effect = stop
                with self.assertRaises(GenerationCancelled):
                    engine.inspect_retrieval("问题", should_stop=lambda flag=stopped: flag[0])
                self.assertLessEqual(engine.search_embedding.call_count, 1)
                if stage != "keyword":
                    engine.search_bm25.assert_not_called()
                engine.reranker.rerank.assert_not_called()
                self.assertFalse(engine.llm.mock_calls)

    def test_all_book_payload_has_five_global_caps_instead_of_five_caps_per_book(self):
        for count in (5, 50):
            with self.subTest(books=count):
                result = public_retrieval(self.multibook_engine(books=count).inspect_retrieval("问题"))
                self.assertEqual(result["scope"], "all")
                self.assertEqual(len(result["stages"]), 5)
                self.assertEqual([len(stage["items"]) for stage in result["stages"]], [30, 30, 30, 5, 5])
                self.assertEqual(result["stages"][0]["count"], 20 * count)
                self.assertTrue(all(item["book_position"] > 0 for item in result["stages"][0]["items"]))
                self.assertLess(len(json.dumps(result)), 70_000)

    def test_public_inspection_caps_metadata_and_keeps_long_identities_distinct(self):
        row = {"book_name": "b" * 10000, "chunk_id": "c" * 10000, "content": "x" * 13000,
               "chapter": "h" * 10000, "citation_id": {"prompt": "private"},
               "rank": True, "source_ranks": {"bm25": 2**40}}
        other = {**row, "chunk_id": row["chunk_id"] + "d"}
        result = public_retrieval({"scope": "all", "stages": {"keyword": [row, other]},
                                   "context_budget": {"private": True}})
        first, second = result["stages"][0]["items"]
        self.assertLessEqual(len(first["book_id"]), 503)
        self.assertLessEqual(len(first["chunk_id"]), 512)
        self.assertLessEqual(len(first["section"]), 2000)
        self.assertNotEqual(first["chunk_id"], second["chunk_id"])
        self.assertTrue(first["metadata_truncated"])
        self.assertIsNone(first["citation_id"])
        self.assertIsNone(result["context_budget"])
        self.assertNotIn("book_position", first)
        self.assertEqual(first["source_ranks"], {})
        self.assertNotIn("private", json.dumps(result))

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

    def test_cancel_after_embedding_does_not_start_keyword_or_reranker(self):
        engine = self.engine()
        stopped = False
        rows = engine.search_embedding.return_value
        def embedding(*args, **kwargs):
            nonlocal stopped
            stopped = True
            return rows
        engine.search_embedding.side_effect = embedding
        with self.assertRaises(GenerationCancelled):
            engine.inspect_retrieval("进程", "os", should_stop=lambda: stopped)
        engine.search_bm25.assert_not_called()
        engine.reranker.rerank.assert_not_called()
        self.assertFalse(engine.llm.mock_calls)
