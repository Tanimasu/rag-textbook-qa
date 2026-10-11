import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_textbook_qa.rag import RAGEngine


def row(book, identifier, relevance):
    return {"book_name": book, "chunk_id": identifier, "content": f"正文{relevance}",
            "similarity": 1, "section_h2": "知识正文"}


def engine_for(books=("a", "b"), reranker=True):
    engine = object.__new__(RAGEngine)
    engine.verbose = False
    provider = SimpleNamespace(embed_queries=MagicMock(return_value=[[1.0, 0.0]]),
                               embed_documents=MagicMock(return_value=[[0.0, 1.0]]))
    engine.vectorizer = SimpleNamespace(
        embedding_provider=provider,
        client=SimpleNamespace(list_collections=lambda: [SimpleNamespace(name="textbook_" + b)
                                                         for b in books]),
        validate_collection_embedding=MagicMock(),
    )
    engine.enable_hyde = False
    engine.enable_llm = True
    engine.llm = SimpleNamespace(generate_answer=MagicMock(return_value={"success": True,
                                                                       "answer": "假设正文"}))
    engine.reranker = SimpleNamespace(rerank=MagicMock(
        side_effect=lambda q, docs: [float(d.removeprefix("正文")) for d in docs]
    )) if reranker else None
    engine.fusion_weights = {"embedding": 1.0, "bm25": 1.0}
    data = {book: [row(book, str(i), 20 - i if book == "a" else 5 - i)
                   for i in range(7)] for book in books}
    engine.search_embedding = MagicMock(side_effect=lambda b, q, k, **kw: [dict(r) for r in data[b]][:k])
    engine.search_bm25 = MagicMock(side_effect=lambda b, q, k: [dict(r) for r in data[b]][:k])
    return engine


class MultibookRetrievalTests(unittest.TestCase):
    def test_global_top_five_can_all_come_from_one_book(self):
        engine = engine_for()
        results, _, _ = engine._retrieve("问题", {"status": "disabled"}, None, 5, False)
        self.assertEqual([(r["book_name"], r["chunk_id"]) for r in results],
                         [("a", str(i)) for i in range(5)])
        engine.vectorizer.embedding_provider.embed_queries.assert_called_once_with(["问题"])
        engine.reranker.rerank.assert_called_once()
        self.assertEqual(len(engine.reranker.rerank.call_args.args[1]), 14)
        # The same chunk ID in different books must remain two candidates.
        self.assertEqual(engine.reranker.rerank.call_args.args[1].count("正文5"), 1)

    def test_shared_vector_and_hyde_are_computed_once_per_question(self):
        engine = engine_for()
        engine.search_all_books("问题", use_hyde=True, use_reranker=False)
        engine.llm.generate_answer.assert_called_once()
        engine.vectorizer.embedding_provider.embed_documents.assert_called_once_with(["假设正文"])
        self.assertEqual([c.kwargs["_query_embedding"] for c in engine.search_embedding.call_args_list],
                         [[[0.0, 1.0]], [[0.0, 1.0]]])
        engine.search_all_books("另一个问题", use_hyde=False, use_reranker=False)
        engine.vectorizer.embedding_provider.embed_queries.assert_called_once_with(["另一个问题"])

    def test_single_book_keeps_the_existing_candidate_budget(self):
        engine = engine_for()
        rows = engine.search_single_book("a", "问题", 5, use_hyde=False)
        self.assertEqual([r["chunk_id"] for r in rows], [str(i) for i in range(5)])
        engine.search_embedding.assert_called_once_with("a", "问题", 45, use_hyde=False)
        engine.reranker.rerank.assert_called_once()

    def test_fingerprint_mismatch_fails_before_shared_model_call(self):
        engine = engine_for()
        engine.vectorizer.validate_collection_embedding.side_effect = ValueError("fingerprint")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            engine.search_all_books("问题")
        engine.vectorizer.embedding_provider.embed_queries.assert_not_called()
        engine.search_embedding.assert_not_called()

    def test_no_books_does_not_call_a_model(self):
        engine = engine_for(books=())
        self.assertEqual(engine.search_all_books("问题"), {})
        engine.vectorizer.embedding_provider.embed_queries.assert_not_called()

    def test_without_reranker_the_combined_output_is_bounded_by_top_k(self):
        engine = engine_for(reranker=False)
        rows, _, _ = engine._retrieve("问题", {"status": "disabled"}, None, 5, False)
        self.assertEqual(len(rows), 5)
        engine.vectorizer.embedding_provider.embed_queries.assert_called_once()


if __name__ == "__main__":
    unittest.main()
