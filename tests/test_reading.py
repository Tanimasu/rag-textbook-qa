import hashlib
import unittest
from types import SimpleNamespace

from rag_textbook_qa.rag.engine import RAGEngine
from rag_textbook_qa.rag.reading import passage_window


class ReadingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"chunk_id": f"chapter_p{i}", "content": f" 原文 {i} ", "book_name": "os"}
            for i in range(5)
        ]

    def test_window_keeps_source_order_anchor_and_boundaries(self):
        result = passage_window(self.rows, "chapter_p2")
        self.assertEqual([row["chunk_id"] for row in result["passages"]],
                         ["chapter_p1", "chapter_p2", "chapter_p3"])
        self.assertEqual(result["previous_chunk_id"], "chapter_p1")
        self.assertEqual(result["next_chunk_id"], "chapter_p3")
        self.assertEqual([row["is_anchor"] for row in result["passages"]], [False, True, False])
        self.assertEqual(result["source_sha256"], hashlib.sha256("原文 2".encode()).hexdigest())
        self.assertIsNone(passage_window(self.rows, "chapter_p0")["previous_chunk_id"])
        self.assertIsNone(passage_window(self.rows, "chapter_p4")["next_chunk_id"])

    def test_missing_anchor_and_unbounded_radius_are_rejected(self):
        with self.assertRaises(LookupError):
            passage_window(self.rows, "missing")
        for radius in (-1, 3, True, 1.5):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                passage_window(self.rows, "chapter_p1", radius=radius)

    def test_engine_reads_cached_corpus_without_any_model(self):
        engine = object.__new__(RAGEngine)
        engine.vectorizer = SimpleNamespace()
        engine.bm25_doc_ids = {"os": ["chapter_p2", "chapter_p0", "chapter_p1"]}
        engine.bm25_corpus = {"os": ["三", "一", "二"]}
        engine.bm25_metadatas = {"os": [{"chapter": "章节"}] * 3}
        result = engine.read_passages("os", "chapter_p1")
        self.assertEqual([row["content"] for row in result["passages"]], ["一", "二", "三"])
        with self.assertRaises(LookupError):
            engine.read_passages("missing", "chapter_p1")

    def test_opaque_imported_ids_show_exact_anchor_without_inventing_order(self):
        engine = object.__new__(RAGEngine)
        engine.vectorizer = SimpleNamespace()
        engine.bm25_doc_ids = {"os": ["opaque-second", "opaque-first"]}
        engine.bm25_corpus = {"os": ["另一段正文", "准确定位的正文"]}
        engine.bm25_metadatas = {"os": [{}, {}]}
        result = engine.read_passages("os", "opaque-first")
        self.assertFalse(result["navigation_available"])
        self.assertEqual([row["content"] for row in result["passages"]], ["准确定位的正文"])
        self.assertIsNone(result["previous_chunk_id"])
        self.assertIsNone(result["next_chunk_id"])
        with self.assertRaises(LookupError):
            engine.read_passages("os", "missing")
