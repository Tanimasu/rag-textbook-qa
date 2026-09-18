import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_textbook_qa.rag.adjacent import (
    append_same_section_neighbours,
    load_ordered_chunks,
)
from rag_textbook_qa.rag.context import DEFAULT_CONTEXT_BUDGET
from rag_textbook_qa.rag.engine import RAGEngine


def _chunk(chunk_id: str, section: str, content: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "book_name": "computer_network",
        "chapter": "第3章",
        "section_h2": "3.1 数据链路层",
        "section_h3": "3.1.2 三个基本问题",
        "section_h4": section,
        "content": content,
        "has_code": False,
        "has_image": False,
        "char_count": len(content),
        "level": 4,
    }


class AdjacentContextTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            _chunk("ch3_s3_1_p185", "1.封装成帧", "封装正文"),
            _chunk("ch3_s3_1_p186", "2.透明传输", "先说明问题"),
            _chunk("ch3_s3_1_p187", "2.透明传输", "再说明转义方法"),
            _chunk("ch3_s3_1_p188", "3.差错检测", "差错正文"),
        ]

    def test_loads_index_records_in_numeric_source_order(self):
        collection = MagicMock()
        collection.get.return_value = {
            "ids": ["ch3_s3_1_p187", "ch3_s3_1_p186"],
            "documents": ["再说明转义方法", "先说明问题"],
            "metadatas": [
                {key: value for key, value in self.corpus[2].items() if key not in {"chunk_id", "content"}},
                {key: value for key, value in self.corpus[1].items() if key not in {"chunk_id", "content"}},
            ],
        }

        rows = load_ordered_chunks(collection, "computer_network")

        self.assertEqual([row["chunk_id"] for row in rows], ["ch3_s3_1_p186", "ch3_s3_1_p187"])
        self.assertEqual(rows[1]["content"], "再说明转义方法")

    def test_rejects_missing_or_duplicate_source_positions(self):
        collection = MagicMock()
        collection.get.return_value = {
            "ids": ["custom-id"],
            "documents": ["正文"],
            "metadatas": [{}],
        }
        with self.assertRaisesRegex(ValueError, "缺少顺序编号"):
            load_ordered_chunks(collection, "book")

        collection.get.return_value = {
            "ids": ["a_p001", "b_p001"],
            "documents": ["一", "二"],
            "metadatas": [{}, {}],
        }
        with self.assertRaisesRegex(ValueError, "顺序编号重复"):
            load_ordered_chunks(collection, "book")

    def test_appends_only_exact_section_neighbours_after_all_ranked_results(self):
        results = [
            {**self.corpus[1], "rank": 1, "query_ids": [0, 1]},
            {**self.corpus[3], "rank": 2},
        ]

        expanded = append_same_section_neighbours(
            results, {"computer_network": self.corpus}
        )

        self.assertEqual(
            [row["chunk_id"] for row in expanded],
            ["ch3_s3_1_p186", "ch3_s3_1_p188", "ch3_s3_1_p187"],
        )
        self.assertEqual(expanded[2]["adjacent_of"], "ch3_s3_1_p186")
        self.assertEqual(expanded[2]["adjacent_direction"], "next")
        self.assertEqual(expanded[2]["query_ids"], [0, 1])

    def test_unknown_results_are_preserved_without_inventing_neighbours(self):
        result = {**self.corpus[1], "chunk_id": "missing"}

        self.assertEqual(
            append_same_section_neighbours(
                [result], {"computer_network": self.corpus}
            ),
            [result],
        )

    def _engine(self):
        engine = object.__new__(RAGEngine)
        engine.verbose = False
        engine.enable_llm = False
        engine.enable_adjacent_context = False
        engine.llm = None
        engine.reranker = None
        engine.context_budget = DEFAULT_CONTEXT_BUDGET
        engine.compute_settings = MagicMock(backend="local", query_fallback_to_local=False)
        collection = MagicMock()
        collection.get.return_value = {
            "ids": [row["chunk_id"] for row in self.corpus],
            "documents": [row["content"] for row in self.corpus],
            "metadatas": [
                {key: value for key, value in row.items() if key not in {"chunk_id", "content"}}
                for row in self.corpus
            ],
        }
        client = MagicMock()
        client.get_collection.return_value = collection
        engine.vectorizer = SimpleNamespace(embedding_provider=None, client=client)
        engine.search_single_book = MagicMock(
            return_value=[{**self.corpus[1], "rank": 1, "method": "hybrid-rerank"}]
        )
        engine._execution_summary = MagicMock(return_value={})
        return engine

    def test_engine_default_does_not_load_or_append_neighbours(self):
        engine = self._engine()

        result = engine.ask("透明传输是什么", book_name="computer_network", use_llm=False)

        engine.vectorizer.client.get_collection.assert_not_called()
        self.assertEqual(
            [source["chunk_id"] for source in result["context_sources"]],
            ["ch3_s3_1_p186"],
        )
        self.assertEqual(
            result["context_expansion"],
            {
                "enabled": False,
                "applied": False,
                "reason": None,
                "added_candidates": 0,
                "added_sources": 0,
            },
        )

    def test_engine_opt_in_appends_neighbour_without_changing_results(self):
        engine = self._engine()

        result = engine.ask(
            "透明传输是什么",
            book_name="computer_network",
            use_llm=False,
            use_adjacent_context=True,
        )

        self.assertEqual(
            [row["chunk_id"] for row in result["results"]],
            ["ch3_s3_1_p186"],
        )
        self.assertEqual(
            [source["chunk_id"] for source in result["context_sources"]],
            ["ch3_s3_1_p186", "ch3_s3_1_p187"],
        )
        self.assertEqual(
            result["context_expansion"],
            {
                "enabled": True,
                "applied": True,
                "reason": None,
                "added_candidates": 1,
                "added_sources": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
