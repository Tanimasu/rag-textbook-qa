"""Offline safety checks for a reviewed, contradictory pair of textbook excerpts."""

import copy
import unittest

import test_decomposition as fixtures

from rag_textbook_qa.rag.conflicts import conflict_prompt_note, find_source_conflicts
from rag_textbook_qa.rag.engine import RAGEngine


class SourceConflictTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {
                "book_name": "database",
                "chunk_id": "ch7_s7_3_p467",
                "citation_id": 1,
                "content": "唯一索引允许所在列包含多个NULL值。",
            },
            {
                "book_name": "database",
                "chunk_id": "ch5_p281",
                "citation_id": 2,
                "content": "唯一码允许为空，但系统为保证其唯一性，最多只允许出现一个NULL值。",
            },
        ]

    def engine(self):
        engine = fixtures.DecompositionTests().engine()
        engine.search_single_book.return_value = copy.deepcopy(self.sources)
        return engine

    def test_a_conflict_is_disclosed_in_the_prompt_and_the_answer_still_runs(self):
        engine = self.engine()
        result = engine.ask("主键和唯一索引的区别？", book_name="database")
        prompt = engine.llm.generate_answer.call_args.args[0]

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "答案")
        self.assertTrue(result["source_conflicts"])
        self.assertNotIn("response_type", result)
        self.assertIn(conflict_prompt_note(result["source_conflicts"]), prompt)
        self.assertIn("必须如实指出", prompt)
        for source in result["context_sources"]:
            self.assertIn(f"【参考资料 {source['citation_id']}】", prompt)

    def test_one_side_or_unreviewed_sources_do_not_trigger(self):
        self.assertFalse(find_source_conflicts(self.sources[:1]))
        self.assertFalse(find_source_conflicts(self.sources[1:]))
        for field, value in [
            ("book_name", "os"),
            ("chunk_id", "new-edition"),
            ("content", "唯一码允许多个NULL值。"),
            ("citation_id", True),
        ]:
            sources = copy.deepcopy(self.sources)
            sources[1][field] = value
            self.assertFalse(find_source_conflicts(sources), (field, value))

    def test_truncated_away_opposing_statement_is_not_reported(self):
        results = copy.deepcopy(self.sources)
        results[1]["content"] = "前言。" * 100 + results[1]["content"]
        _, sources = RAGEngine.select_context(results, max_length=180)
        self.assertFalse(find_source_conflicts(sources))

    def test_reordered_sources_use_new_citation_numbers(self):
        engine = self.engine()
        engine.search_single_book.return_value.reverse()
        result = engine.ask("问题", book_name="database")
        note = conflict_prompt_note(result["source_conflicts"])

        self.assertIn("多个NULL值。”【参考资料 2】", note)
        self.assertIn("一个NULL值。”【参考资料 1】", note)

    def test_primary_key_section_also_triggers(self):
        self.sources[1].update(
            chunk_id="ch5_p285",
            content=("对于UNIQUE所约束的唯一码，则允许为NULL，但是只能有一个NULL值。"),
        )
        self.assertTrue(find_source_conflicts(self.sources))

    def test_answers_without_a_conflict_carry_no_note(self):
        engine = self.engine()
        engine.search_single_book.return_value = self.sources[:1]
        result = engine.ask("问题", book_name="database")
        prompt = engine.llm.generate_answer.call_args.args[0]

        engine.llm.generate_answer.assert_called_once()
        self.assertEqual(result["answer"], "答案")
        self.assertEqual(result["source_conflicts"], [])
        self.assertNotIn("已核实的表述冲突", prompt)

    def test_retrieval_only_keeps_no_answer_semantics(self):
        engine = self.engine()
        result = engine.ask("问题", book_name="database", use_llm=False)

        self.assertIsNone(result["answer"])
        self.assertTrue(result["source_conflicts"])
        engine.llm.generate_answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
