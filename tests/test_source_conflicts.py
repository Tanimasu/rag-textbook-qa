"""Offline safety checks for a reviewed, contradictory pair of textbook excerpts."""

import copy
import unittest

import test_decomposition as fixtures

from rag_textbook_qa.rag.conflicts import (
    CONFLICT_RULES,
    conflict_prompt_note,
    find_source_conflicts,
    validate_conflict_rules,
)
from rag_textbook_qa.rag.context import select_context


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
        _, sources = select_context(results, max_length=180)
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

    def test_validation_passes_when_every_pinned_quote_is_still_indexed(self):
        corpus = {
            (rule.book_name, pin.chunk_id): f"前言。{pin.quote}后文。"
            for rule in CONFLICT_RULES
            for pin in rule.pins()
        }
        problems = validate_conflict_rules(
            lambda book, chunk: corpus.get((book, chunk))
        )

        self.assertTrue(corpus)
        self.assertEqual(problems, [])

    def test_validation_separates_a_vanished_chunk_from_a_rewritten_quote(self):
        pins = [(rule.book_name, pin) for rule in CONFLICT_RULES for pin in rule.pins()]
        self.assertGreaterEqual(len(pins), 2)
        corpus = {(book, pin.chunk_id): pin.quote for book, pin in pins}
        del corpus[(pins[0][0], pins[0][1].chunk_id)]
        corpus[(pins[1][0], pins[1][1].chunk_id)] = "改版后重写的句子。"

        problems = validate_conflict_rules(
            lambda book, chunk: corpus.get((book, chunk))
        )

        self.assertEqual(
            [(problem["chunk_id"], problem["status"]) for problem in problems],
            [
                (pins[0][1].chunk_id, "missing"),
                (pins[1][1].chunk_id, "quote_changed"),
            ],
        )

    def test_every_registered_rule_needs_two_sides_to_be_a_conflict(self):
        self.assertTrue(CONFLICT_RULES)
        for rule in CONFLICT_RULES:
            with self.subTest(rule=rule.id):
                self.assertGreaterEqual(len(rule.sides), 2)
                self.assertTrue(all(side for side in rule.sides))

    def test_retrieval_only_keeps_no_answer_semantics(self):
        engine = self.engine()
        result = engine.ask("问题", book_name="database", use_llm=False)

        self.assertIsNone(result["answer"])
        self.assertTrue(result["source_conflicts"])
        engine.llm.generate_answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
