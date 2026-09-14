import unittest

from rag_textbook_qa.evaluation.retrieval import (
    RetrievalQuestion,
    evaluate_retrieval,
    score_context_retention,
)

EXACT = {"chunk_id": "a", "chapter": "第3章", "section_h3": "3.5.3 避免", "content": "甲" * 100}
SIBLING = {"chunk_id": "b", "chapter": "第3章", "section_h3": "3.5.2 检测", "content": "乙" * 100}
UNRELATED = {"chunk_id": "c", "chapter": "第9章", "section_h3": "9.1 文件", "content": "丙" * 100}


class ContextRetentionTests(unittest.TestCase):
    def test_counts_relevant_evidence_the_budget_dropped(self):
        packed = [{"chunk_id": "a", "content": "甲" * 60, "truncated": True}]
        score = score_context_retention([EXACT, SIBLING, UNRELATED], packed, ["3.5.3 避免"])

        self.assertEqual((score["relevant_retrieved"], score["relevant_retained"]), (2, 1))
        self.assertEqual((score["exact_retrieved"], score["exact_retained"]), (1, 1))
        self.assertEqual(score["relevant_truncated"], 1)
        self.assertAlmostEqual(score["context_retention"], 0.5)
        self.assertEqual((score["sources_packed"], score["context_chars"]), (1, 60))

    def test_a_question_without_relevant_results_has_no_retention(self):
        score = score_context_retention([UNRELATED], [UNRELATED], ["3.5.3 避免"])

        self.assertEqual(score["relevant_retrieved"], 0)
        self.assertIsNone(score["context_retention"])

    def test_evaluation_reports_retention_only_when_packing_is_supplied(self):
        question = RetrievalQuestion("问题", "os", ("3.5.3 避免",))
        results = [EXACT, SIBLING]

        without = evaluate_retrieval([question], lambda q, k: results, top_k=5)
        packed = evaluate_retrieval(
            [question],
            lambda q, k: results,
            top_k=5,
            pack=lambda rows: ("", [dict(EXACT)]),
        )

        self.assertNotIn("mean_context_retention", without)
        self.assertNotIn("context_retention", without["cases"][0])
        self.assertAlmostEqual(packed["mean_context_retention"], 0.5)
        self.assertEqual(packed["relevant_dropped_total"], 1)
        self.assertEqual(packed["questions_with_relevant_evidence"], 1)
        self.assertEqual(packed["cases"][0]["relevant_retained"], 1)


if __name__ == "__main__":
    unittest.main()
