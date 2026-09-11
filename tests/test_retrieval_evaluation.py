import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from rag_textbook_qa.evaluation.retrieval import (
    RetrievalQuestion,
    evaluate_retrieval,
    load_retrieval_questions,
    result_section,
    score_ranked_results,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHUNK_FILES = {
    "os": "操作系统_mineru_chunks.json",
    "computer_organization": "计算机组成原理_mineru_chunks.json",
    "computer_network": "计算机网络_mineru_chunks.json",
    "data_structure": "数据结构_mineru_chunks.json",
    "database": "数据库原理及应用教程_mineru_chunks.json",
}


class RetrievalEvaluationTests(unittest.TestCase):
    def test_loads_validated_questions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "questions.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "question": " 什么是死锁？ ",
                            "book_name": " os ",
                            "relevant_sections": [" 3.5 "],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            questions = load_retrieval_questions(path)

        self.assertEqual(
            questions,
            [RetrievalQuestion("什么是死锁？", "os", ("3.5",))],
        )

    def test_rejects_missing_relevance_annotations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "questions.json"
            path.write_text(
                '[{"question":"问题","book_name":"os"}]',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "relevant_sections"):
                load_retrieval_questions(path)

    def test_repository_annotations_cover_two_questions_per_book(self):
        questions = load_retrieval_questions(
            REPOSITORY_ROOT / "data" / "evaluation" / "retrieval_questions.json"
        )
        self.assertEqual(
            Counter(question.book_name for question in questions),
            {book_name: 2 for book_name in CHUNK_FILES},
        )

        chunks_by_book = {
            book_name: json.loads(
                (REPOSITORY_ROOT / "data" / "chunks" / filename).read_text(encoding="utf-8")
            )
            for book_name, filename in CHUNK_FILES.items()
        }
        for question in questions:
            normalized_hierarchies = [
                "".join(result_section(chunk).lower().split())
                for chunk in chunks_by_book[question.book_name]
            ]
            for marker in question.relevant_sections:
                normalized_marker = "".join(marker.lower().split())
                self.assertTrue(
                    any(normalized_marker in hierarchy for hierarchy in normalized_hierarchies),
                    f"未在 {question.book_name} chunks 中找到标注章节: {marker}",
                )

    def test_scores_section_recall_and_reciprocal_rank(self):
        score = score_ranked_results(
            [
                {"chapter": "第1章", "section_h2": "1.1 引论"},
                {"chapter": "第3章", "section_h2": "3.5 死锁概述"},
                {"chapter": "第3章", "section_h3": "3.6 死锁预防"},
            ],
            ["3.5", "3.6"],
            top_k=3,
        )

        self.assertEqual(score["recall_at_k"], 1.0)
        self.assertEqual(score["reciprocal_rank"], 0.5)
        self.assertEqual(score["first_relevant_rank"], 2)
        self.assertEqual(score["matched_sections"], ["3.5", "3.6"])

    def test_aggregates_metrics_without_models_or_network(self):
        questions = [
            RetrievalQuestion("命中", "os", ("3.5",)),
            RetrievalQuestion("未命中", "os", ("6.1",)),
        ]

        def search(question, top_k):
            self.assertEqual(top_k, 5)
            if question.question == "命中":
                return [{"chapter": "第3章", "section_h2": "3.5 死锁概述"}]
            return [{"chapter": "第1章", "section_h2": "1.1 引论"}]

        report = evaluate_retrieval(questions, search)

        self.assertEqual(report["question_count"], 2)
        self.assertEqual(report["mean_recall_at_k"], 0.5)
        self.assertEqual(report["hit_rate_at_k"], 0.5)
        self.assertEqual(report["mrr"], 0.5)
        self.assertEqual(len(report["cases"]), 2)


if __name__ == "__main__":
    unittest.main()
