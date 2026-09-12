import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from rag_textbook_qa.evaluation.retrieval import (
    RETRIEVAL_STRATEGIES,
    RetrievalQuestion,
    evaluate_retrieval,
    load_retrieval_questions,
    result_section,
    run_retrieval_strategies,
    save_retrieval_report,
    score_ranked_results,
    search_with_strategy,
    select_split,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Annotations must resolve against the chunks the live index is built from.
# data/chunks/ is the frozen original artifact set and no longer matches the
# current chunker, so it cannot validate them. artifacts/chunks/ is gitignored,
# hence the skip on a fresh checkout and in CI.
CHUNKS_DIR = REPOSITORY_ROOT / "artifacts" / "chunks"
QUESTIONS_PER_BOOK = 10
CHUNK_FILES = {
    "os": "操作系统_mineru_chunks.json",
    "computer_organization": "计算机组成原理_mineru_chunks.json",
    "computer_network": "计算机网络_mineru_chunks.json",
    "data_structure": "数据结构_mineru_chunks.json",
    "database": "数据库原理及应用教程_mineru_chunks.json",
}


class FakeRetrievalEngine:
    def __init__(self):
        self.calls = []

    def search_bm25(self, book_name, question, top_k):
        self.calls.append(("bm25", book_name, question, top_k))
        return [{"chapter": "第3章", "section_h2": "3.5 死锁概述"}]

    def search_embedding(self, book_name, question, top_k, *, use_hyde):
        self.calls.append(("embedding", book_name, question, top_k, use_hyde))
        return [{"chapter": "第3章", "section_h2": "3.5 死锁概述"}]

    def search_single_book(
        self,
        book_name,
        question,
        top_k,
        *,
        use_hyde,
        use_reranker,
    ):
        self.calls.append(
            ("hybrid", book_name, question, top_k, use_hyde, use_reranker)
        )
        return [{"chapter": "第3章", "section_h2": "3.5 死锁概述"}]


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

    def test_questions_without_a_split_default_to_dev(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "questions.json"
            path.write_text(
                '[{"question":"问题","book_name":"os","relevant_sections":["3.5"]}]',
                encoding="utf-8",
            )

            self.assertEqual(load_retrieval_questions(path)[0].split, "dev")

    def test_an_unknown_split_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "questions.json"
            path.write_text(
                '[{"question":"问题","book_name":"os","relevant_sections":["3.5"],'
                '"split":"test"}]',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "split"):
                load_retrieval_questions(path)

    def test_select_split_filters_and_refuses_an_empty_selection(self):
        dev = RetrievalQuestion("问题1", "os", ("3.5",), "dev")
        holdout = RetrievalQuestion("问题2", "os", ("3.6",), "holdout")

        self.assertEqual(select_split([dev, holdout], "dev"), [dev])
        self.assertEqual(select_split([dev, holdout], "all"), [dev, holdout])
        with self.assertRaisesRegex(ValueError, "没有问题"):
            select_split([dev], "holdout")

    def test_historical_holdout_is_stratified(self):
        questions = load_retrieval_questions(
            REPOSITORY_ROOT / "data" / "evaluation" / "retrieval_questions.json"
        )
        holdout = [question for question in questions if question.split == "holdout"]

        self.assertEqual(len(holdout), 15)
        self.assertEqual(
            Counter(question.book_name for question in holdout),
            {book_name: 3 for book_name in CHUNK_FILES},
        )

    def test_repository_annotations_cover_ten_questions_per_book(self):
        questions = load_retrieval_questions(
            REPOSITORY_ROOT / "data" / "evaluation" / "retrieval_questions.json"
        )
        self.assertEqual(
            Counter(question.book_name for question in questions),
            {book_name: QUESTIONS_PER_BOOK for book_name in CHUNK_FILES},
        )

        missing = [name for name in CHUNK_FILES.values() if not (CHUNKS_DIR / name).is_file()]
        if missing:
            self.skipTest(f"未生成 {CHUNKS_DIR}，跳过标注章节校验")

        chunks_by_book = {
            book_name: json.loads((CHUNKS_DIR / filename).read_text(encoding="utf-8"))
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
                    f"未在 {question.book_name} 的 artifacts/chunks 中找到标注章节: {marker}",
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

    def test_routes_each_named_strategy_without_hyde(self):
        engine = FakeRetrievalEngine()
        question = RetrievalQuestion("什么是死锁？", "os", ("3.5",))

        for strategy in RETRIEVAL_STRATEGIES:
            search_with_strategy(engine, question, 5, strategy=strategy)

        self.assertEqual(
            engine.calls,
            [
                ("bm25", "os", "什么是死锁？", 5),
                ("embedding", "os", "什么是死锁？", 5, False),
                ("hybrid", "os", "什么是死锁？", 5, False, False),
                ("hybrid", "os", "什么是死锁？", 5, False, True),
            ],
        )

    def test_runs_multiple_strategies_and_saves_timestamped_report(self):
        engine = FakeRetrievalEngine()
        questions = [RetrievalQuestion("什么是死锁？", "os", ("3.5",))]

        report = run_retrieval_strategies(
            engine,
            questions,
            ("bm25", "hybrid-rerank"),
            top_k=3,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["question_count"], 1)
        self.assertEqual(report["top_k"], 3)
        self.assertEqual(set(report["strategies"]), {"bm25", "hybrid-rerank"})
        self.assertEqual(report["strategies"]["bm25"]["hit_rate_at_k"], 1.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = save_retrieval_report(report, temporary_directory)
            saved = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(report_path.name.startswith("retrieval_"))
        self.assertEqual(saved, report)


if __name__ == "__main__":
    unittest.main()
