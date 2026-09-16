import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_textbook_qa.api.feedback import (
    AnswerRecordExpiredError,
    AnswerRegistry,
    FeedbackStore,
    build_feedback_candidates,
    summarize_feedback,
)


def result(answer: str = "教材回答") -> dict:
    return {
        "status": "answered",
        "answer": answer,
        "sources": [{"citation_id": 1, "book_id": "os", "excerpt": "教材原文"}],
        "conflicts": [],
        "timing": {"total_seconds": 1.25},
    }


class AnswerRegistryTests(unittest.TestCase):
    def test_answers_are_bounded_and_expire_without_being_persisted(self):
        now = [0.0]
        registry = AnswerRegistry(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
        first = registry.remember(query="问题1", book_id="os", result=result("回答1"))
        registry.remember(query="问题2", book_id="os", result=result("回答2"))
        third = registry.remember(query="问题3", book_id="os", result=result("回答3"))

        with self.assertRaises(AnswerRecordExpiredError):
            registry.resolve(first)
        self.assertEqual(registry.resolve(third).answer, "回答3")

        now[0] = 11.0
        with self.assertRaises(AnswerRecordExpiredError):
            registry.resolve(third)


class FeedbackStoreTests(unittest.TestCase):
    def test_every_short_lived_database_connection_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.sqlite3")
            connection = MagicMock()
            connection.execute.return_value.fetchall.return_value = []

            with patch.object(store, "_connect", return_value=connection):
                self.assertEqual(store.records(), [])

            connection.close.assert_called_once_with()

    def test_feedback_is_upserted_and_exported_as_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = FeedbackStore(root / "product" / "feedback.sqlite3")
            registry = AnswerRegistry()
            answer_id = registry.remember(query="什么是进程？", book_id="os", result=result())
            snapshot = registry.resolve(answer_id)

            store.save(snapshot, rating="helpful", reason=None, comment="")
            store.save(
                snapshot,
                rating="needs_improvement",
                reason="unsupported_answer",
                comment="第二段缺少依据",
            )
            output = root / "feedback.jsonl"
            count = store.export_jsonl(output)
            record = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(count, 1)
            self.assertEqual(record["answer_id"], answer_id)
            self.assertEqual(record["rating"], "needs_improvement")
            self.assertEqual(record["reason"], "unsupported_answer")
            self.assertEqual(record["sources"][0]["excerpt"], "教材原文")
            self.assertNotIn("ip", record)
            with self.assertRaises(FileExistsError):
                store.export_jsonl(output)
            self.assertEqual(store.export_jsonl(output, overwrite=True), 1)

    def test_summary_aggregates_feedback_without_exposing_text(self):
        records = [
            {
                "query": "不应出现在摘要中的问题",
                "answer": "不应出现在摘要中的答案",
                "book_id": "os",
                "rating": "helpful",
                "reason": None,
                "timing": {"total_seconds": 1.0},
            },
            {
                "query": "另一个问题",
                "answer": "另一个答案",
                "book_id": "os",
                "rating": "needs_improvement",
                "reason": "irrelevant_sources",
                "timing": {"total_seconds": 3.0},
            },
            {
                "book_id": "database",
                "rating": "needs_improvement",
                "reason": "too_slow",
                "timing": {"total_seconds": float("nan")},
            },
        ]

        summary = summarize_feedback(records)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["ratings"], {"helpful": 1, "needs_improvement": 2})
        self.assertEqual(summary["helpful_rate"], 0.3333)
        self.assertEqual(summary["reasons"], {"irrelevant_sources": 1, "too_slow": 1})
        self.assertEqual(summary["negative_by_book"], {"database": 1, "os": 1})
        self.assertEqual(
            summary["latency_seconds"],
            {"samples": 2, "average": 2.0, "p50": 2.0, "p95": 2.9},
        )
        rendered = json.dumps(summary, ensure_ascii=False)
        self.assertNotIn("不应出现在摘要", rendered)

    def test_negative_feedback_becomes_unlabelled_review_candidates(self):
        records = [
            {
                "answer_id": "a" * 32,
                "query": " 为什么会发生死锁？ ",
                "book_id": "os",
                "rating": "needs_improvement",
                "reason": "irrelevant_sources",
                "answer": "不应进入候选文件的回答",
                "comment": "不应进入候选文件的评论",
            },
            {
                "answer_id": "b" * 32,
                "query": "为什么会发生死锁？",
                "book_id": "os",
                "rating": "needs_improvement",
                "reason": "unsupported_answer",
            },
            {
                "answer_id": "c" * 32,
                "query": "什么是进程？",
                "book_id": "os",
                "rating": "helpful",
                "reason": None,
            },
            {
                "answer_id": "d" * 32,
                "query": "比较两本教材",
                "book_id": None,
                "rating": "needs_improvement",
                "reason": "other",
            },
        ]

        candidates = build_feedback_candidates(records)

        self.assertEqual(len(candidates), 2)
        candidate = next(item for item in candidates if item["book_name"] == "os")
        self.assertRegex(candidate["candidate_id"], r"^feedback-[0-9a-f]{12}$")
        self.assertEqual(candidate["question"], "为什么会发生死锁？")
        self.assertEqual(candidate["occurrences"], 2)
        self.assertEqual(
            candidate["feedback_reasons"],
            {"irrelevant_sources": 1, "unsupported_answer": 1},
        )
        self.assertEqual(candidate["suggested_checks"], ["grounding", "retrieval"])
        self.assertEqual(candidate["relevant_sections"], [])
        self.assertEqual(candidate["ground_truth"], "")
        rendered = json.dumps(candidate, ensure_ascii=False)
        self.assertNotIn("不应进入候选文件", rendered)
        all_books = next(item for item in candidates if item["book_name"] == "all_books")
        self.assertEqual(all_books["suggested_checks"], ["manual_review"])

    def test_candidate_export_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = FeedbackStore(root / "feedback.sqlite3")
            registry = AnswerRegistry()
            answer_id = registry.remember(
                query="为什么会发生死锁？",
                book_id="os",
                result=result(),
            )
            store.save(
                registry.resolve(answer_id),
                rating="needs_improvement",
                reason="not_answered",
                comment="",
            )
            output = root / "candidates.json"

            self.assertEqual(store.export_candidates(output), 1)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_count"], 1)
            self.assertEqual(payload["candidates"][0]["review_status"], "pending")
            with self.assertRaises(FileExistsError):
                store.export_candidates(output)


if __name__ == "__main__":
    unittest.main()
