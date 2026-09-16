import json
import tempfile
import unittest
from pathlib import Path

from rag_textbook_qa.api.feedback import (
    AnswerRecordExpiredError,
    AnswerRegistry,
    FeedbackStore,
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


if __name__ == "__main__":
    unittest.main()
