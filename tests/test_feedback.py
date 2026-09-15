import json
import tempfile
import unittest
from pathlib import Path

from rag_textbook_qa.api.feedback import (
    AnswerRecordExpiredError,
    AnswerRegistry,
    FeedbackStore,
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


if __name__ == "__main__":
    unittest.main()
