import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from rag_textbook_qa.evaluation import load_retrieval_questions, load_test_questions

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_PATH = REPOSITORY_ROOT / "data" / "evaluation" / "product_acceptance_v1.json"
SOURCE_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "evaluation"
    / "retrieval_holdout_candidates_v4.json"
)


class ProductAcceptanceDatasetTests(unittest.TestCase):
    def test_dataset_is_shared_by_retrieval_and_answer_evaluation(self):
        retrieval_questions = load_retrieval_questions(ACCEPTANCE_PATH)
        answer_questions = load_test_questions(ACCEPTANCE_PATH)

        self.assertEqual(len(retrieval_questions), 15)
        self.assertEqual(len(answer_questions), 15)
        self.assertEqual(
            Counter(question.book_name for question in retrieval_questions),
            {
                "os": 3,
                "computer_organization": 3,
                "computer_network": 3,
                "data_structure": 3,
                "database": 3,
            },
        )
        self.assertTrue(all(question.split == "holdout" for question in retrieval_questions))
        self.assertTrue(all(item.get("ground_truth", "").strip() for item in answer_questions))

    def test_snapshot_matches_its_reviewed_source_and_evidence(self):
        acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))
        source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(len({item["id"] for item in acceptance}), len(acceptance))
        self.assertEqual([item["id"] for item in acceptance], [item["id"] for item in source])

        digests: dict[Path, str] = {}
        line_counts: dict[Path, int] = {}
        for item, original in zip(acceptance, source, strict=True):
            with self.subTest(item=item["id"]):
                self.assertEqual(item["question"], original["question"])
                self.assertEqual(item["book_name"], original["book_name"])
                self.assertEqual(item["relevant_sections"], original["relevant_sections"])
                self.assertEqual(item["ground_truth"], original["answer_key"])
                self.assertEqual(item["review_status"], original["annotation_status"])
                self.assertEqual(item["source_revision"], original["revision"])

                evidence = item["evidence"]
                evidence_path = REPOSITORY_ROOT / evidence["path"]
                if evidence_path not in digests:
                    content = evidence_path.read_bytes()
                    digests[evidence_path] = hashlib.sha256(content).hexdigest()
                    line_counts[evidence_path] = len(content.decode("utf-8").splitlines())
                self.assertEqual(digests[evidence_path], evidence["sha256"])
                self.assertGreaterEqual(evidence["start_line"], 1)
                self.assertGreaterEqual(evidence["end_line"], evidence["start_line"])
                self.assertLessEqual(evidence["end_line"], line_counts[evidence_path])


if __name__ == "__main__":
    unittest.main()
