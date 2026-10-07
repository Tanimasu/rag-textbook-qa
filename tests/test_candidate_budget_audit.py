import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rag_textbook_qa.evaluation.retrieval import RetrievalQuestion, SourceEvidence
from scripts.audit_candidate_budget import audit


class CandidateBudgetAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "questions.json"
        self.dataset.write_text(json.dumps([{
            "question": "如何管理资源？", "book_name": "os", "split": "dev", "relevant_sections": ["资源"],
        }]), encoding="utf-8")
        self.report_path = self.root / "report.json"
        self.report = {
            "complete": True, "split": "dev", "top_k": 5, "index_revision": "a" * 64,
            "dataset_sha256": hashlib.sha256(self.dataset.read_bytes()).hexdigest(),
            "cases": [{"question": "如何管理资源？", "book_name": "os", "budgets": {
                "2": {"pairs": 50, "ranking": [["os", f"c{i}"] for i in range(1, 6)]},
                "3": {"pairs": 75, "ranking": [["os", f"c{i}"] for i in range(2, 7)]},
            }}],
        }
        self.write()
        self.client = Mock()
        self.client.get_collection.return_value.get.side_effect = lambda *, ids, include: {
            "ids": ids, "documents": ["核心证据：资源管理与分配。" for _ in ids],
            "metadatas": [{"section_h2": "资源管理"} for _ in ids],
        }

    def write(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")

    def run_audit(self):
        with (
            patch("scripts.audit_candidate_budget.copy_index", return_value="a" * 64) as copier,
            patch("scripts.audit_candidate_budget.chromadb.PersistentClient", return_value=self.client) as factory,
        ):
            result = audit(self.report_path, self.dataset, self.root / "original-db")
            self.assertEqual(copier.call_args.args[0], self.root / "original-db")
            self.assertNotEqual(factory.call_args.kwargs["path"], str(self.root / "original-db"))
            self.client.close.assert_called_once()
            return result

    def test_differences_preserve_body_hash_and_do_not_assign_body_grades(self):
        result = self.run_audit()
        self.assertEqual((result["ranking_unchanged"], result["lost_top5_occurrences"]), (0, 1))
        case = result["cases"][0]
        lost, gained = (case[key][0] for key in ("lost_from_top5_at_50", "gained_in_top5_at_50"))
        self.assertEqual((lost["chunk_id"], gained["chunk_id"]), ("c6", "c1"))
        self.assertEqual(lost["content_sha256"], hashlib.sha256(lost["content"].encode()).hexdigest())
        self.assertNotIn("grade", lost)
        self.assertIsNone(case["annotated_source_coverage"])
        self.assertEqual(result["questions_with_annotated_source_spans"], 0)

    def test_only_existing_source_annotations_enable_body_coverage(self):
        evidence = SourceEvidence("fixture.md", 1, 1, "b" * 64, "核心证据：资源管理与分配。")
        question = RetrievalQuestion("如何管理资源？", "os", ("资源",), evidence=evidence)
        with patch("scripts.audit_candidate_budget.load_retrieval_questions", return_value=[question]):
            result = self.run_audit()
        coverage = result["cases"][0]["annotated_source_coverage"]
        self.assertEqual(coverage["50"]["source_evidence_coverage_at_k"], 1)
        self.assertEqual(result["questions_with_annotated_source_spans"], 1)

    def test_mismatched_source_and_invalid_rankings_fail_before_index_access(self):
        original = json.loads(self.report_path.read_bytes())
        for problem in ("holdout", "hash", "question", "duplicate", "budget"):
            with self.subTest(problem=problem):
                self.report = json.loads(json.dumps(original))
                if problem == "holdout":
                    self.report["split"] = "holdout"
                elif problem == "hash":
                    self.report["dataset_sha256"] = "b" * 64
                elif problem == "question":
                    self.report["cases"][0]["question"] = "替换问题"
                elif problem == "duplicate":
                    self.report["cases"][0]["budgets"]["2"]["ranking"][0] = ["os", "c2"]
                else:
                    self.report["cases"][0]["budgets"]["2"]["pairs"] = 75
                self.write()
                with patch("scripts.audit_candidate_budget.copy_index") as copier, self.assertRaises(ValueError):
                    audit(self.report_path, self.dataset, self.root / "original-db")
                copier.assert_not_called()

    def test_stale_index_fails_before_open_and_missing_body_still_closes_client(self):
        with (
            patch("scripts.audit_candidate_budget.copy_index", return_value="b" * 64),
            patch("scripts.audit_candidate_budget.chromadb.PersistentClient") as factory,
            self.assertRaisesRegex(ValueError, "版本不同"),
        ):
            audit(self.report_path, self.dataset, self.root / "original-db")
        factory.assert_not_called()
        self.client.get_collection.return_value.get.side_effect = None
        self.client.get_collection.return_value.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        with (
            patch("scripts.audit_candidate_budget.copy_index", return_value="a" * 64),
            patch("scripts.audit_candidate_budget.chromadb.PersistentClient", return_value=self.client),
            self.assertRaisesRegex(ValueError, "缺失候选"),
        ):
            audit(self.report_path, self.dataset, self.root / "original-db")
        self.client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
