import copy
import json
import unittest

from rag_textbook_qa.evaluation.judge_review import compare_judge_review


class JudgeReviewTests(unittest.TestCase):
    def setUp(self):
        self.extracted = {"judge_version": 3, "rows": [{"case_id": "q1", "claims": [
            {"id": index, "text": f"陈述{index}", "type": "fact"} for index in range(1, 5)
        ]}]}
        self.labels = {"q1": {"labels": {"1": 1, "2": 0, "3": 1, "4": 0},
                               "missed": ["遗漏的事实"]}}
        self.verified = [{"case_id": "q1", "score": {"claims": [
            {"id": index, "text": f"陈述{index}", "status": status}
            for index, status in enumerate(("unsupported", "unverified", "supported", "minor"), 1)
        ]}}]

    def compare(self):
        return compare_judge_review(self.extracted, self.labels, self.verified)

    def test_confusion_and_undefined_metrics_are_not_reported_as_accuracy(self):
        report = self.compare()
        self.assertEqual(report["confusion"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        self.assertEqual(report["kappa"], 0)
        self.assertEqual(report["judge_recall"], 0.5)
        self.assertEqual(report["judge_precision"], 0.5)
        self.assertEqual(report["missed_by_extraction"], 1)
        self.assertIn("not independent human accuracy", report["scope"])
        for claim in self.verified[0]["score"]["claims"]:
            claim["status"] = "supported"
        self.labels["q1"]["labels"] = {str(index): 0 for index in range(1, 5)}
        report = self.compare()
        self.assertIsNone(report["kappa"])
        self.assertIsNone(report["judge_recall"])
        self.assertIsNone(report["judge_precision"])
        json.dumps(report, allow_nan=False)

    def test_missing_duplicate_or_substituted_claims_are_rejected(self):
        original = copy.deepcopy(self.verified)
        for problem in ("missing", "extra", "duplicate", "changed_text", "unknown_status", "bool_id"):
            with self.subTest(problem=problem):
                self.verified = copy.deepcopy(original)
                claims = self.verified[0]["score"]["claims"]
                if problem == "missing":
                    claims.pop()
                elif problem == "extra":
                    claims.append({"id": 5, "text": "陈述5", "status": "supported"})
                elif problem == "duplicate":
                    claims.append(copy.deepcopy(claims[0]))
                elif problem == "changed_text":
                    claims[0]["text"] = "不同的陈述"
                elif problem == "unknown_status":
                    claims[0]["status"] = "pass"
                else:
                    claims[0]["id"] = True
                with self.assertRaises(ValueError):
                    self.compare()

    def test_answers_without_fact_claims_have_undefined_agreement(self):
        self.extracted["rows"][0]["claims"] = [{"id": 1, "type": "meta", "text": "资料不足"}]
        self.verified[0]["score"]["claims"] = []
        self.labels["q1"] = {"labels": {}, "missed": []}
        report = self.compare()
        self.assertEqual(report["fact_claims"], 0)
        self.assertIsNone(report["agreement"])
        self.assertIsNone(report["kappa"])
        json.dumps(report, allow_nan=False)

    def test_invalid_labels_and_case_alignment_are_rejected(self):
        original = copy.deepcopy(self.labels)
        for problem in ("missing_case", "extra_case", "missing_label", "bool_label", "missing_omissions"):
            with self.subTest(problem=problem):
                self.labels = copy.deepcopy(original)
                if problem == "missing_case":
                    self.labels.clear()
                elif problem == "extra_case":
                    self.labels["q2"] = copy.deepcopy(self.labels["q1"])
                elif problem == "missing_label":
                    self.labels["q1"]["labels"].pop("1")
                elif problem == "bool_label":
                    self.labels["q1"]["labels"]["1"] = True
                else:
                    self.labels["q1"].pop("missed")
                with self.assertRaises(ValueError):
                    self.compare()


if __name__ == "__main__":
    unittest.main()
