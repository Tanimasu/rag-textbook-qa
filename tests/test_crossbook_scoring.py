"""Integrity checks and hand-calculated scores for imported candidate ratings."""

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.evaluation.crossbook import score_review, validate_review
from scripts.score_crossbook_review import main


class CrossbookScoringTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.template_dir = self.root / "export"
        self.template_dir.mkdir()
        self.review_path = self.root / "reviewed.json"
        candidates = []
        for book, chunk, content in (
            ("os", "chunk-1", "核心证据：管理系统资源。"),
            ("database", "chunk-1", "相关背景：数据资源。"),
            ("os", "chunk-2", "部分证据：分配资源。"),
        ):
            candidates.append({
                "candidate_id": hashlib.sha256(json.dumps([book, chunk]).encode()).hexdigest(),
                "book_name": book, "chunk_id": chunk, "section": {"section_h2": "资源管理"},
                "content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "grade": None, "evidence_quotes": [], "rationale": "",
            })
        self.template = {
            "schema_version": 1, "split": "dev", "review_status": "pending",
            "reviewer": None, "reviewed_at_utc": None, "instructions": "依据正文评分",
            "questions": [{"question_id": "q1", "question": "如何管理资源？", "candidates": candidates}],
        }
        identifiers = [row["candidate_id"] for row in candidates]
        self.manifest = {
            "source_report_sha256": "a" * 64, "source_dataset_sha256": "b" * 64,
            "index_revision": "c" * 64, "routes": ["before", "after"],
            "questions": [{"question_id": "q1", "rankings": {
                "before": identifiers[:2], "after": [identifiers[2], identifiers[1]],
            }}],
        }
        self.review = copy.deepcopy(self.template)
        self.review.update(review_status="complete", reviewer="test fixture",
                           reviewed_at_utc="2026-10-07T00:00:00+00:00")
        for row, grade in zip(self.review["questions"][0]["candidates"], (3, 1, 2), strict=True):
            row.update(grade=grade, evidence_quotes=[row["content"]], rationale="测试评分")
        self.write()

    def write(self):
        for path, value in (
            (self.template_dir / "review.json", self.template),
            (self.template_dir / "manifest.json", self.manifest), (self.review_path, self.review),
        ):
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def score(self, top_k=2):
        return score_review(self.template_dir, self.review_path, top_k=top_k)

    def test_scores_use_book_and_chunk_identity_and_the_pool_ideal(self):
        result = self.score()
        discount = math.log2(3)
        ideal = 7 + 3 / discount
        before, after = (result["summary"][route] for route in ("before", "after"))
        self.assertAlmostEqual(before["mean_pooled_ndcg_at_k"], (7 + 1 / discount) / ideal)
        self.assertAlmostEqual(after["mean_pooled_ndcg_at_k"], (3 + 1 / discount) / ideal)
        for route in (before, after):
            self.assertEqual(route["mean_useful_coverage_of_pool_at_k"], 0.5)
            self.assertEqual(route["useful_hit_rate_at_k"], 1)
        self.assertEqual(result["comparison"]["pooled_ndcg_worsened_questions"], 1)
        self.assertIn("unpooled candidates are unknown", result["scope"])
        self.assertIn("self-declared", result["reviewer_verification"])

    def test_pending_review_can_be_validated_but_not_scored(self):
        result = validate_review(self.template_dir, self.template_dir / "review.json")
        self.assertEqual((result["scored"], result["unscored"]), (0, 3))
        self.review = copy.deepcopy(self.template)
        self.write()
        with self.assertRaisesRegex(ValueError, "尚未完成"):
            self.score()
        with self.assertRaisesRegex(ValueError, "原模板的副本"):
            score_review(self.template_dir, self.template_dir / "review.json")

    def test_changed_fixed_fields_are_rejected_even_with_recomputed_body_hash(self):
        clean = copy.deepcopy(self.review)
        for field in ("content", "book_name", "chunk_id", "candidate_id", "section"):
            with self.subTest(field=field):
                self.review = copy.deepcopy(clean)
                row = self.review["questions"][0]["candidates"][0]
                row[field] = "改变"
                row["content_sha256"] = hashlib.sha256(row["content"].encode()).hexdigest()
                self.write()
                with self.assertRaises(ValueError):
                    self.score()
        self.review = copy.deepcopy(clean)
        self.review["schema_version"] = True  # True == 1 must not bypass immutable JSON types.
        self.write()
        with self.assertRaises(ValueError):
            self.score()

    def test_bad_grades_quotes_and_rationale_are_rejected(self):
        clean = copy.deepcopy(self.review)
        for changes in ({"grade": True}, {"grade": "3"}, {"grade": 3.0}, {"grade": 4},
                        {"grade": -1}, {"evidence_quotes": []},
                        {"evidence_quotes": ["正文没有这句话"]}, {"rationale": " "}):
            with self.subTest(changes=changes):
                self.review = copy.deepcopy(clean)
                self.review["questions"][0]["candidates"][0].update(changes)
                self.write()
                with self.assertRaises(ValueError):
                    self.score()

    def test_incomplete_review_and_invalid_reviewer_time_are_rejected(self):
        clean = copy.deepcopy(self.review)
        for changes in ({"reviewer": None}, {"reviewer": " "}, {"reviewed_at_utc": None},
                        {"reviewed_at_utc": "2026-10-07"},
                        {"reviewed_at_utc": "2026-10-07T00:00:00+08:00"},
                        {"review_status": []}):
            with self.subTest(changes=changes):
                self.review = copy.deepcopy(clean)
                self.review.update(changes)
                self.write()
                with self.assertRaises(ValueError):
                    self.score()
        self.review = copy.deepcopy(clean)
        self.review["questions"][0]["candidates"][0]["grade"] = None
        self.write()
        with self.assertRaisesRegex(ValueError, "未评分"):
            self.score()

    def test_missing_duplicate_and_unknown_candidates_or_questions_are_rejected(self):
        for problem in ("missing_candidate", "duplicate_candidate", "unknown_candidate",
                        "missing_question", "duplicate_question", "changed_question"):
            with self.subTest(problem=problem):
                review = json.loads(self.review_path.read_bytes())
                questions = review["questions"]
                candidates = questions[0]["candidates"]
                if problem == "missing_candidate":
                    candidates.pop()
                elif problem == "duplicate_candidate":
                    candidates.append(candidates[0])
                elif problem == "unknown_candidate":
                    candidates[0]["candidate_id"] = "unknown"
                elif problem == "missing_question":
                    questions.clear()
                elif problem == "duplicate_question":
                    questions.append(questions[0])
                else:
                    questions[0]["question"] = "替换后的问题"
                self.review_path.write_text(json.dumps(review), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.score()
                self.write()

    def test_manifest_cannot_invent_rankings_or_drop_provenance(self):
        clean = copy.deepcopy(self.manifest)
        for problem in ("unknown", "duplicate", "pool", "route", "hash"):
            with self.subTest(problem=problem):
                self.manifest = copy.deepcopy(clean)
                rankings = self.manifest["questions"][0]["rankings"]
                if problem == "unknown":
                    rankings["before"][0] = "unknown"
                elif problem == "duplicate":
                    rankings["before"][1] = rankings["before"][0]
                elif problem == "pool":
                    rankings["after"] = rankings["before"]
                elif problem == "route":
                    self.manifest["routes"].append("invented")
                else:
                    self.manifest["source_report_sha256"] = ""
                self.write()
                with self.assertRaises(ValueError):
                    self.score()

    def test_duplicate_json_fields_and_missing_original_rating_field_are_rejected(self):
        self.review_path.write_text('{"questions": [], "questions": []}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "重复字段"):
            self.score()
        del self.template["questions"][0]["candidates"][0]["grade"]
        self.write()
        with self.assertRaises(ValueError):
            self.score()

    def test_cutoff_cannot_extend_beyond_exported_ranking(self):
        for top_k in (True, 0, -1, 3):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                self.score(top_k)

    def test_no_useful_pool_does_not_report_corpus_coverage(self):
        for row in self.review["questions"][0]["candidates"]:
            row.update(grade=0, evidence_quotes=[])
        self.write()
        result = self.score()
        for route in result["summary"].values():
            self.assertEqual(route["mean_pooled_ndcg_at_k"], 0)
            self.assertEqual(route["useful_hit_rate_at_k"], 0)
            self.assertIsNone(route["mean_useful_coverage_of_pool_at_k"])
            self.assertEqual(route["questions_with_useful_candidates_in_pool"], 0)

    def test_cli_preserves_existing_output_and_does_not_publish_pending_scores(self):
        output = self.root / "output.json"
        output.write_text("keep", encoding="utf-8")
        args = ["score", "--template-dir", str(self.template_dir), "--review", str(self.review_path),
                "--output", str(output), "--top-k", "2"]
        with patch("sys.argv", args), patch("sys.stderr"), self.assertRaises(SystemExit):
            main()
        self.assertEqual(output.read_text(), "keep")
        output.unlink()
        self.review = copy.deepcopy(self.template)
        self.write()
        with patch("sys.argv", args), patch("sys.stderr"), self.assertRaises(SystemExit):
            main()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
