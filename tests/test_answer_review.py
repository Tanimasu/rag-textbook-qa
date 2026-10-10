"""A saved answer cannot silently change its question, key or evidence."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_answer_review import prepare_review


class AnswerReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        raw = "第一行\n申请改为1，归还改为0。\n第三行\n".encode()
        (self.root / "book.md").write_bytes(raw)
        self.questions = [{
            "id": "q1", "question": "如何标记？", "ground_truth": "申请1，归还0。",
            "book_name": "os", "split": "holdout", "evidence": {
                "path": "book.md", "sha256": hashlib.sha256(raw).hexdigest(),
                "start_line": 2, "end_line": 2,
            },
        }]
        self.answers = [{"question": "如何标记？", "ground_truth": "申请1，归还0。",
                         "answer": "申请1，归还0。【参考资料 2】"}]
        self.output = self.root / "out"
        self.write()

    def write(self):
        for name, value in (("questions.json", self.questions), ("answers.json", self.answers)):
            (self.root / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def prepare(self):
        return prepare_review(self.root / "questions.json", self.root / "answers.json",
                              self.root, self.output, run_label="historical test fixture")

    def test_export_records_gaps_without_inventing_grades_or_completion(self):
        summary = self.prepare()
        self.assertEqual(summary["matched_answers"], 1)
        self.assertEqual(summary["missing_contexts"], 1)
        self.assertEqual(summary["missing_finish_reason"], 1)
        template = json.loads((self.output / "review.json").read_text())
        manifest = json.loads((self.output / "manifest.json").read_text())
        row = template["questions"][0]
        self.assertEqual(row["evidence_excerpt"], "申请改为1，归还改为0。")
        self.assertEqual(row["citation_ids_in_answer"], [2])
        self.assertIsNone(row["review"]["grounding"])
        self.assertIsNone(row["recorded_finish_reason"])
        self.assertEqual(manifest["review_template_sha256"],
                         hashlib.sha256((self.output / "review.json").read_bytes()).hexdigest())
        self.assertIn("not current-version", template["scope"])

    def test_ambiguous_or_stale_input_does_not_publish_any_output(self):
        questions, answers = copy.deepcopy(self.questions), copy.deepcopy(self.answers)
        for problem in ("duplicate_answer", "duplicate_question", "different_key", "stale_evidence",
                        "invalid_lines", "outside_workspace", "bad_contexts"):
            with self.subTest(problem=problem):
                self.questions, self.answers = copy.deepcopy(questions), copy.deepcopy(answers)
                if problem == "duplicate_answer":
                    self.answers.append(copy.deepcopy(self.answers[0]))
                elif problem == "duplicate_question":
                    self.questions.append(copy.deepcopy(self.questions[0]))
                elif problem == "different_key":
                    self.answers[0]["ground_truth"] = "旧答案要点"
                elif problem == "stale_evidence":
                    self.questions[0]["evidence"]["sha256"] = "0" * 64
                elif problem == "invalid_lines":
                    self.questions[0]["evidence"]["start_line"] = True
                elif problem == "outside_workspace":
                    self.questions[0]["evidence"]["path"] = "../book.md"
                else:
                    self.answers[0]["contexts"] = "不是数组"
                self.write()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_join_uses_question_text_and_exposes_missing_and_extra_answers(self):
        self.answers[0]["question"] = "另外的问题"
        self.write()
        summary = self.prepare()
        self.assertEqual((summary["missing_answers"], summary["unmatched_saved_answers"]), (1, 1))
        row = json.loads((self.output / "review.json").read_text())["questions"][0]
        self.assertIsNone(row["answer"])
        self.assertEqual(row["answer_status"], "missing")

    def test_current_exports_preserve_sources_and_refuse_conflicting_identity(self):
        saved = self.answers[0]
        saved.update(question_id="q1", context="资料2正文", contexts=["资料2正文"],
                     context_sources=[{"citation_id": 2, "context_text": "资料2正文"}],
                     finish_reason="stop")
        self.write()
        summary = self.prepare()
        self.assertEqual(summary["missing_contexts"], 0)
        row = json.loads((self.output / "review.json").read_text())["questions"][0]
        self.assertEqual(row["recorded_context_sources"], saved["context_sources"])
        self.assertEqual(row["recorded_finish_reason"], "stop")
        original = copy.deepcopy(saved)
        for key, value in (("question_id", "q2"), ("context", "不同正文"),
                           ("context_sources", [{"context_text": "错误片段"}])):
            with self.subTest(key=key):
                self.output = self.root / key
                self.answers[0] = {**original, key: value}
                self.write()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_existing_output_and_failed_export_preserve_inputs(self):
        original = (self.root / "answers.json").read_bytes()
        self.output.mkdir()
        (self.output / "keep").write_text("existing")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual((self.output / "keep").read_text(), "existing")
        self.output = self.root / "failed"
        with (patch.object(Path, "write_text", side_effect=OSError("disk failure")),
              self.assertRaises(OSError)):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual((self.root / "answers.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
