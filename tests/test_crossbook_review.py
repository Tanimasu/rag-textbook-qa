import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_vectorizer_provider import FakeEmbeddingProvider, _chunks

from rag_textbook_qa.indexing import MultiBookVectorizer
from rag_textbook_qa.indexing.revision import index_revision
from rag_textbook_qa.indexing.snapshot import copy_index
from scripts.prepare_crossbook_review import prepare_review


class CrossbookReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.db = self.root / "db"
        self.chunks = self.root / "chunks.json"
        self.chunks.write_text(json.dumps(_chunks(), ensure_ascii=False), encoding="utf-8")
        self.build("os")
        self.build("database")
        self.report = {
            "complete": True, "split": "dev", "index_revision": index_revision(self.db),
            "dataset_sha256": "0" * 64,
            "cases": [{"question": "如何管理资源？", "book_name": "os", "routes": {
                "all_before": {"ranking": [["os", "chunk-1"], ["database", "chunk-1"]]},
                "all_after": {"ranking": [["os", "chunk-2"], ["database", "chunk-1"]]},
            }}],
        }
        self.report_path = self.root / "report.json"
        self.write_report()

    def build(self, book):
        with (
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()),
            MultiBookVectorizer(db_path=self.db, embedding_provider=FakeEmbeddingProvider()) as writer,
        ):
            writer.vectorize_book(self.chunks, book)

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")

    def test_review_preserves_book_identity_and_has_no_inferred_grades_or_routes(self):
        result = prepare_review(self.report_path, self.db, self.root / "review")
        template = json.loads((self.root / "review/review.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.root / "review/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(result, {"questions": 1, "candidates": 3})
        self.assertEqual(template["review_status"], "pending")
        self.assertIsNone(template["reviewer"])
        question = template["questions"][0]
        self.assertNotIn("book_name", question)
        self.assertNotIn("rankings", question)
        self.assertEqual({(row["book_name"], row["chunk_id"]) for row in question["candidates"]},
                         {("os", "chunk-1"), ("database", "chunk-1"), ("os", "chunk-2")})
        self.assertTrue(all(row["grade"] is None and row["evidence_quotes"] == [] for row in question["candidates"]))
        self.assertEqual(len(manifest["questions"][0]["rankings"]["all_before"]), 2)
        prepare_review(self.report_path, self.db, self.root / "again")
        self.assertEqual((self.root / "review/review.json").read_bytes(),
                         (self.root / "again/review.json").read_bytes())

    def test_unreviewable_inputs_do_not_publish_a_template(self):
        for problem in ("holdout", "stale_index", "missing_candidate"):
            with self.subTest(problem=problem):
                report = json.loads(json.dumps(self.report))
                if problem == "holdout":
                    report["split"] = "holdout"
                elif problem == "stale_index":
                    report["index_revision"] = "different"
                else:
                    report["cases"][0]["routes"]["all_before"]["ranking"][0][1] = "gone"
                self.report_path.write_text(json.dumps(report), encoding="utf-8")
                output = self.root / problem
                with self.assertRaises(ValueError):
                    prepare_review(self.report_path, self.db, output)
                self.assertFalse(output.exists())

    def test_copy_refuses_a_catalog_publication_during_file_copy(self):
        actual_copy = shutil.copytree

        def publish_during_copy(*args, **kwargs):
            result = actual_copy(*args, **kwargs)
            if Path(args[0]).resolve() == self.db.resolve():
                self.build("os")
            return result

        with (
            patch("rag_textbook_qa.indexing.snapshot.shutil.copytree", side_effect=publish_during_copy),
            self.assertRaisesRegex(RuntimeError, "发生更新"),
        ):
            copy_index(self.db, self.root / "copy")

    def test_copy_refuses_a_destination_within_the_source(self):
        destination = self.db / "inside"
        with self.assertRaisesRegex(ValueError, "原索引目录"):
            copy_index(self.db, destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
