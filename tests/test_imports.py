import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from rag_textbook_qa.api.imports import MAX_UPLOAD_BYTES, ImportBusyError, ImportManager

CHUNKS = [{
    "chunk_id": "chapter_p1", "chapter": "第一章", "section_h2": "1.1 进程",
    "content": "进程是程序的一次执行。", "level": 2, "char_count": 12,
    "has_code": False, "has_image": False,
}]


class ImportManagerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "imports"
        self.builder = MagicMock()
        self.manager = ImportManager(self.root, self.builder)
        self.addCleanup(self.manager.close)

    def prepare(self):
        return self.manager.prepare("os", "教材.json", json.dumps(CHUNKS).encode())

    def test_prepare_preserves_source_and_previews_without_model_or_index_write(self):
        original = json.dumps(CHUNKS, ensure_ascii=False).encode()
        job = self.manager.prepare("os", "../../教材.json", original)
        self.assertEqual(job["state"], "prepared")
        self.assertEqual(job["filename"], "教材.json")
        self.assertEqual(job["preview"][0]["excerpt"], CHUNKS[0]["content"])
        self.assertEqual((self.root / job["job_id"] / "source.json").read_bytes(), original)
        self.assertEqual(self.manager.status(job["job_id"]), job)
        self.builder.assert_not_called()

    def test_markdown_uses_real_chunker_and_accepts_utf8_bom(self):
        content = "# 第1章\n## 1.1 进程\n" + "进程是程序的一次执行过程。" * 20
        job = self.manager.prepare("os", "教材.md", content.encode("utf-8-sig"))
        self.assertGreater(job["total_chunks"], 0)
        self.assertEqual(job["total_chunks"], 1)
        self.assertTrue(job["preview"][0]["section"].startswith("第1章"))
        self.assertIn("进程是程序", job["preview"][0]["excerpt"])
        self.builder.assert_not_called()

    def test_invalid_and_ambiguous_uploads_fail_before_building(self):
        for name, value in (("x.pdf", b"pdf"), ("x.md", b"\xff"), ("x.md", b""),
                            ("x.json", b"null"), ("x.json", b"[]"),
                            ("x.json", b'[{"content":"one","content":"two"}]'),
                            ("x.json", b'[NaN]')):
            with self.subTest(name=name, value=value), self.assertRaises((ValueError, TypeError)):
                self.manager.prepare("os", name, value)
        with self.assertRaises(ValueError):
            self.manager.prepare("os", "x.md", b"a" * (MAX_UPLOAD_BYTES + 1))
        with self.assertRaises(ValueError):
            self.manager.prepare("../../other", "x.json", json.dumps(CHUNKS).encode())
        self.builder.assert_not_called()

    def test_real_progress_single_writer_and_idempotent_start(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def build(path, book, progress):
            progress(1, 2, "embedding")
            entered.set()
            release.wait(3)
            progress(2, 2, "publishing")
        self.manager._build = MagicMock(side_effect=build)
        first, second = self.prepare(), self.prepare()
        self.manager.start(first["job_id"])
        self.assertTrue(entered.wait(3))
        progress = self.manager.status(first["job_id"])
        self.assertEqual(progress["completed_chunks"], 1)
        self.assertEqual(progress["state"], "building")
        self.assertEqual(self.manager.start(first["job_id"])["state"], "building")
        with self.assertRaises(ImportBusyError):
            self.manager.start(second["job_id"])
        release.set()
        self.manager.close()
        self.assertEqual(self.manager.status(first["job_id"])["state"], "complete")
        self.manager.start(first["job_id"])
        self.manager._build.assert_called_once()

    def test_failed_build_redacts_details_and_can_retry_preserved_input(self):
        self.builder.side_effect = [RuntimeError("private-worker-token"), None]
        job = self.prepare()
        self.manager.start(job["job_id"])
        self.manager.close()
        failed = self.manager.status(job["job_id"])
        self.assertEqual(failed["state"], "failed")
        self.assertNotIn("private-worker-token", json.dumps(failed))
        self.assertTrue((self.root / job["job_id"] / "source.json").is_file())
        self.manager.start(job["job_id"])
        self.manager.close()
        self.assertEqual(self.manager.status(job["job_id"])["state"], "complete")

    def test_changed_preview_rejects_index_build_and_restart_marks_unconfirmed_job(self):
        job = self.prepare()
        folder = self.root / job["job_id"]
        (folder / "chunks.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "发生变化"):
            self.manager.start(job["job_id"])
        self.builder.assert_not_called()
        job["state"] = "building"
        (folder / "job.json").write_text(json.dumps(job))
        restored = ImportManager(self.root, self.builder).status(job["job_id"])
        self.assertEqual(restored["state"], "interrupted")
        self.assertTrue((folder / "source.json").is_file())
        with self.assertRaises(LookupError):
            self.manager.status("../../outside")
