import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def start_blocked_import(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def build(path, book, progress):
            entered.set()
            release.wait(10)
        self.manager._build = MagicMock(side_effect=build)
        job = self.prepare()
        self.manager.start(job["job_id"])
        self.assertTrue(entered.wait(3))
        return job

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

    def test_invalid_schema_leaves_no_unmanageable_upload_and_preserves_other_jobs(self):
        existing = self.prepare()
        before = {path.name for path in self.root.iterdir()}
        with self.assertRaisesRegex(ValueError, "缺少字段"):
            self.manager.prepare("os", "bad.json", b'[{"content":"incomplete"}]')
        self.assertEqual({path.name for path in self.root.iterdir()}, before)
        self.assertEqual(self.manager.status(existing["job_id"]), existing)
        self.assertEqual(self.manager.list_jobs()["unreadable_count"], 0)
        self.builder.assert_not_called()

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

    def test_recent_jobs_skip_corrupt_metadata_and_discard_only_selected_copy(self):
        first, second = self.prepare(), self.prepare()
        corrupt = self.root / ("b" * 32)
        corrupt.mkdir()
        (corrupt / "job.json").write_text("not json")
        listed = self.manager.list_jobs()
        self.assertEqual({job["job_id"] for job in listed["jobs"]}, {first["job_id"], second["job_id"]})
        self.assertEqual(listed["unreadable_count"], 1)
        self.manager.discard(first["job_id"])
        self.assertFalse((self.root / first["job_id"]).exists())
        self.assertTrue((self.root / second["job_id"] / "source.json").is_file())
        self.builder.assert_not_called()

    def test_recent_jobs_tolerate_another_service_removing_a_listed_folder(self):
        keep, removed = self.prepare(), self.prepare()
        folder = self.manager.root / removed["job_id"]
        other = ImportManager(self.root, self.builder)
        original_stat, deleted = Path.stat, False

        def stat(path, *args, **kwargs):
            nonlocal deleted
            if path == folder and not deleted:
                deleted = True
                other.discard(removed["job_id"])
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", stat):
            listed = self.manager.list_jobs()
        self.assertTrue(deleted)
        self.assertEqual([job["job_id"] for job in listed["jobs"]], [keep["job_id"]])
        self.assertEqual(listed["unreadable_count"], 0)
        self.assertTrue((self.root / keep["job_id"] / "source.json").is_file())
        self.builder.assert_not_called()

    def test_running_job_cannot_be_discarded(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.manager._build = lambda *_: (entered.set(), release.wait(3))
        job = self.prepare()
        self.manager.start(job["job_id"])
        self.assertTrue(entered.wait(3))
        with self.assertRaises(ImportBusyError):
            self.manager.discard(job["job_id"])
        self.assertTrue((self.root / job["job_id"] / "chunks.json").is_file())
        release.set()

    def test_long_upload_name_preserves_supported_extension(self):
        job = self.manager.prepare("os", "a" * 200 + ".json", json.dumps(CHUNKS).encode())
        self.assertLessEqual(len(job["filename"]), 160)
        self.assertTrue(job["filename"].endswith(".json"))
        self.assertEqual(job["total_chunks"], 1)

    def test_other_manager_keeps_running_job_in_building_state(self):
        job = self.start_blocked_import()
        other = ImportManager(self.root, self.builder)
        self.assertEqual(other.status(job["job_id"])["state"], "building")
        self.assertEqual(self.manager.status(job["job_id"])["state"], "building")

    def test_other_manager_cannot_discard_an_active_import_copy(self):
        job = self.start_blocked_import()
        other = ImportManager(self.root, self.builder)
        with self.assertRaises(ImportBusyError):
            other.discard(job["job_id"])
        self.assertTrue((self.root / job["job_id"] / "chunks.json").is_file())

    def test_other_manager_repeated_start_observes_same_running_job(self):
        job = self.start_blocked_import()
        other = ImportManager(self.root, self.builder)
        self.addCleanup(other.close)
        self.assertEqual(other.start(job["job_id"])["state"], "building")
        other.close()
        self.builder.assert_not_called()

    def test_another_process_observes_and_protects_running_job_without_building(self):
        job = self.start_blocked_import()
        script = """
import json, sys
from pathlib import Path
from rag_textbook_qa.api.imports import ImportManager, ImportBusyError
builds = []
manager = ImportManager(Path(sys.argv[1]), lambda *args: builds.append(True))
result = {"state": manager.status(sys.argv[2])["state"]}
result["start_state"] = manager.start(sys.argv[2])["state"]
try:
    manager.discard(sys.argv[2])
    result["discard_blocked"] = False
except ImportBusyError:
    result["discard_blocked"] = True
manager.close()
result["builds"] = len(builds)
print(json.dumps(result))
"""
        child = subprocess.run([sys.executable, "-c", script, str(self.root), job["job_id"]],
                               capture_output=True, text=True, timeout=10, check=True)
        observed = json.loads(child.stdout.strip().splitlines()[-1])
        self.assertEqual(observed, {"state": "building", "start_state": "building",
                                    "discard_blocked": True, "builds": 0})
        self.assertTrue((self.root / job["job_id"] / "chunks.json").is_file())

    def test_process_exit_releases_job_lease_and_allows_explicit_retry(self):
        job = self.prepare()
        script = """
import os, sys, threading
from pathlib import Path
from rag_textbook_qa.api.imports import ImportManager
entered = threading.Event()
def build(*args):
    entered.set()
    threading.Event().wait(30)
manager = ImportManager(Path(sys.argv[1]), build)
manager.start(sys.argv[2])
if not entered.wait(3):
    os._exit(2)
os._exit(0)
"""
        subprocess.run([sys.executable, "-c", script, str(self.root), job["job_id"]],
                       capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual(self.manager.status(job["job_id"])["state"], "interrupted")
        self.manager.start(job["job_id"])
        self.manager.close()
        self.assertEqual(self.manager.status(job["job_id"])["state"], "complete")
        self.builder.assert_called_once()

    def test_old_pending_completion_does_not_overwrite_a_new_running_attempt(self):
        job = self.prepare()
        original_write = self.manager._write
        def fail_completion(value):
            if value["state"] == "complete":
                raise OSError("storage failure")
            original_write(value)
        with patch.object(self.manager, "_write", side_effect=fail_completion):
            self.manager.start(job["job_id"])
            self.manager.close()
        entered, release = threading.Event(), threading.Event()
        def build(path, book, progress):
            entered.set()
            release.wait(10)
        other = ImportManager(self.root, build)
        self.addCleanup(other.close)
        self.addCleanup(release.set)
        self.assertEqual(other.status(job["job_id"])["state"], "interrupted")
        other.start(job["job_id"])
        self.assertTrue(entered.wait(3))
        self.assertEqual(self.manager.status(job["job_id"])["state"], "building")
        disk = json.loads((self.root / job["job_id"] / "job.json").read_text())
        self.assertEqual(disk["state"], "building")

    def test_thread_start_failure_does_not_block_later_import_or_retry(self):
        first, second = self.prepare(), self.prepare()
        with (patch("rag_textbook_qa.api.imports.threading.Thread.start", side_effect=RuntimeError("no thread")),
              self.assertRaises(RuntimeError)):
            self.manager.start(first["job_id"])
        self.assertEqual(self.manager.status(first["job_id"])["state"], "failed")
        self.manager.start(second["job_id"])
        self.manager.close()
        self.assertEqual(self.manager.status(second["job_id"])["state"], "complete")
        self.manager.start(first["job_id"])
        self.manager.close()
        self.assertEqual(self.manager.status(first["job_id"])["state"], "complete")
        self.assertEqual(self.builder.call_count, 2)

    def test_completed_build_is_not_reported_failed_when_completion_metadata_write_fails(self):
        job = self.prepare()
        original_write = self.manager._write
        def disk_full_on_completion(value):
            if value["state"] == "complete":
                raise OSError("private disk path")
            original_write(value)
        with patch.object(self.manager, "_write", side_effect=disk_full_on_completion):
            self.manager.start(job["job_id"])
            self.manager.close()
            observed = self.manager.status(job["job_id"])
            self.assertEqual(observed["state"], "complete")
            self.assertFalse(observed["metadata_persisted"])
            self.assertNotIn("private disk path", observed["message"])
            self.manager.start(job["job_id"])
            self.builder.assert_called_once()
        # Once storage is available, reading progress persists the known result.
        self.assertEqual(self.manager.status(job["job_id"])["state"], "complete")
        restored = ImportManager(self.root, self.builder)
        self.assertEqual(restored.status(job["job_id"])["state"], "complete")
