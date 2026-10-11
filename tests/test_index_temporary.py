import tempfile
import unittest
from unittest.mock import patch

from rag_textbook_qa.indexing.snapshot import TemporaryIndexDirectory


def windows_error(code):
    error = PermissionError("native file is still open")
    error.winerror = code
    return error


class TemporaryIndexDirectoryTests(unittest.TestCase):
    def directory(self):
        directory = TemporaryIndexDirectory()
        self.addCleanup(directory.cleanup)
        return directory

    def test_waits_for_a_transient_windows_sharing_violation(self):
        directory = self.directory()
        with (
            patch("rag_textbook_qa.indexing.snapshot.sys.platform", "win32"),
            patch.object(tempfile.TemporaryDirectory, "cleanup", side_effect=[windows_error(32), None]) as cleanup,
            patch("rag_textbook_qa.indexing.snapshot.time.sleep") as sleep,
        ):
            directory.cleanup()
        self.assertEqual(cleanup.call_count, 2)
        sleep.assert_called_once_with(0.01)

    def test_persistent_windows_handle_leak_still_fails(self):
        directory = self.directory()
        with (
            patch("rag_textbook_qa.indexing.snapshot.sys.platform", "win32"),
            patch.object(tempfile.TemporaryDirectory, "cleanup", side_effect=windows_error(32)),
            patch("rag_textbook_qa.indexing.snapshot.time.monotonic", side_effect=[0, 2]),
            self.assertRaises(PermissionError),
        ):
            directory.cleanup()

    def test_other_permission_errors_and_platforms_fail_immediately(self):
        directory = self.directory()
        for platform, code in (("win32", 5), ("linux", 32), ("darwin", 32)):
            with (
                self.subTest(platform=platform, code=code),
                patch("rag_textbook_qa.indexing.snapshot.sys.platform", platform),
                patch.object(tempfile.TemporaryDirectory, "cleanup", side_effect=windows_error(code)) as cleanup,
                patch("rag_textbook_qa.indexing.snapshot.time.sleep") as sleep,
                self.assertRaises(PermissionError),
            ):
                directory.cleanup()
            self.assertEqual(cleanup.call_count, 1)
            sleep.assert_not_called()
