import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.config import Settings, WorkspaceNotFoundError, resolve_workspace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class WorkspaceResolutionTests(unittest.TestCase):
    def test_explicit_workspace_resolves_to_repository(self):
        self.assertEqual(resolve_workspace(REPOSITORY_ROOT), REPOSITORY_ROOT)

    def test_invalid_explicit_workspace_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(WorkspaceNotFoundError):
                resolve_workspace(tmp)

    def test_environment_override_is_supported(self):
        with patch.dict(os.environ, {"RAG_QA_HOME": str(REPOSITORY_ROOT)}):
            self.assertEqual(resolve_workspace(), REPOSITORY_ROOT)

    def test_editable_source_checkout_is_found_outside_repository(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(resolve_workspace(), REPOSITORY_ROOT)
            finally:
                os.chdir(previous)

    def test_settings_paths_are_absolute_and_stable(self):
        settings = Settings.load(REPOSITORY_ROOT)
        self.assertEqual(settings.paths.chunks, REPOSITORY_ROOT / "data" / "chunks")
        self.assertEqual(
            settings.paths.vector_db,
            REPOSITORY_ROOT / "artifacts" / "vector_db",
        )


if __name__ == "__main__":
    unittest.main()
