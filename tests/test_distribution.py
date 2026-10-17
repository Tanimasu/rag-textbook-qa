"""Release archives must agree on the project version and retain safe assets."""

import contextlib
import io
import runpy
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_distribution.py"))

    def make_release(self, root, *, lock_version="1.2.3", runtime_version="1.2.3", wheel_version="1.2.3",
                     missing_asset=False, private_file=False):
        wheel_files = dict.fromkeys(self.checker["WHEEL_REQUIRED"], "fixture")
        wheel_files["rag_textbook_qa/__init__.py"] = f'__version__ = "{runtime_version}"\n'
        wheel_files["rag_textbook_qa-1.2.3.dist-info/METADATA"] = f"Name: rag-textbook-qa\nVersion: {wheel_version}\n"
        if missing_asset:
            wheel_files.pop("rag_textbook_qa/api/static/history-backup.js")
        if private_file:
            wheel_files["project/.env"] = "fixture-only-not-a-secret"
        with zipfile.ZipFile(root / "rag_textbook_qa-1.2.3-py3-none-any.whl", "w") as archive:
            for name, content in wheel_files.items():
                archive.writestr(name, content)
        source_files = dict.fromkeys(self.checker["SDIST_REQUIRED"], "fixture")
        source_files["pyproject.toml"] = '[project]\nname="rag-textbook-qa"\nversion="1.2.3"\n'
        source_files["uv.lock"] = f'[[package]]\nname="rag-textbook-qa"\nversion="{lock_version}"\n'
        with tarfile.open(root / "rag_textbook_qa-1.2.3.tar.gz", "w:gz") as archive:
            for name, content in source_files.items():
                payload = content.encode()
                entry = tarfile.TarInfo("rag_textbook_qa-1.2.3/" + name)
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))

    def check_release(self, **options):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            self.make_release(root, **options)
            self.checker["check_distribution"](root)

    def test_consistent_release_passes(self):
        self.check_release()

    def test_project_lock_version_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "版本"):
            self.check_release(lock_version="1.2.2")

    def test_runtime_version_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "版本"):
            self.check_release(runtime_version="1.2.2")

    def test_wheel_metadata_version_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "版本"):
            self.check_release(wheel_version="1.2.2")

    def test_required_backup_asset_cannot_be_omitted(self):
        with self.assertRaisesRegex(ValueError, "缺少"):
            self.check_release(missing_asset=True)

    def test_private_environment_files_cannot_enter_release(self):
        with self.assertRaisesRegex(ValueError, "敏感"):
            self.check_release(private_file=True)
