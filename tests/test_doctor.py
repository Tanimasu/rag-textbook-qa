import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.cli import main
from rag_textbook_qa.config import Settings
from rag_textbook_qa.diagnostics.doctor import collect_diagnostics

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HEAVY_MODULES = {
    "chromadb",
    "docling",
    "mineru",
    "openai",
    "ragas",
    "sentence_transformers",
    "streamlit",
    "torch",
    "torchvision",
    "transformers",
}


class DoctorTests(unittest.TestCase):
    def test_doctor_does_not_import_heavy_runtimes(self):
        before = HEAVY_MODULES.intersection(sys.modules)
        checks = collect_diagnostics(Settings.load(REPOSITORY_ROOT))
        after = HEAVY_MODULES.intersection(sys.modules)

        self.assertTrue(checks)
        self.assertEqual(after, before)

    def test_doctor_json_is_machine_readable(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(
                ["--workspace", str(REPOSITORY_ROOT), "doctor", "--json"]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["workspace"]["root"], str(REPOSITORY_ROOT))
        self.assertGreater(len(payload["checks"]), 0)

    def test_doctor_reports_the_current_data_layout(self):
        checks = collect_diagnostics(Settings.load(REPOSITORY_ROOT))
        by_name = {check.name: check for check in checks}

        self.assertNotIn("legacy-assets", by_name)
        self.assertIn(by_name["data:raw-pdfs"].status, {"ok", "optional"})
        for name in ("data:parsed", "data:cleaned", "data:chunks", "data:evaluation"):
            with self.subTest(name=name):
                self.assertEqual(by_name[name].status, "ok")
        self.assertIn(by_name["artifact:vector-db"].status, {"ok", "optional"})
        self.assertEqual(by_name["artifact:evaluations"].status, "ok")

    def test_doctor_distinguishes_required_and_optional_dependencies(self):
        checks = collect_diagnostics(Settings.load(REPOSITORY_ROOT))
        by_name = {check.name: check for check in checks}

        for import_name in ("chromadb", "dotenv", "jieba", "openai", "rank_bm25", "tqdm"):
            with self.subTest(import_name=import_name):
                self.assertIn(by_name[f"module:{import_name}"].status, {"ok", "missing"})
        self.assertIn(
            by_name["module:sentence_transformers"].status,
            {"ok", "optional"},
        )

    def test_doctor_reports_compute_backend_without_network_access(self):
        with patch.dict(
            "os.environ",
            {
                "RAG_QA_COMPUTE_BACKEND": "remote",
                "RAG_QA_REMOTE_URL": "http://100.64.0.10:8765",
                "RAG_QA_WORKER_TOKEN": "not-for-output",
            },
            clear=True,
        ):
            checks = collect_diagnostics(Settings.load(REPOSITORY_ROOT))

        compute = next(check for check in checks if check.name == "compute-backend")
        self.assertEqual(compute.status, "ok")
        self.assertIn("http://100.64.0.10:8765", compute.detail)
        self.assertNotIn("not-for-output", compute.detail)


if __name__ == "__main__":
    unittest.main()
