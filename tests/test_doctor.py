import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
