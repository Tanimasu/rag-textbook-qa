import importlib.util
import unittest
from pathlib import Path

from rag_textbook_qa.catalog import BOOK_LABELS
from rag_textbook_qa.web.constants import RAGAS_METRIC_LABELS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


class WebPackageTests(unittest.TestCase):
    def test_web_labels_reuse_the_package_catalog(self):
        self.assertEqual(BOOK_LABELS["database"], "数据库原理及应用")
        self.assertEqual(RAGAS_METRIC_LABELS["faithfulness"], "忠实度")

    def test_legacy_app_is_a_thin_compatibility_entrypoint(self):
        source = (REPOSITORY_ROOT / "project" / "app.py").read_text(encoding="utf-8")

        self.assertIn("rag_textbook_qa.web.app", source)
        self.assertNotIn("streamlit as st", source)
        self.assertNotIn("sys.path", source)

    def test_web_services_use_the_packaged_evaluator(self):
        source = (
            REPOSITORY_ROOT / "src" / "rag_textbook_qa" / "web" / "services.py"
        ).read_text(encoding="utf-8")

        self.assertIn("rag_textbook_qa.evaluation", source)
        self.assertNotIn("spec_from_file_location", source)
        self.assertNotIn("project/ragas_evaluation.py", source)

    def test_chat_uses_streaming_and_hyde_is_opt_in(self):
        chat_source = (
            REPOSITORY_ROOT / "src" / "rag_textbook_qa" / "web" / "chat_page.py"
        ).read_text(encoding="utf-8")
        layout_source = (
            REPOSITORY_ROOT / "src" / "rag_textbook_qa" / "web" / "layout.py"
        ).read_text(encoding="utf-8")

        self.assertIn("on_answer_chunk=render_chunk", chat_source)
        self.assertIn("use_hyde=enable_hyde", chat_source)
        self.assertIn('"启用 HyDE 增强检索"', layout_source)
        self.assertIn("value=False", layout_source)

    @unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit UI extra is not installed")
    def test_packaged_and_legacy_entrypoints_render_without_exceptions(self):
        from streamlit import config
        from streamlit.testing.v1 import AppTest

        entrypoints = [
            REPOSITORY_ROOT / "src" / "rag_textbook_qa" / "web" / "app.py",
            REPOSITORY_ROOT / "project" / "app.py",
        ]
        for entrypoint in entrypoints:
            with self.subTest(entrypoint=entrypoint):
                app = AppTest.from_file(str(entrypoint)).run(timeout=20)
                self.assertFalse([exception.value for exception in app.exception])
                self.assertEqual(config.get_option("client.toolbarMode"), "viewer")


if __name__ == "__main__":
    unittest.main()
