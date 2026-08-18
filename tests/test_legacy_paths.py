import importlib.util
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_legacy_constants():
    path = REPOSITORY_ROOT / "project" / "config" / "constants.py"
    spec = importlib.util.spec_from_file_location("legacy_path_constants", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载旧入口路径配置: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyPathTests(unittest.TestCase):
    def test_legacy_ui_constants_follow_the_workspace_layout(self):
        constants = load_legacy_constants()

        self.assertEqual(constants.REPOSITORY_ROOT, REPOSITORY_ROOT)
        self.assertEqual(constants.CHUNKS_DIR, REPOSITORY_ROOT / "data" / "chunks")
        self.assertEqual(
            constants.TEST_QUESTIONS_PATH,
            REPOSITORY_ROOT / "data" / "evaluation" / "test_questions.json",
        )
        self.assertEqual(
            constants.VECTOR_DB_PATH,
            REPOSITORY_ROOT / "artifacts" / "vector_db",
        )
        self.assertEqual(
            constants.RAGAS_RESULTS_PATH,
            REPOSITORY_ROOT
            / "artifacts"
            / "evaluations"
            / "ragas_evaluation_results.csv",
        )

    def test_runnable_python_files_have_no_old_windows_or_output_paths(self):
        forbidden = ("D:\\", "project/output", "./vector_db")
        python_files = sorted((REPOSITORY_ROOT / "project").rglob("*.py"))
        self.assertTrue(python_files)

        for path in python_files:
            content = path.read_text(encoding="utf-8")
            for marker in forbidden:
                with self.subTest(path=path, marker=marker):
                    self.assertNotIn(marker, content)


if __name__ == "__main__":
    unittest.main()
