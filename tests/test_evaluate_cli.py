import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_textbook_qa.cli import main


class EvaluateCliTests(unittest.TestCase):
    def test_evaluate_command_delegates_without_loading_real_models(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "src" / "rag_textbook_qa").mkdir(parents=True)
            (root / "project").mkdir()
            (root / "data" / "evaluation").mkdir(parents=True)
            (root / "pyproject.toml").write_text(
                "[project]\nname='test'\n",
                encoding="utf-8",
            )
            (root / "project" / ".env").write_text("", encoding="utf-8")
            questions_path = root / "questions.json"
            questions_path.write_text("[]", encoding="utf-8")
            database_path = root / "custom-db"
            questions = [{"question": "什么是进程？", "book_name": "os"}]
            engine = MagicMock()
            engine.__enter__.return_value = engine

            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "rag_textbook_qa.evaluation.load_test_questions",
                    return_value=questions,
                ) as load_questions,
                patch("rag_textbook_qa.evaluation.run_evaluation") as run_evaluation,
                patch("rag_textbook_qa.rag.RAGEngine", return_value=engine) as engine_type,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = main(
                    [
                        "--workspace",
                        str(root),
                        "evaluate",
                        "--questions",
                        str(questions_path),
                        "--db-path",
                        str(database_path),
                        "--baseline",
                    ]
                )

            self.assertEqual(exit_code, 0)
            load_questions.assert_called_once_with(questions_path)
            engine_type.assert_called_once_with(
                db_path=database_path,
                enable_llm=True,
                verbose=False,
                enable_hyde=True,
            )
            run_evaluation.assert_called_once_with(
                engine,
                questions,
                root.resolve() / "artifacts" / "evaluations",
                include_baseline=True,
            )
            engine.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
