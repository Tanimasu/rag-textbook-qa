import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.cli import main


class ChatCliTests(unittest.TestCase):
    def test_chat_command_delegates_to_packaged_interactive_entrypoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "src" / "rag_textbook_qa").mkdir(parents=True)
            (root / "project").mkdir()
            (root / "pyproject.toml").write_text(
                "[project]\nname='test'\n",
                encoding="utf-8",
            )
            (root / "project" / ".env").write_text(
                "RAG_QA_COMPUTE_BACKEND=local\n",
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("rag_textbook_qa.rag.interactive_main") as interactive,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = main(
                    [
                        "--workspace",
                        str(root),
                        "chat",
                        "--no-llm",
                        "--no-hyde",
                        "--no-reranker",
                    ]
                )

            self.assertEqual(exit_code, 0)
            interactive.assert_called_once_with(
                workspace=root.resolve(),
                db_path=root.resolve() / "artifacts" / "vector_db",
                enable_llm=False,
                enable_reranker=False,
                enable_hyde=False,
            )


if __name__ == "__main__":
    unittest.main()
