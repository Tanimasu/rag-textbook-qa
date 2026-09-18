import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.cli import main
from rag_textbook_qa.rag.context import DEFAULT_CONTEXT_BUDGET


class ChatCliTests(unittest.TestCase):
    def make_workspace(self, root: Path) -> None:
        (root / "src" / "rag_textbook_qa").mkdir(parents=True)
        (root / "project").mkdir()
        (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
        (root / "project" / ".env").write_text(
            "RAG_QA_COMPUTE_BACKEND=local\n",
            encoding="utf-8",
        )

    def run_chat(self, root: Path, *extra: str):
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
                    *extra,
                ]
            )
        return exit_code, interactive

    def test_chat_command_delegates_to_packaged_interactive_entrypoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(root)

            exit_code, interactive = self.run_chat(root)

            self.assertEqual(exit_code, 0)
            interactive.assert_called_once_with(
                workspace=root.resolve(),
                db_path=root.resolve() / "artifacts" / "vector_db",
                enable_llm=False,
                enable_reranker=False,
                enable_hyde=False,
                enable_adjacent_context=False,
                context_budget=DEFAULT_CONTEXT_BUDGET,
            )

    def test_context_budget_flag_overrides_the_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(root)

            exit_code, interactive = self.run_chat(root, "--context-budget", "1500")

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                interactive.call_args.kwargs["context_budget"],
                1500,
            )

    def test_adjacent_context_is_explicitly_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(root)

            _, interactive = self.run_chat(root, "--adjacent-context")

            self.assertTrue(interactive.call_args.kwargs["enable_adjacent_context"])


if __name__ == "__main__":
    unittest.main()
