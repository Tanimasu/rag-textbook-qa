import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_textbook_qa.cli import main


class IndexCliTests(unittest.TestCase):
    def make_workspace(self, root: Path, *, env_text: str = "") -> None:
        (root / "src" / "rag_textbook_qa").mkdir(parents=True)
        (root / "project").mkdir()
        (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
        (root / "project" / ".env").write_text(env_text, encoding="utf-8")

    def test_build_uses_packaged_vectorizer_and_workspace_db(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(
                root,
                env_text="RAG_QA_COMPUTE_BACKEND=local\nRAG_QA_DEVICE=cpu\n",
            )
            chunks_path = root / "数据库原理及应用教程_chunks.json"
            vectorizer = MagicMock()
            vectorizer.vectorize_book.return_value = "textbook_database"
            output = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "rag_textbook_qa.indexing.MultiBookVectorizer",
                    return_value=vectorizer,
                ) as vectorizer_class,
                contextlib.redirect_stdout(output),
            ):
                exit_code = main(
                    [
                        "--workspace",
                        str(root),
                        "index",
                        "build",
                        str(chunks_path),
                        "--batch-size",
                        "16",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                vectorizer_class.call_args.kwargs["db_path"],
                root.resolve() / "artifacts" / "vector_db",
            )
            vectorizer.vectorize_book.assert_called_once_with(
                chunks_path,
                "database",
                batch_size=16,
                clear_existing=True,
            )
            self.assertIn("textbook_database", output.getvalue())

    def test_list_does_not_load_model_or_require_valid_worker_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            secret = "中文-secret"
            self.make_workspace(root, env_text=f"RAG_QA_WORKER_TOKEN={secret}\n")
            books = [
                {
                    "book_name": "database",
                    "collection_name": "textbook_database",
                    "count": 42,
                    "embedding_model": "embedding-model",
                    "embedding_fingerprint": "fingerprint",
                }
            ]
            output = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("rag_textbook_qa.indexing.list_indexed_books", return_value=books),
                patch("rag_textbook_qa.indexing.MultiBookVectorizer") as vectorizer_class,
                contextlib.redirect_stdout(output),
            ):
                exit_code = main(
                    ["--workspace", str(root), "index", "list", "--json"]
                )

            self.assertEqual(exit_code, 0)
            vectorizer_class.assert_not_called()
            self.assertEqual(json.loads(output.getvalue()), books)
            self.assertNotIn(secret, output.getvalue())


if __name__ == "__main__":
    unittest.main()
