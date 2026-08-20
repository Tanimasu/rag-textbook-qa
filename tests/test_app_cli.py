import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.cli import main


class AppCliTests(unittest.TestCase):
    def make_workspace(self, root: Path, *, backend: str = "remote") -> None:
        (root / "src" / "rag_textbook_qa").mkdir(parents=True)
        (root / "project").mkdir()
        (root / "pyproject.toml").write_text(
            "[project]\nname='test'\n",
            encoding="utf-8",
        )
        (root / "project" / "app.py").write_text("# test app\n", encoding="utf-8")
        (root / "project" / ".env").write_text(
            "\n".join(
                [
                    f"RAG_QA_COMPUTE_BACKEND={backend}",
                    "RAG_QA_REMOTE_URL=http://100.64.0.10:8765",
                    "RAG_QA_WORKER_TOKEN=app-cli-secret",
                ]
            ),
            encoding="utf-8",
        )

    def test_app_launches_from_workspace_with_safe_overrides(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(root)
            output = io.StringIO()
            completed = subprocess.CompletedProcess(args=[], returncode=0)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("rag_textbook_qa.cli.importlib.util.find_spec", return_value=object()),
                patch("rag_textbook_qa.cli.subprocess.run", return_value=completed) as run,
                contextlib.redirect_stdout(output),
            ):
                exit_code = main(
                    [
                        "--workspace",
                        str(root),
                        "app",
                        "--backend",
                        "local",
                        "--device",
                        "cpu",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "9000",
                        "--no-browser",
                    ]
                )

            self.assertEqual(exit_code, 0)
            command = run.call_args.args[0]
            self.assertEqual(
                command[:5],
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(root.resolve() / "project" / "app.py"),
                ],
            )
            self.assertIn("--server.address=0.0.0.0", command)
            self.assertIn("--server.port=9000", command)
            self.assertIn("--server.headless=true", command)
            self.assertEqual(run.call_args.kwargs["cwd"], root.resolve())
            child_environment = run.call_args.kwargs["env"]
            self.assertEqual(child_environment["RAG_QA_COMPUTE_BACKEND"], "local")
            self.assertEqual(child_environment["RAG_QA_DEVICE"], "cpu")
            self.assertEqual(child_environment["RAG_QA_HOME"], str(root.resolve()))
            self.assertNotIn("app-cli-secret", output.getvalue())

    def test_remote_launch_does_not_require_local_model_packages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(root)

            def find_spec(module: str):
                return object() if module == "streamlit" else None

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("rag_textbook_qa.cli.importlib.util.find_spec", side_effect=find_spec),
                patch(
                    "rag_textbook_qa.cli.subprocess.run",
                    return_value=subprocess.CompletedProcess(args=[], returncode=0),
                ) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = main(["--workspace", str(root), "app", "--no-browser"])

            self.assertEqual(exit_code, 0)
            run.assert_called_once()

    def test_local_launch_explains_missing_local_models_extra(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(root, backend="local")
            error = io.StringIO()

            def find_spec(module: str):
                return object() if module == "streamlit" else None

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("rag_textbook_qa.cli.importlib.util.find_spec", side_effect=find_spec),
                patch("rag_textbook_qa.cli.subprocess.run") as run,
                contextlib.redirect_stderr(error),
                self.assertRaises(SystemExit) as raised,
            ):
                main(["--workspace", str(root), "app"])

            self.assertEqual(raised.exception.code, 1)
            self.assertIn("--extra local-models", error.getvalue())
            run.assert_not_called()

    def test_app_rejects_invalid_port_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(root)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("rag_textbook_qa.cli.subprocess.run") as run,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main(["--workspace", str(root), "app", "--port", "70000"])

            self.assertEqual(raised.exception.code, 1)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
