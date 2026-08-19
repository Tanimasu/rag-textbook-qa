import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.cli import main
from rag_textbook_qa.providers.base import DEFAULT_QUERY_INSTRUCTION, ModelIdentity


class WorkerCliTests(unittest.TestCase):
    def make_workspace(self, root: Path, *, token: str) -> None:
        (root / "src" / "rag_textbook_qa").mkdir(parents=True)
        (root / "project").mkdir()
        (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
        (root / "project" / ".env").write_text(
            "\n".join(
                [
                    "RAG_QA_COMPUTE_BACKEND=remote",
                    "RAG_QA_REMOTE_URL=http://100.64.0.10:8765",
                    f"RAG_QA_WORKER_TOKEN={token}",
                    "RAG_QA_EMBEDDING_MODEL=embedding-model",
                    "RAG_QA_RERANKER_MODEL=reranker-model",
                ]
            ),
            encoding="utf-8",
        )

    @staticmethod
    def health_payload() -> dict[str, object]:
        embedding = ModelIdentity(
            task="embedding",
            model="embedding-model",
            normalized=True,
            query_instruction=DEFAULT_QUERY_INSTRUCTION,
        )
        reranker = ModelIdentity(task="reranker", model="reranker-model")
        return {
            "status": "ok",
            "protocol_version": "1",
            "device": "cuda",
            "models": {
                "embedding": embedding.as_dict(),
                "reranker": reranker.as_dict(),
            },
        }

    def test_check_only_requests_health_and_prints_safe_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            token = "cli-test-secret"
            self.make_workspace(root, token=token)
            output = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "rag_textbook_qa.providers.remote.RemoteWorkerClient.request",
                    return_value=self.health_payload(),
                ) as request,
                contextlib.redirect_stdout(output),
            ):
                exit_code = main(
                    ["--workspace", str(root), "worker", "check", "--json"]
                )

            self.assertEqual(exit_code, 0)
            request.assert_called_once_with("/health")
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["http_status"], 200)
            self.assertEqual(summary["device"], "cuda")
            self.assertEqual(summary["models"]["embedding"], "embedding-model")
            self.assertNotIn(token, output.getvalue())

    def test_process_environment_conflict_warns_without_leaking_tokens(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            file_token = "file-secret"
            process_token = "process-secret"
            self.make_workspace(root, token=file_token)
            error = io.StringIO()

            with (
                patch.dict(
                    os.environ,
                    {"RAG_QA_WORKER_TOKEN": process_token},
                    clear=True,
                ),
                patch(
                    "rag_textbook_qa.providers.remote.RemoteWorkerClient.request",
                    return_value=self.health_payload(),
                ),
                contextlib.redirect_stderr(error),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = main(
                    ["--workspace", str(root), "worker", "check", "--json"]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("进程环境变量", error.getvalue())
            self.assertNotIn(file_token, error.getvalue())
            self.assertNotIn(process_token, error.getvalue())


if __name__ == "__main__":
    unittest.main()
