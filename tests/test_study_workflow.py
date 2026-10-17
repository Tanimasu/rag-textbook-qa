"""Exercise import, live index refresh, reading and observation with no real models."""

import contextlib
import io
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from rag_textbook_qa.api.app import create_api_app
from rag_textbook_qa.api.feedback import FeedbackStore
from rag_textbook_qa.api.guard import AccessGuard, GuardSettings
from rag_textbook_qa.api.imports import ImportManager
from rag_textbook_qa.indexing.snapshot import TemporaryIndexDirectory
from rag_textbook_qa.providers import ModelIdentity
from rag_textbook_qa.rag.engine import RAGEngine


class OfflineEmbedding:
    identity = ModelIdentity(task="embedding", model="offline-study-fixture", normalized=True,
                             query_instruction="")

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_queries(self, texts):
        return [[1.0, 0.0] for _ in texts]


class StudyWorkflowTests(unittest.TestCase):
    def test_imported_book_refreshes_chat_reading_and_all_inspection_stages(self):
        with (TemporaryIndexDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO())):
            root = Path(temporary)
            with RAGEngine(db_path=root / "db", embedding_provider=OfflineEmbedding(),
                           enable_llm=False, enable_reranker=False, enable_hyde=False,
                           verbose=False) as engine:
                def build(path, book, progress):
                    engine.vectorizer.vectorize_book(path, book, batch_size=1, on_progress=progress)
                manager = ImportManager(root / "imports", build)
                try:
                    guard = AccessGuard(GuardSettings(requests_per_window=30))
                    app = create_api_app(engine, guard, [], import_manager=manager,
                                         feedback_store=FeedbackStore(root / "feedback.sqlite3"))
                    with TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1") as client:
                        self.assertEqual(client.get("/v1/books").json(), [])
                        def upload(text):
                            response = client.post("/v1/imports/prepare?book_id=os&filename=教材.md",
                                                   content=text.encode(),
                                                   headers={"Content-Type": "application/octet-stream"})
                            self.assertEqual(response.status_code, 200, response.text)
                            job_id = response.json()["job_id"]
                            started = client.post(f"/v1/imports/{job_id}/start", json={"confirm_replace": True})
                            self.assertEqual(started.status_code, 200, started.text)
                            manager.close()
                            self.assertEqual(client.get(f"/v1/imports/{job_id}").json()["state"], "complete")
                            return job_id
                        upload("# 第1章\n## 1.1 进程\n" + "进程是程序的一次执行过程。" * 20)
                        self.assertEqual(client.get("/v1/books").json()[0]["book_id"], "os")
                        response = client.post("/v1/ask", json={"query": "进程", "book_id": "os"})
                        self.assertEqual(response.status_code, 200, response.text)
                        saved = response.json()
                        self.assertEqual(saved["status"], "retrieval_only")
                        source = saved["sources"][0]
                        reading_url = f"/v1/books/os/passages?chunk_id={source['chunk_id']}"
                        read = client.get(reading_url).json()
                        self.assertEqual(read["source_sha256"], source["source_sha256"])
                        inspected = client.post("/v1/retrieval/inspect", json={"query": "进程", "book_id": "os"})
                        self.assertEqual(inspected.status_code, 200, inspected.text)
                        self.assertEqual(len(inspected.json()["stages"]), 5)
                        self.assertEqual(inspected.json()["stages"][-1]["items"][0]["chunk_id"], source["chunk_id"])
                        latest_job = upload("# 第1章\n## 1.1 进程\n" + "进程具有动态性，其状态随执行而变化。" * 20)
                        current = client.get(reading_url).json()
                        self.assertNotEqual(current["source_sha256"], source["source_sha256"])
                        self.assertIn("进程是程序的一次执行", source["excerpt"])
                        self.assertIn("动态性", current["passages"][0]["excerpt"])
                        self.assertEqual(len(client.get("/v1/imports/jobs").json()["jobs"]), 2)
                        self.assertEqual(client.delete(f"/v1/imports/{latest_job}").json()["status"], "deleted")
                        self.assertEqual(len(client.get("/v1/imports/jobs").json()["jobs"]), 1)
                        self.assertIn("动态性", client.get(reading_url).json()["passages"][0]["excerpt"])
                        self.assertEqual(guard.status()["generations_remaining_today"], 200)
                finally:
                    manager.close()
