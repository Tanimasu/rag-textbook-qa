"""Check two local API processes sharing import jobs and a real temporary index.

All model providers are deterministic local simulators. No existing index,
textbook, environment file, remote Worker or paid API is used.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import shutil
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any


def serve_instance(label: str, root: str, ready: Any, stop: Any, counters: Any,
                   entered: Any, release: Any, log_path: str) -> None:
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with (Path(log_path).open("w", encoding="utf-8") as log,
          contextlib.redirect_stdout(log), contextlib.redirect_stderr(log)):
        try:
            import uvicorn

            from rag_textbook_qa.api.app import create_api_app
            from rag_textbook_qa.api.feedback import FeedbackStore
            from rag_textbook_qa.api.guard import AccessGuard, GuardSettings
            from rag_textbook_qa.api.imports import ImportManager
            from rag_textbook_qa.providers import ModelIdentity
            from rag_textbook_qa.rag.engine import RAGEngine

            class Embedding:
                identity = ModelIdentity(task="embedding", model="offline-coordination-fixture",
                                         normalized=True, query_instruction="")

                def embed_documents(self, texts):
                    with counters.get_lock():
                        counters[0] += 1
                    if entered is not None:
                        entered.set()
                    if release is not None and not release.wait(20):
                        raise TimeoutError("fixture gate was not released")
                    return [[1.0, 0.0] for _ in texts]

                def embed_queries(self, texts):
                    with counters.get_lock():
                        counters[1] += 1
                    return [[1.0, 0.0] for _ in texts]

            directory = Path(root)
            with RAGEngine(db_path=directory / "db", embedding_provider=Embedding(), enable_llm=False,
                           enable_reranker=False, enable_hyde=False, verbose=False) as engine:
                def build(path, book, progress):
                    with counters.get_lock():
                        counters[2] += 1
                    engine.vectorizer.vectorize_book(path, book, on_progress=progress)

                importer = ImportManager(directory / "imports", build)
                app = create_api_app(engine, AccessGuard(GuardSettings(requests_per_window=300)), [],
                                     import_manager=importer,
                                     feedback_store=FeedbackStore(directory / f"feedback-{label}.sqlite3"))
                sock = socket.socket()
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
                server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                                      loop="asyncio", http="h11", log_level="error"))
                thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
                try:
                    thread.start()
                    deadline = time.monotonic() + 15
                    while not server.started:
                        if not thread.is_alive() or time.monotonic() > deadline:
                            raise RuntimeError("fixture server did not start")
                        time.sleep(.02)
                    ready.put({"label": label, "port": port})
                    while not stop.wait(.1):
                        parent = multiprocessing.parent_process()
                        if parent is not None and not parent.is_alive():
                            break
                finally:
                    server.should_exit = True
                    thread.join(10)
                    sock.close()
                    importer.close()
                    if thread.is_alive():
                        raise RuntimeError("fixture server did not stop")
        except BaseException as exc:
            ready.put({"label": label, "error": type(exc).__name__})
            traceback.print_exc()
            raise


def check(output: Path) -> dict[str, Any]:
    import requests

    from rag_textbook_qa import __version__
    from rag_textbook_qa.indexing.snapshot import TemporaryIndexDirectory

    if output.exists():
        raise ValueError("输出目录已经存在，请使用新目录，旧记录不会覆盖")
    output.mkdir(parents=True)
    report: dict[str, Any] = {"complete": False, "version": __version__, "synthetic_fixture": True,
                              "paid_model_calls": 0, "remote_worker_calls": 0, "steps": []}
    context = multiprocessing.get_context("spawn")
    ready, entered, release = context.Queue(), context.Event(), context.Event()
    processes, stops, counts, urls = {}, {}, {}, {}
    checked = False
    try:
        with TemporaryIndexDirectory() as temporary, requests.Session() as client:
            client.trust_env = False
            try:
                for label in ("a", "b"):
                    stops[label] = context.Event()
                    counts[label] = context.Array("i", [0, 0, 0])
                    process = context.Process(target=serve_instance, args=(
                        label, str(temporary), ready, stops[label], counts[label],
                        entered if label == "a" else None, release if label == "a" else None,
                        str(output / f"service-{label}.log")))
                    process.start()
                    processes[label] = process
                    result = ready.get(timeout=20)
                    if result.get("label") != label or "error" in result:
                        raise RuntimeError(f"fixture service startup failed: {result}")
                    urls[label] = f"http://127.0.0.1:{result['port']}"

                text = "进程是本次双服务流程测试的检索关键词。这是合成资料，不作为教材内容或正确答案。"
                payload = [{"chunk_id": "coordination_p1", "chapter": "合成流程资料", "content": text,
                            "section_h2": "任务归属",
                            "level": 1, "char_count": len(text), "has_code": False, "has_image": False}]
                response = client.post(urls["a"] + "/v1/imports/prepare", params={"book_id": "os", "filename": "fixture.json"},
                                       json=payload, timeout=5)
                response.raise_for_status()
                job_id = response.json()["job_id"]
                job_url = f"/v1/imports/{job_id}"
                started = client.post(urls["a"] + job_url + "/start", json={"confirm_replace": True}, timeout=5)
                started.raise_for_status()
                assert entered.wait(5), "first embedding did not start"
                observed = client.get(urls["b"] + job_url, timeout=5)
                observed.raise_for_status()
                assert observed.json()["state"] == "building", observed.text
                duplicate = client.post(urls["b"] + job_url + "/start", json={"confirm_replace": True}, timeout=5)
                duplicate.raise_for_status()
                assert duplicate.json()["state"] == "building", duplicate.text
                assert client.delete(urls["b"] + job_url, timeout=5).status_code == 409
                assert client.get(urls["b"] + "/v1/books", timeout=5).json() == []
                assert list(counts["b"]) == [0, 0, 0]
                report["steps"].append("Other process observes running job, does not rebuild or delete it; staging index stays hidden")

                release.set()
                deadline = time.monotonic() + 15
                while True:
                    result = client.get(urls["a"] + job_url, timeout=5).json()
                    if result["state"] == "complete":
                        break
                    assert result["state"] == "building" and time.monotonic() < deadline, result
                    time.sleep(.05)
                assert client.get(urls["b"] + job_url, timeout=5).json()["state"] == "complete"
                books = client.get(urls["b"] + "/v1/books", timeout=5).json()
                assert [(book["book_id"], book["chunks"]) for book in books] == [("os", 1)], books
                reading = "/v1/books/os/passages?chunk_id=coordination_p1"
                response = client.get(urls["b"] + reading, timeout=5)
                response.raise_for_status()
                assert response.json()["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
                answer = client.post(urls["b"] + "/v1/ask", json={"query": "进程", "book_id": "os"}, timeout=5)
                answer.raise_for_status()
                assert answer.json()["status"] == "retrieval_only", answer.text
                assert answer.json()["sources"][0]["excerpt"] == text
                report["steps"].append("Published index is refreshed, readable and retrievable from the other process")
                deadline = time.monotonic() + 5
                while True:
                    deleted = client.delete(urls["b"] + job_url, timeout=5)
                    if deleted.status_code != 409 or time.monotonic() > deadline:
                        deleted.raise_for_status()
                        break
                    time.sleep(.05)
                for label in ("a", "b"):
                    assert client.get(urls[label] + job_url, timeout=5).status_code == 404
                    assert client.get(urls[label] + reading, timeout=5).status_code == 200
                assert list(counts["a"]) == [1, 0, 1] and list(counts["b"]) == [0, 1, 0]
                report["steps"].append("Completed copies can be removed without losing either process's published index")
                checked = True
            finally:
                release.set()
                for stop in stops.values():
                    stop.set()
                for process in processes.values():
                    process.join(10)
                    if process.is_alive():
                        process.terminate()
                        process.join(5)
    finally:
        report["process_exit_codes"] = {label: process.exitcode for label, process in processes.items()}
        report["simulated_calls"] = {label: {"document_batches": counters[0], "query_batches": counters[1],
                                             "index_builds": counters[2]} for label, counters in counts.items()}
        report["complete"] = checked and report["process_exit_codes"] == {"a": 0, "b": 0}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.copyfile(__file__, output / "validation_harness.py")
        ready.close()
        ready.join_thread()
    if not report["complete"]:
        raise RuntimeError("fixture checks or process cleanup failed; inspect the saved report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新的验收输出目录")
    args = parser.parse_args()
    print(json.dumps(check(args.output.expanduser().resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
