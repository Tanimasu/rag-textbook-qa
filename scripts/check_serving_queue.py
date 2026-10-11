"""Exercise the real HTTP API queue with a controlled, model-free producer.

This checks Uvicorn/Starlette disconnect behaviour, saturation, health responses
and slot ownership. The producer's artificial delay is not a retrieval benchmark.
"""

from __future__ import annotations

import argparse
import json
import socket
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests
import uvicorn

from rag_textbook_qa.api.app import create_api_app
from rag_textbook_qa.api.feedback import FeedbackStore
from rag_textbook_qa.api.guard import AccessGuard, GuardSettings
from rag_textbook_qa.llm import GenerationCancelled


class ControlledEngine:
    enable_llm = False

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.cancel_seen = threading.Event()
        self.exit_allowed = threading.Event()
        self.finish = threading.Event()
        self.burst_entered = threading.Event()
        self.burst_release = threading.Event()
        self.lock = threading.Lock()
        self.active = self.peak_active = 0
        self.calls: list[str] = []

    def ask(self, *, query: str, should_stop: Any, **kwargs: Any) -> dict[str, Any]:
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            self.calls.append(query)
        try:
            if query == "hold":
                self.entered.set()
                while not self.finish.wait(0.005):
                    if should_stop():
                        self.cancel_seen.set()
                        if not self.exit_allowed.wait(5):
                            raise RuntimeError("Test did not release the cancelled producer")
                        raise GenerationCancelled(request_sent=False)
            else:
                if query.startswith("burst-"):
                    self.burst_entered.set()
                    if not self.burst_release.wait(5):
                        raise RuntimeError("Test did not release the burst producer")
                time.sleep(0.08)
            return {"success": False, "answer": None, "context_sources": [{
                "book_name": "os", "chapter": "测试章节", "content": "队列测试用原文",
                "citation_id": 1,
            }]}
        finally:
            with self.lock:
                self.active -= 1


def wait_until(condition: Any, message: str, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            raise RuntimeError(message)
        time.sleep(0.01)


def run() -> dict[str, Any]:
    engine = ControlledEngine()
    guard = AccessGuard(GuardSettings(max_pending_requests=3, requests_per_window=1000,
                                      queue_timeout_seconds=10))
    report: dict[str, Any] = {"backend": "controlled model-free producer", "capacity": 3}
    with tempfile.TemporaryDirectory(prefix="rag-queue-") as temporary, ExitStack() as cleanup:
        cleanup.callback(engine.finish.set)
        cleanup.callback(engine.exit_allowed.set)
        cleanup.callback(engine.burst_release.set)
        app = create_api_app(engine, guard, [{"book_id": "os", "label": "操作系统", "chunks": 1}],
                             feedback_store=FeedbackStore(Path(temporary) / "feedback.sqlite3"))
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                              loop="asyncio", http="h11", log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()

        def stop_server() -> None:
            engine.finish.set()
            engine.exit_allowed.set()
            engine.burst_release.set()
            server.should_exit = True
            thread.join(5)
            sock.close()
            if thread.is_alive():
                raise RuntimeError("Local API server failed to stop")

        cleanup.callback(stop_server)
        wait_until(lambda: server.started, "Local API did not start")
        base = f"http://127.0.0.1:{port}"

        def health() -> dict[str, Any]:
            response = requests.get(base + "/health", timeout=2)
            response.raise_for_status()
            return response.json()

        def post(query: str) -> Any:
            return requests.post(base + "/v1/ask", json={"query": query}, timeout=15)

        def stream(query: str) -> Any:
            response = requests.post(base + "/v1/ask/stream", json={"query": query},
                                     stream=True, timeout=(2, 5))
            response.raise_for_status()
            cleanup.callback(response.close)
            return response

        active = stream("hold")
        if not engine.entered.wait(2):
            raise RuntimeError("Active producer did not start")
        first, second = stream("queued-a"), stream("queued-b")
        wait_until(lambda: health()["requests_queued"] == 2, "Two requests were not queued")
        started = time.monotonic()
        state = health()
        report["health_while_busy_ms"] = (time.monotonic() - started) * 1000
        report["saturated_state"] = state
        statuses = {}
        for endpoint in ("/v1/ask", "/v1/ask/stream"):
            response = requests.post(base + endpoint, json={"query": "rejected"}, timeout=2)
            statuses[endpoint] = response.status_code
            if response.status_code != 503 or "json" not in response.headers["content-type"]:
                raise RuntimeError("Full queue was not refused before a streaming response")
        report["saturation_http_statuses"] = statuses
        started = time.monotonic()
        first.close()
        second.close()
        wait_until(lambda: health()["requests_queued"] == 0, "Queued disconnect did not release capacity")
        report["queued_disconnect_release_ms"] = (time.monotonic() - started) * 1000
        if engine.calls != ["hold"]:
            raise RuntimeError("A disconnected queued request executed")
        active.close()
        if not engine.cancel_seen.wait(2):
            raise RuntimeError("Active HTTP disconnect did not signal cancellation")
        with ThreadPoolExecutor(max_workers=8) as clients:
            followup = clients.submit(post, "after-cancel")
            wait_until(lambda: health()["requests_queued"] == 1, "Follow-up was not queued")
            if engine.calls != ["hold"] or engine.peak_active != 1:
                raise RuntimeError("Next request ran before the cancelled worker exited")
            report["active_cancel_holds_slot_until_worker_exit"] = True
            engine.exit_allowed.set()
            response = followup.result(timeout=5)
            if response.status_code != 200 or response.json()["status"] != "retrieval_only":
                raise RuntimeError("A request failed after cancelled-worker cleanup")
            barrier = threading.Barrier(8)

            def burst(index: int) -> int:
                barrier.wait(timeout=3)
                return post(f"burst-{index}").status_code

            futures = [clients.submit(burst, index) for index in range(8)]
            try:
                if not engine.burst_entered.wait(2):
                    raise RuntimeError("Burst producer did not start")
                wait_until(lambda: sum(future.done() for future in futures) >= 5,
                           "Full burst did not refuse the excess clients")
                if health()["requests_in_flight"] != 3:
                    raise RuntimeError("Burst did not fill the three available slots")
            finally:
                engine.burst_release.set()
            statuses = [future.result(timeout=10) for future in futures]
            if statuses.count(200) != 3 or statuses.count(503) != 5:
                raise RuntimeError(f"Unexpected burst admission results: {statuses}")
            report["eight_client_burst_statuses"] = statuses
        wait_until(lambda: health()["requests_in_flight"] == 0, "Queue did not drain")

        # Book validation must not block the event loop during synchronous index
        # reads. Keep one catalog read waiting while a second request is rejected.
        validation_statuses = {}
        for endpoint in ("/v1/ask", "/v1/ask/stream"):
            catalog_entered, catalog_release = threading.Event(), threading.Event()
            cleanup.callback(catalog_release.set)

            def catalog(entered=catalog_entered, release=catalog_release) -> list[dict[str, Any]]:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Test did not release the catalog read")
                return [{"book_name": "os", "count": 1}]

            engine.list_indexed_books = catalog
            with ThreadPoolExecutor(max_workers=1) as clients:
                pending = clients.submit(
                    requests.post, base + endpoint,
                    json={"query": "catalog-check", "book_id": "os"}, timeout=10,
                )
                try:
                    if not catalog_entered.wait(2):
                        raise RuntimeError("Book validation did not read the catalog")
                    response = requests.post(base + "/v1/ask", json={"query": " "}, timeout=2)
                    if response.status_code != 422:
                        raise RuntimeError("Other requests stalled during book validation")
                    validation_statuses[endpoint] = response.status_code
                finally:
                    catalog_release.set()
                if pending.result(timeout=5).status_code != 200:
                    raise RuntimeError("Request failed after catalog validation resumed")
        report["validation_while_catalog_busy_http_statuses"] = validation_statuses
        report["final_state"] = health()
        report["peak_concurrent_producers"] = engine.peak_active
        if engine.peak_active != 1 or report["final_state"]["generations_remaining_today"] != 200:
            raise RuntimeError("Queue overlapped producers or consumed a generation quota")
        report["complete"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出文件已存在，请换一个路径")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
