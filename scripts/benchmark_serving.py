"""Exercise the loopback HTTP API with cached local models and a copied index.

Only dev questions and retrieval are used. No dotenv, model downloads, remote
Worker, or LLM calls. Timings describe a small local sample, not a production SLO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import requests
import uvicorn

from rag_textbook_qa.api.app import create_api_app
from rag_textbook_qa.api.feedback import FeedbackStore
from rag_textbook_qa.api.guard import AccessGuard, GuardSettings
from rag_textbook_qa.config import Settings
from rag_textbook_qa.evaluation.retrieval import load_retrieval_questions
from rag_textbook_qa.indexing.revision import index_revision
from rag_textbook_qa.indexing.snapshot import TemporaryIndexDirectory, copy_index
from rag_textbook_qa.providers import ComputeSettings
from rag_textbook_qa.providers.local import LocalEmbeddingProvider, LocalRerankerProvider
from rag_textbook_qa.rag import RAGEngine


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def wait_until(condition: Any, message: str, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            raise RuntimeError(message)
        time.sleep(0.01)


class ObservedAdmission:
    def __init__(self, admission: Any, record: dict[str, Any]) -> None:
        self.admission, self.record = admission, record

    def try_start(self) -> bool:
        started = self.admission.try_start()
        if started:
            self.record["started_at"] = time.monotonic()
            self.record["queue_seconds"] = self.record["started_at"] - self.record["admitted_at"]
        return started

    def is_cancelled(self) -> bool:
        return self.admission.is_cancelled()

    def cancel(self) -> None:
        self.record.setdefault("cancelled_at", time.monotonic())
        self.admission.cancel()
        if "started_at" not in self.record:
            self.record.setdefault("released_at", time.monotonic())

    def release(self) -> None:
        self.record.setdefault("release_called_at", time.monotonic())
        self.admission.release()
        self.record.setdefault("released_at", time.monotonic())


class ObservedGuard(AccessGuard):
    def __init__(self) -> None:
        super().__init__(GuardSettings(max_pending_requests=3, requests_per_window=1000,
                                       queue_timeout_seconds=120))
        self.records: list[dict[str, Any]] = []

    def admit_request(self) -> Any:
        admission = super().admit_request()
        record = {"admitted_at": time.monotonic()}
        self.records.append(record)
        return ObservedAdmission(admission, record)


class ObservedReranker:
    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.identity, self.telemetry = provider.identity, provider.telemetry
        self.entered = threading.Event()

    def rerank(self, query: str, documents: Any) -> Any:
        self.entered.set()
        return self.provider.rerank(query, documents)


class ObservedEngine:
    enable_llm = False

    def __init__(self, engine: RAGEngine) -> None:
        self.engine = engine
        self.records: list[dict[str, Any]] = []
        self.active = self.peak_active = 0
        self.lock = threading.Lock()

    def list_indexed_books(self) -> Any:
        return self.engine.list_indexed_books()

    def ask(self, **kwargs: Any) -> Any:
        record = {"query_sha256": digest(kwargs["query"]), "book_id": kwargs["book_name"],
                  "started_at": time.monotonic()}
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            self.records.append(record)
        try:
            return self.engine.ask(**kwargs)
        finally:
            record["finished_at"] = time.monotonic()
            record["cancelled_before_return"] = kwargs["should_stop"]()
            with self.lock:
                self.active -= 1


def run(device: str, repeats: int, output: Path) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    paths = Settings.load().paths
    dataset = paths.evaluation_data / "retrieval_questions.json"
    dev = [question for question in load_retrieval_questions(dataset) if question.split == "dev"]
    by_book = {}
    for question in dev:
        by_book.setdefault(question.book_name, question)
    selected = [by_book[book] for book in sorted(by_book)]
    if len({question.question for question in dev[:8]}) != 8:
        raise ValueError("并发验收需要八道不同的 dev 题")
    report: dict[str, Any] = {
        "complete": False, "started_at_utc": datetime.now(UTC).isoformat(),
        "device": device, "platform": platform.platform(), "python": platform.python_version(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.root, text=True).strip(),
        "source_sha256": {name: digest((paths.root / name).read_bytes()) for name in (
            "scripts/benchmark_serving.py", "src/rag_textbook_qa/api/app.py",
            "src/rag_textbook_qa/api/guard.py", "src/rag_textbook_qa/rag/engine.py",
            "src/rag_textbook_qa/providers/local.py",
        )},
        "libraries": {name: version(name) for name in (
            "torch", "sentence-transformers", "chromadb", "fastapi", "uvicorn",
        )},
        "dataset_sha256": digest(dataset.read_bytes()), "split": "dev", "top_k": 5,
        "question_selection": "first dev question per book; burst uses first eight dev questions",
        "repeats": repeats, "llm_enabled": False, "inference_cache": False,
        "capacity": 3, "reranker_batch_size": 32, "samples": [],
        "p95_scope": "nearest rank, descriptive small sample, not a production SLO",
        "rss_scope": "whole process with both models; excludes device memory; not per-route memory or a leak test",
        "rss_source": "ps RSS in KiB when available; unavailable values are null",
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)

    def request(method: str, url: str, **kwargs: Any) -> requests.Response:
        with requests.Session() as session:
            session.trust_env = False  # Loopback requests never use an environment proxy.
            return session.request(method, url, timeout=90, **kwargs)

    def rss_mib() -> float | None:
        try:
            return float(subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True,
                stderr=subprocess.DEVNULL,
            )) / 1024
        except (OSError, subprocess.CalledProcessError, ValueError):
            return None

    report["rss_before_engine_mib"] = rss_mib()
    save()
    with TemporaryIndexDirectory(prefix="rag-http-models-") as temporary, ExitStack() as stack:
        directory = Path(temporary)
        report["index_revision"] = copy_index(paths.vector_db, directory / "db")
        embedding = LocalEmbeddingProvider("BAAI/bge-large-zh-v1.5", device=device)
        reranker = ObservedReranker(LocalRerankerProvider("BAAI/bge-reranker-base", device=device))
        engine = stack.enter_context(RAGEngine(
            db_path=directory / "db", embedding_provider=embedding, reranker_provider=reranker,
            compute_settings=ComputeSettings(device=device), enable_llm=False, enable_hyde=False,
            verbose=False,
        ))
        report["model_fingerprints"] = {"embedding": embedding.identity.fingerprint,
                                         "reranker": reranker.identity.fingerprint}
        warm_started = time.monotonic()
        for book in (selected[0].book_name, None):
            engine.ask(selected[0].question, book_name=book, use_llm=False, use_hyde=False)
        report["warmup_seconds"] = time.monotonic() - warm_started
        report["rss_after_warmup_mib"] = rss_mib()
        observed, guard = ObservedEngine(engine), ObservedGuard()
        app = create_api_app(observed, guard, [], feedback_store=FeedbackStore(directory / "feedback.sqlite3"))
        sock = socket.socket()
        stack.callback(sock.close)
        sock.bind(("127.0.0.1", 0))
        base = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, loop="asyncio", http="h11", log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()

        def stop_server() -> None:
            server.should_exit = True
            thread.join(30)
            if thread.is_alive():
                raise RuntimeError("本机 HTTP 服务未停止")

        stack.callback(stop_server)
        wait_until(lambda: server.started, "本机 HTTP 服务未启动")

        def health() -> dict[str, Any]:
            response = request("GET", base + "/health")
            response.raise_for_status()
            return response.json()

        def post(question: Any, scope: str = "all") -> dict[str, Any]:
            started = time.monotonic()
            response = request("POST", base + "/v1/ask", json={
                "query": question.question, "book_id": question.book_name if scope == "single" else None,
            })
            elapsed = time.monotonic() - started
            body = response.json()
            if response.status_code == 200:
                if body["status"] != "retrieval_only" or not body["sources"]:
                    raise RuntimeError("真实模型 HTTP 请求未返回教材证据")
                if set(body["compute"]) != {"embedding", "reranker"} or any(
                       stage["backend"] != "local" or stage["device"] != device
                       for stage in body["compute"].values()):
                    raise RuntimeError("实际模型设备与验收配置不一致")
            return {"http_status": response.status_code, "seconds": elapsed,
                    "query_sha256": digest(question.question), "scope": scope,
                    "timing": body.get("timing"), "sources": [
                        {"book_id": row["book_id"], "section": row["section"],
                         "excerpt_sha256": digest(row["excerpt"])} for row in body.get("sources", [])
                    ]}

        for repeat in range(repeats):
            for index, question in enumerate(selected):
                for scope in (("single", "all") if (repeat + index) % 2 == 0 else ("all", "single")):
                    row = post(question, scope)
                    if row["http_status"] != 200:
                        raise RuntimeError("顺序 HTTP 请求失败")
                    row.update(repeat=repeat, book_name=question.book_name,
                               queue_seconds=guard.records[-1]["queue_seconds"])
                    report["samples"].append(row)
                    save()
                    print(f"{device} {repeat + 1}/{repeats} {question.book_name} {scope}: {row['seconds']:.3f}s", flush=True)
        report["rss_after_sequential_mib"] = rss_mib()
        burst_start = len(guard.records)
        with ThreadPoolExecutor(max_workers=8) as clients:
            barrier = threading.Barrier(8)

            def burst(question: Any) -> Any:
                barrier.wait(timeout=5)
                return post(question)

            futures = [clients.submit(burst, question) for question in dev[:8]]
            rows = [future.result(timeout=90) for future in futures]
        statuses = [row["http_status"] for row in rows]
        if statuses.count(200) != 3 or statuses.count(503) != 5:
            raise RuntimeError(f"八请求并发验收失败: {statuses}")
        report["burst"] = {"requests": rows, "admissions": guard.records[burst_start:]}
        report["rss_after_burst_mib"] = rss_mib()
        save()

        def stream(question: Any) -> requests.Response:
            session = stack.enter_context(requests.Session())
            session.trust_env = False
            response = session.post(base + "/v1/ask/stream", json={"query": question.question},
                                    stream=True, timeout=(5, 90))
            stack.callback(response.close)
            response.raise_for_status()
            return response

        cancellation_start, engine_start = len(guard.records), len(observed.records)
        reranker.entered.clear()
        active = stream(dev[0])
        if not reranker.entered.wait(20):
            raise RuntimeError("取消验收未进入真实重排推理")
        queued = stream(dev[1])
        wait_until(lambda: health()["requests_queued"] == 1, "取消验收的后续请求未排队")
        queued.close()
        wait_until(lambda: health()["requests_queued"] == 0, "排队断连未归还容量")
        active.close()
        followup = post(dev[2])
        wait_until(lambda: health()["requests_in_flight"] == 0, "真实模型请求队列未排空")
        admissions = guard.records[cancellation_start:]
        calls = observed.records[engine_start:]
        if (len(admissions) != 3 or len(calls) != 2
                or not calls[0]["cancelled_before_return"]
                or "started_at" in admissions[1]
                or admissions[2]["started_at"] < admissions[0]["release_called_at"]
                or calls[1]["started_at"] < calls[0]["finished_at"]
                or followup["http_status"] != 200):
            raise RuntimeError("真实模型取消或执行槽归还验收失败")
        report["cancellation"] = {"admissions": admissions, "engine_calls": calls,
                                   "followup": followup, "native_inference_interrupted": False}
        report["final_state"] = health()
        report["peak_active_engines"] = observed.peak_active
        if observed.peak_active != 1 or report["final_state"]["generations_remaining_today"] != 200:
            raise RuntimeError("推理并发峰值或生成额度验收失败")
        report["rss_final_mib"] = rss_mib()
        report["summary"] = {}
        for scope in ("single", "all"):
            durations = sorted(row["seconds"] for row in report["samples"] if row["scope"] == scope)
            report["summary"][scope] = {"samples": len(durations), "median_seconds": statistics.median(durations),
                                         "p95_seconds": durations[math.ceil(len(durations) * 0.95) - 1]}
        if index_revision(paths.vector_db) != report["index_revision"]:
            raise RuntimeError("运行期间原索引发生变化")
        report["complete"] = True
        save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("输出文件必须未使用过，--repeats 必须大于 0")
    report = run(args.device, args.repeats, args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
