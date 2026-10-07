"""Measure the configured Worker through a copied index and loopback HTTP API.

Only model inference is remote. LLM and query fallback are disabled. Credentials
are read from the existing environment/dotenv but never written to the report.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import socket
import statistics
import subprocess
import threading
import time
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import uvicorn
from dotenv import dotenv_values

from rag_textbook_qa.api.app import create_api_app
from rag_textbook_qa.api.feedback import FeedbackStore
from rag_textbook_qa.api.guard import AccessGuard, GuardSettings
from rag_textbook_qa.config import Settings
from rag_textbook_qa.evaluation.retrieval import load_retrieval_questions, run_retrieval_strategies
from rag_textbook_qa.indexing.revision import index_revision
from rag_textbook_qa.indexing.snapshot import TemporaryIndexDirectory, copy_index
from rag_textbook_qa.providers.config import ComputeSettings
from rag_textbook_qa.providers.remote import (
    RemoteEmbeddingProvider,
    RemoteRerankerProvider,
    RemoteWorkerClient,
)
from rag_textbook_qa.rag import RAGEngine


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def checked_result(body: dict[str, Any], device: str) -> dict[str, Any]:
    if body.get("status") != "retrieval_only" or not body.get("sources"):
        raise RuntimeError("真实远程 HTTP 请求未返回教材证据")
    stages = body.get("compute", {})
    if set(stages) != {"embedding", "reranker"} or any(
        stage["backend"] != "remote" or stage["device"] != device or stage["fallback_used"]
        for stage in stages.values()
    ):
        raise RuntimeError("远程验收混入其他计算后端或设备")
    return {"status": body["status"], "timing": body.get("timing"), "compute": stages,
            "sources": [{"book_id": row["book_id"], "section": row["section"],
                         "excerpt_sha256": digest(row["excerpt"])} for row in body["sources"]]}


def run(output: Path, repeats: int = 2) -> dict[str, Any]:
    if output.exists() or repeats < 1:
        raise ValueError("输出文件必须未使用过，repeats 必须大于 0")
    paths = Settings.load().paths
    values = {key: value for key, value in dotenv_values(paths.root / "project/.env").items() if value is not None}
    values.update(os.environ)
    values["RAG_QA_COMPUTE_BACKEND"] = "remote"
    compute = replace(ComputeSettings.from_env(values), query_fallback_to_local=False)
    client = RemoteWorkerClient(compute.remote_url, token=compute.remote_token, timeout=compute.remote_timeout_seconds)
    embedding = RemoteEmbeddingProvider(client, compute.embedding_model)
    reranker = RemoteRerankerProvider(client, compute.reranker_model)
    dataset = paths.evaluation_data / "retrieval_questions.json"
    frozen = paths.evaluation_data / "product_acceptance_v1.json"
    by_book = {}
    for question in load_retrieval_questions(dataset):
        if question.split == "dev":
            by_book.setdefault(question.book_name, question)
    selected = [by_book[book] for book in sorted(by_book)]
    report: dict[str, Any] = {
        "complete": False, "started_at_utc": datetime.now(UTC).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.root, text=True).strip(),
        "source_sha256": {name: digest((paths.root / name).read_bytes()) for name in (
            "scripts/benchmark_remote.py", "src/rag_textbook_qa/providers/remote.py",
            "src/rag_textbook_qa/api/app.py", "src/rag_textbook_qa/rag/engine.py",
            "src/rag_textbook_qa/evaluation/retrieval.py",
        )},
        "dataset_sha256": digest(dataset.read_bytes()), "frozen_dataset_sha256": digest(frozen.read_bytes()),
        "llm_enabled": False, "query_fallback_enabled": False, "repeats": repeats,
        "question_selection": "first dev question per book; frozen 15 are acceptance only, not tuning",
        "http_samples": [], "p95_scope": "nearest rank, descriptive small sample, not a production SLO",
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)

    save()
    try:
        health = client.request("/health")
        if health.get("status") != "ok" or health.get("protocol_version") != "1":
            raise RuntimeError("Worker 健康状态或协议不兼容")
        device, platform = health.get("device"), health.get("platform")
        if device not in {"cpu", "cuda", "mps"} or platform not in {"Windows", "Darwin", "Linux"}:
            raise RuntimeError("Worker 设备类型不受此验收脚本支持")
        for provider in (embedding, reranker):
            if health.get("models", {}).get(provider.identity.task, {}).get("fingerprint") != provider.identity.fingerprint:
                raise RuntimeError("Worker 模型指纹不匹配")
        report["worker"] = {"device": device, "platform": platform, "protocol_version": "1",
                            "model_fingerprints": {provider.identity.task: provider.identity.fingerprint
                                                   for provider in (embedding, reranker)}}
        with TemporaryIndexDirectory(prefix="rag-remote-acceptance-") as temporary, ExitStack() as stack:
            directory = Path(temporary)
            report["index_revision"] = copy_index(paths.vector_db, directory / "db")
            with contextlib.redirect_stdout(io.StringIO()):
                engine = stack.enter_context(RAGEngine(
                    db_path=directory / "db", embedding_provider=embedding, reranker_provider=reranker,
                    compute_settings=compute, enable_llm=False, enable_hyde=False, verbose=False,
                ))
            started = time.monotonic()
            for book in (selected[0].book_name, None):
                engine.ask(selected[0].question, book_name=book, use_llm=False, use_hyde=False)
            report["warmup_seconds"] = time.monotonic() - started
            guard = AccessGuard(GuardSettings(requests_per_window=1000))
            app = create_api_app(engine, guard, [], feedback_store=FeedbackStore(directory / "feedback.sqlite3"))
            sock = socket.socket()
            stack.callback(sock.close)
            sock.bind(("127.0.0.1", 0))
            base = f"http://127.0.0.1:{sock.getsockname()[1]}"
            server = uvicorn.Server(uvicorn.Config(app, loop="asyncio", http="h11", log_level="error"))
            thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
            thread.start()

            def stop() -> None:
                server.should_exit = True
                thread.join(30)
                if thread.is_alive():
                    raise RuntimeError("本机 HTTP 服务未停止")

            stack.callback(stop)
            deadline = time.monotonic() + 10
            while not server.started:
                if time.monotonic() > deadline:
                    raise RuntimeError("本机 HTTP 服务未启动")
                time.sleep(0.01)
            session = stack.enter_context(requests.Session())
            session.trust_env = False
            for repeat in range(repeats):
                for index, question in enumerate(selected):
                    for scope in (("single", "all") if (repeat + index) % 2 == 0 else ("all", "single")):
                        started = time.monotonic()
                        response = session.post(base + "/v1/ask", json={"query": question.question,
                                                "book_id": question.book_name if scope == "single" else None}, timeout=60)
                        response.raise_for_status()
                        elapsed = time.monotonic() - started
                        row = checked_result(response.json(), device)
                        row.update(scope=scope, seconds=elapsed, repeat=repeat,
                                   book_name=question.book_name, query_sha256=digest(question.question))
                        report["http_samples"].append(row)
                        save()
                        print(f"remote {repeat + 1}/{repeats} {question.book_name} {scope}: {elapsed:.3f}s", flush=True)
            with session.post(base + "/v1/ask/stream", json={"query": selected[0].question,
                              "book_id": selected[0].book_name}, stream=True, timeout=60) as response:
                response.raise_for_status()
                lines = response.text.splitlines()
                events = [line.removeprefix("event: ") for line in lines if line.startswith("event: ")]
                data = [json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")]
                if events != ["result"] or len(data) != 1:
                    raise RuntimeError("真实远程 SSE 未正常结束")
                report["sse"] = {"events": events, **checked_result(data[0], device)}
            report["frozen_acceptance"] = run_retrieval_strategies(
                engine, load_retrieval_questions(frozen), ("bm25", "embedding", "hybrid", "hybrid-rerank"),
                top_k=5, context_budget=4000,
            )
            report["final_state"] = session.get(base + "/health", timeout=5).json()
            if (report["final_state"]["generations_remaining_today"] != 200
                    or report["final_state"]["requests_in_flight"] != 0):
                raise RuntimeError("验收后队列或生成额度异常")
            report["summary"] = {}
            for scope in ("single", "all"):
                durations = sorted(row["seconds"] for row in report["http_samples"] if row["scope"] == scope)
                report["summary"][scope] = {"samples": len(durations), "median_seconds": statistics.median(durations),
                                             "p95_seconds": durations[math.ceil(len(durations) * 0.95) - 1]}
            if index_revision(paths.vector_db) != report["index_revision"]:
                raise RuntimeError("运行期间原索引发生变化")
        report["complete"] = True
    except Exception as exc:
        report["error_category"] = type(exc).__name__
        raise
    finally:
        save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    try:
        report = run(args.output, args.repeats)
    except Exception as exc:  # noqa: BLE001 - never print connection details or credentials
        parser.exit(1, f"远程验收失败: {type(exc).__name__}\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
