"""Measure local retrieval routes with real, offline model inference on an index copy.

Uses dev questions only and never loads project/.env, calls an LLM, or changes the
source index. A historical Git ref can be compared with the working checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from rag_textbook_qa.config import Settings
from rag_textbook_qa.evaluation.retrieval import load_retrieval_questions
from rag_textbook_qa.indexing.revision import index_revision
from rag_textbook_qa.indexing.snapshot import copy_index
from rag_textbook_qa.providers import ComputeSettings
from rag_textbook_qa.providers.local import LocalEmbeddingProvider, LocalRerankerProvider
from rag_textbook_qa.rag import RAGEngine


class CountedProvider:
    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.identity, self.telemetry = provider.identity, provider.telemetry
        self.calls = self.items = 0
        self.seconds = 0.0

    def _call(self, name: str, values: Any, *args: Any) -> Any:
        self.calls += 1
        self.items += len(values)
        started = time.monotonic()
        try:
            return getattr(self.provider, name)(*args, values)
        finally:
            self.seconds += time.monotonic() - started

    def embed_queries(self, texts: Any) -> Any:
        return self._call("embed_queries", texts)

    def embed_documents(self, texts: Any) -> Any:
        return self._call("embed_documents", texts)

    def rerank(self, query: str, documents: Any) -> Any:
        return self._call("rerank", documents, query)


def historical_engine(root: Path, ref: str, directory: Path) -> type:
    source = subprocess.check_output(
        ["git", "show", f"{ref}:src/rag_textbook_qa/rag/engine.py"], cwd=root,
    )
    path = directory / "baseline_engine.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("rag_benchmark_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载基线引擎")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RAGEngine


def summaries(samples: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({sample["route"] for sample in samples})
    output = {}
    for group in groups:
        rows = [row for row in samples if row["route"] == group]
        durations = sorted(row["seconds"] for row in rows)
        output[group] = {
            "samples": len(rows),
            "median_seconds": statistics.median(durations),
            "p95_seconds": durations[math.ceil(len(durations) * 0.95) - 1],
            "mean_embedding_calls": statistics.mean(row["embedding_calls"] for row in rows),
            "mean_reranker_calls": statistics.mean(row["reranker_calls"] for row in rows),
            "mean_rerank_pairs": statistics.mean(row["rerank_pairs"] for row in rows),
        }
    return output


def peak_rss_mib() -> float | None:
    try:
        import resource
    except ImportError:  # Windows has no standard-library RSS high-water counter.
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2 if platform.system() == "Darwin" else 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--baseline-ref")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats 必须大于 0")
    if args.reranker_batch_size < 1:
        parser.error("--reranker-batch-size 必须大于 0")
    if args.output.exists():
        parser.error("输出文件已存在，请换一个路径")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    paths = Settings.load().paths
    questions = load_retrieval_questions(paths.evaluation_data / "retrieval_questions.json")
    by_book = {}
    for question in questions:
        if question.split == "dev":
            by_book.setdefault(question.book_name, question)
    selected = [by_book[book] for book in sorted(by_book)]
    embedding = CountedProvider(LocalEmbeddingProvider("BAAI/bge-large-zh-v1.5", device=args.device))
    reranker = CountedProvider(LocalRerankerProvider("BAAI/bge-reranker-base", device=args.device,
                                                   batch_size=args.reranker_batch_size))
    report: dict[str, Any] = {
        "complete": False, "started_at_utc": datetime.now(UTC).isoformat(),
        "device": args.device, "platform": platform.system(), "python": platform.python_version(),
        "machine": platform.machine(),
        "libraries": {name: version(name) for name in ("torch", "sentence-transformers", "chromadb")},
        "baseline_ref": args.baseline_ref, "split": "dev", "top_k": 5,
        "reranker_batch_size": args.reranker_batch_size,
        "question_selection": "first dev question from each book, sorted by book id",
        "repeats": args.repeats, "inference_cache": False,
        "p95_method": "nearest rank; descriptive small-sample statistic, not a production SLO",
        "embedding_fingerprint": embedding.identity.fingerprint,
        "reranker_fingerprint": reranker.identity.fingerprint,
        "engine_sha256": hashlib.sha256((paths.root / "src/rag_textbook_qa/rag/engine.py").read_bytes()).hexdigest(),
        "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        report["summary"] = summaries(report["samples"])
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)

    with tempfile.TemporaryDirectory(prefix="rag-benchmark-") as temporary, ExitStack() as stack:
        directory = Path(temporary)
        report["index_revision"] = copy_index(paths.vector_db, directory / "db")
        options = {"db_path": directory / "db", "embedding_provider": embedding,
                   "reranker_provider": reranker, "compute_settings": ComputeSettings(device=args.device),
                   "enable_llm": False, "enable_hyde": False, "verbose": False}
        engines = {"current": stack.enter_context(RAGEngine(**options))}
        if args.baseline_ref:
            engine_class = historical_engine(paths.root, args.baseline_ref, directory)
            engines["baseline"] = stack.enter_context(engine_class(**options))
        warm_started = time.monotonic()
        engines["current"].search_single_book(selected[0].book_name, selected[0].question, 5, use_hyde=False)
        report["warmup_seconds"] = time.monotonic() - warm_started
        import torch

        def synchronize() -> None:
            if args.device == "mps":
                torch.mps.synchronize()

        routes = [(f"{name}_{scope}", engine, scope)
                  for name, engine in engines.items() for scope in ("single", "all")]
        save()
        for repeat in range(args.repeats):
            for question_index, question in enumerate(selected):
                # Alternate route order so one implementation does not always run first.
                ordered = routes if (repeat + question_index) % 2 == 0 else list(reversed(routes))
                for route, engine, scope in ordered:
                    counters = embedding.calls, reranker.calls, reranker.items
                    provider_seconds = embedding.seconds, reranker.seconds
                    synchronize()
                    started = time.monotonic()
                    rows, _, _ = engine._retrieve(question.question, {"status": "disabled"},
                                                  question.book_name if scope == "single" else None,
                                                  5, False)
                    synchronize()
                    seconds = time.monotonic() - started
                    report["samples"].append({
                        "route": route, "repeat": repeat, "book_name": question.book_name,
                        "seconds": seconds, "results": len(rows),
                        "embedding_calls": embedding.calls - counters[0],
                        "reranker_calls": reranker.calls - counters[1],
                        "rerank_pairs": reranker.items - counters[2],
                        "embedding_seconds": embedding.seconds - provider_seconds[0],
                        "reranker_seconds": reranker.seconds - provider_seconds[1],
                    })
                    save()
                    print(f"{route} {question.book_name} {seconds:.3f}s", flush=True)
        if index_revision(paths.vector_db) != report["index_revision"]:
            raise RuntimeError("运行期间原索引发生更新；结果仅对应开始时的副本")
        report["process_peak_rss_mib"] = peak_rss_mib()
        report["memory_scope"] = "peak across both engines and all routes, not a per-route comparison"
        report["complete"] = True
        save()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
