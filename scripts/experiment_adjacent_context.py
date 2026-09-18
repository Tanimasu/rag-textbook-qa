"""Compare production context packing with same-section adjacent chunk expansion.

This is a development-set experiment only. It never calls an answer model and it
does not change the product retrieval path. The current chunks artifact is used as
the source of truth for document order, while Chroma is checked for the exact same
set of chunk ids before any query is run.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHUNK_FILES = {
    "os": "操作系统_mineru_chunks.json",
    "computer_organization": "计算机组成原理_mineru_chunks.json",
    "computer_network": "计算机网络_mineru_chunks.json",
    "data_structure": "数据结构_mineru_chunks.json",
    "database": "数据库原理及应用教程_mineru_chunks.json",
}
HEADING_FIELDS = ("chapter", "section_h2", "section_h3", "section_h4")
ARMS = ("baseline", "base_first_adjacent", "interleaved_adjacent")
SHINGLE_SIZE = 16
_EVIDENCE_CHARACTER = re.compile(r"[\w\u3400-\u4dbf\u4e00-\u9fff]", re.UNICODE)


def _normalized(value: object) -> str:
    return "".join(str(value or "").lower().split())


def _evidence_characters(value: object) -> str:
    return "".join(_EVIDENCE_CHARACTER.findall(str(value or "").casefold()))


def _shingles(value: object, *, size: int = SHINGLE_SIZE) -> set[str]:
    normalized = _evidence_characters(value)
    if not normalized:
        return set()
    if len(normalized) <= size:
        return {normalized}
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def _heading_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")).strip() for field in HEADING_FIELDS)


def _same_named_section(left: dict[str, Any], right: dict[str, Any]) -> bool:
    key = _heading_key(left)
    return any(key) and key == _heading_key(right)


def _unique_by_chunk_id(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        chunk_id = str(row.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(row)
    return unique


def adjacent_candidates(
    results: Sequence[dict[str, Any]],
    corpus: Sequence[dict[str, Any]],
    *,
    placement: str,
) -> list[dict[str, Any]]:
    """Add immediate next/previous chunks only within the exact same heading path."""

    if placement not in {"base_first", "interleaved"}:
        raise ValueError("placement 必须是 base_first 或 interleaved")
    position = {str(row["chunk_id"]): index for index, row in enumerate(corpus)}

    def neighbours(anchor: dict[str, Any]) -> list[dict[str, Any]]:
        anchor_id = str(anchor.get("chunk_id", ""))
        index = position.get(anchor_id)
        if index is None:
            raise ValueError(f"检索结果不在当前 chunks 产物中: {anchor_id}")
        found: list[dict[str, Any]] = []
        # Continuations are normally in the next chunk, so test it before the
        # previous chunk. Both remain constrained to the exact heading path.
        for direction, offset in (("next", 1), ("previous", -1)):
            neighbour_index = index + offset
            if not 0 <= neighbour_index < len(corpus):
                continue
            candidate = corpus[neighbour_index]
            if not _same_named_section(anchor, candidate):
                continue
            found.append(
                {
                    **candidate,
                    "book_name": anchor.get("book_name") or candidate.get("book_name"),
                    "rank": anchor.get("rank"),
                    "method": "same-section-adjacent",
                    "adjacent_of": anchor_id,
                    "adjacent_direction": direction,
                }
            )
        return found

    if placement == "base_first":
        return _unique_by_chunk_id(
            [*results, *(row for result in results for row in neighbours(result))]
        )
    return _unique_by_chunk_id(
        row
        for result in results
        for row in (result, *neighbours(result))
    )


def section_body_shingles(
    corpus: Sequence[dict[str, Any]], relevant_sections: Sequence[str]
) -> set[str]:
    """Return the body-text proxy for all chunks under annotated headings."""

    markers = [_normalized(marker) for marker in relevant_sections]
    target: set[str] = set()
    for row in corpus:
        hierarchy = _normalized(
            " > ".join(str(row.get(field, "")) for field in HEADING_FIELDS if row.get(field))
        )
        if any(marker in hierarchy for marker in markers):
            target.update(_shingles(row.get("content", "")))
    if not target:
        raise ValueError(f"当前 chunks 中找不到标注章节: {', '.join(relevant_sections)}")
    return target


def _coverage(sources: Sequence[dict[str, Any]], expected: set[str]) -> float:
    seen: set[str] = set()
    for source in sources:
        seen.update(_shingles(source.get("content", "")))
    return len(expected & seen) / len(expected)


def _pack_arm(
    candidates: list[dict[str, Any]],
    *,
    expected: set[str],
    baseline_ids: set[str],
    context_budget: int,
) -> dict[str, Any]:
    from rag_textbook_qa.rag.context import select_context

    context, sources = select_context(candidates, max_length=context_budget)
    selected_ids = [str(source["chunk_id"]) for source in sources]
    selected = set(selected_ids)
    adjacent_ids = [
        str(source["chunk_id"]) for source in sources if source.get("adjacent_of")
    ]
    return {
        "section_body_coverage": _coverage(sources, expected),
        "context_chars_with_headers": len(context),
        "content_chars": sum(len(str(source.get("content", ""))) for source in sources),
        "sources_packed": len(sources),
        "selected_chunk_ids": selected_ids,
        "adjacent_chunks_packed": len(adjacent_ids),
        "adjacent_chunk_ids": adjacent_ids,
        "baseline_sources_retained": (
            len(baseline_ids & selected) / len(baseline_ids) if baseline_ids else 1.0
        ),
    }


def _aggregate(cases: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    values = [case["arms"][arm] for case in cases]
    baseline = [case["arms"]["baseline"] for case in cases]
    deltas = [
        value["section_body_coverage"] - base["section_body_coverage"]
        for value, base in zip(values, baseline, strict=True)
    ]
    count = len(values)
    return {
        "question_count": count,
        "mean_section_body_coverage": sum(
            value["section_body_coverage"] for value in values
        )
        / count,
        "mean_context_chars_with_headers": sum(
            value["context_chars_with_headers"] for value in values
        )
        / count,
        "mean_sources_packed": sum(value["sources_packed"] for value in values) / count,
        "mean_adjacent_chunks_packed": sum(
            value["adjacent_chunks_packed"] for value in values
        )
        / count,
        "mean_baseline_sources_retained": sum(
            value["baseline_sources_retained"] for value in values
        )
        / count,
        "questions_improved": sum(delta > 1e-12 for delta in deltas),
        "questions_regressed": sum(delta < -1e-12 for delta in deltas),
        "mean_coverage_delta_vs_baseline": sum(deltas) / count,
    }


def _load_corpora(chunks_dir: Path) -> dict[str, list[dict[str, Any]]]:
    corpora: dict[str, list[dict[str, Any]]] = {}
    for book_name, filename in CHUNK_FILES.items():
        path = chunks_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"缺少现行 chunks 产物: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"chunks 产物为空或格式错误: {path}")
        chunk_ids = [str(row.get("chunk_id", "")) for row in rows]
        if not all(chunk_ids) or len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(f"chunks 产物包含空或重复 chunk_id: {path}")
        corpora[book_name] = rows
    return corpora


def _assert_index_alignment(engine: Any, corpora: dict[str, list[dict[str, Any]]]) -> None:
    for book_name, rows in corpora.items():
        collection = engine.vectorizer.client.get_collection(f"textbook_{book_name}")
        indexed = set(collection.get(include=[])["ids"])
        artifact = {str(row["chunk_id"]) for row in rows}
        if indexed != artifact:
            raise ValueError(f"{book_name} 的向量库与 artifacts/chunks 不一致，请先重建索引")


def run_experiment(
    *,
    workspace: Path,
    questions_path: Path,
    chunks_dir: Path,
    db_path: Path,
    output_dir: Path,
    top_k: int,
    context_budget: int,
    strategy: str,
) -> Path:
    from dotenv import load_dotenv

    from rag_textbook_qa.evaluation import load_retrieval_questions, select_split
    from rag_textbook_qa.providers import ComputeSettings
    from rag_textbook_qa.rag import RAGEngine

    if top_k <= 0 or context_budget <= 0:
        raise ValueError("top_k 和 context_budget 必须大于 0")
    if strategy not in {"bm25", "hybrid-rerank"}:
        raise ValueError("strategy 必须是 bm25 或 hybrid-rerank")
    load_dotenv(workspace / "project" / ".env", override=False)
    compute = replace(ComputeSettings.from_env(), query_fallback_to_local=False)
    questions = select_split(load_retrieval_questions(questions_path), "dev")
    corpora = _load_corpora(chunks_dir)
    cases: list[dict[str, Any]] = []

    started = time.monotonic()
    with RAGEngine(
        db_path=db_path,
        enable_llm=False,
        enable_reranker=strategy == "hybrid-rerank",
        enable_hyde=False,
        verbose=False,
        compute_settings=compute,
    ) as engine:
        _assert_index_alignment(engine, corpora)
        for index, question in enumerate(questions, 1):
            if strategy == "bm25":
                results = engine.search_bm25(
                    question.book_name,
                    question.question,
                    top_k,
                )
            else:
                results = engine.search_single_book(
                    question.book_name,
                    question.question,
                    top_k,
                    use_hyde=False,
                    use_reranker=True,
                )
            corpus = corpora[question.book_name]
            expected = section_body_shingles(corpus, question.relevant_sections)
            baseline = _pack_arm(
                list(results),
                expected=expected,
                baseline_ids=set(),
                context_budget=context_budget,
            )
            baseline_ids = set(baseline["selected_chunk_ids"])
            baseline["baseline_sources_retained"] = 1.0
            arms = {"baseline": baseline}
            for arm, placement in (
                ("base_first_adjacent", "base_first"),
                ("interleaved_adjacent", "interleaved"),
            ):
                arms[arm] = _pack_arm(
                    adjacent_candidates(results, corpus, placement=placement),
                    expected=expected,
                    baseline_ids=baseline_ids,
                    context_budget=context_budget,
                )
            cases.append(
                {
                    "question": question.question,
                    "book_name": question.book_name,
                    "relevant_sections": list(question.relevant_sections),
                    "retrieved_chunk_ids": [str(row["chunk_id"]) for row in results],
                    "section_body_shingles": len(expected),
                    "arms": arms,
                }
            )
            print(f"[{index}/{len(questions)}] {question.book_name}: 完成")

        embedding = engine.vectorizer.embedding_provider
        reranker = engine.reranker
        backend = {
            **compute.safe_summary(),
            "embedding_identity": embedding.identity.as_dict(),
            "reranker_identity": reranker.identity.as_dict() if reranker else None,
            "embedding_runtime_device": getattr(embedding, "remote_device", compute.device),
            "reranker_runtime_device": getattr(reranker, "remote_device", compute.device),
        }

    report = {
        "schema_version": 1,
        "experiment": "same_section_adjacent_context",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_split": "dev",
        "selection_warning": "仅用于选择方案；未查看 holdout 的本实验结果",
        "answer_model_called": False,
        "retrieval_strategy": strategy,
        "compute_provider_called": strategy != "bm25",
        "top_k": top_k,
        "context_budget": context_budget,
        "adjacency_policy": "artifact_order_exact_heading_next_then_previous_v1",
        "coverage_policy": "annotated_section_body_normalized_character_shingles_v1",
        "coverage_warning": "章节正文覆盖率是结构代理指标，不等于答案证据充分率",
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "backend": backend,
        "aggregates": {arm: _aggregate(cases, arm) for arm in ARMS},
        "cases": cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_dir / f"adjacent_context_{timestamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--chunks-dir", type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-budget", type=int, default=4000)
    parser.add_argument(
        "--strategy",
        choices=("bm25", "hybrid-rerank"),
        default="hybrid-rerank",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = args.workspace.expanduser().resolve()
    path = run_experiment(
        workspace=workspace,
        questions_path=(
            args.questions or workspace / "data" / "evaluation" / "retrieval_questions.json"
        ).resolve(),
        chunks_dir=(args.chunks_dir or workspace / "artifacts" / "chunks").resolve(),
        db_path=(args.db_path or workspace / "artifacts" / "vector_db").resolve(),
        output_dir=(
            args.output_dir
            or workspace / "artifacts" / "evaluations" / "adjacent-context-dev"
        ).resolve(),
        top_k=args.top_k,
        context_budget=args.context_budget,
        strategy=args.strategy,
    )
    print(f"实验报告: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
