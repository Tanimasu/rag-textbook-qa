"""Deterministic metrics for textbook retrieval experiments."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

RETRIEVAL_SPLITS = ("dev", "holdout")
DEFAULT_SPLIT = "dev"

RETRIEVAL_STRATEGIES = (
    "bm25",
    "embedding",
    "hybrid",
    "hybrid-rerank",
)


@dataclass(frozen=True)
class RetrievalQuestion:
    """One retrieval question with curated relevant section markers."""

    question: str
    book_name: str
    relevant_sections: tuple[str, ...]
    # Tuning runs see "dev" only, so the holdout cannot leak into a choice of
    # parameters. Unlabelled questions count as dev.
    split: str = DEFAULT_SPLIT


def load_retrieval_questions(path: str | Path) -> list[RetrievalQuestion]:
    """Load and validate retrieval annotations without importing model dependencies."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("检索评估集必须是非空 JSON 数组")

    questions: list[RetrievalQuestion] = []
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise TypeError(f"第 {index} 条检索评估数据必须是对象")
        question = item.get("question")
        book_name = item.get("book_name")
        relevant_sections = item.get("relevant_sections")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"第 {index} 条缺少 question")
        if not isinstance(book_name, str) or not book_name.strip():
            raise ValueError(f"第 {index} 条缺少 book_name")
        if not isinstance(relevant_sections, list) or not relevant_sections:
            raise ValueError(f"第 {index} 条缺少 relevant_sections")
        if not all(isinstance(value, str) and value.strip() for value in relevant_sections):
            raise ValueError(f"第 {index} 条 relevant_sections 必须是非空字符串数组")
        split = item.get("split", DEFAULT_SPLIT)
        if split not in RETRIEVAL_SPLITS:
            raise ValueError(f"第 {index} 条 split 只能是 {' 或 '.join(RETRIEVAL_SPLITS)}")
        questions.append(
            RetrievalQuestion(
                question=question.strip(),
                book_name=book_name.strip(),
                relevant_sections=tuple(value.strip() for value in relevant_sections),
                split=split,
            )
        )
    return questions


def select_split(
    questions: Sequence[RetrievalQuestion],
    split: str,
) -> list[RetrievalQuestion]:
    """Filter questions by split; "all" keeps everything."""

    if split == "all":
        return list(questions)
    if split not in RETRIEVAL_SPLITS:
        raise ValueError(f"未知的评测划分: {split}")
    selected = [question for question in questions if question.split == split]
    if not selected:
        raise ValueError(f"划分 {split} 中没有问题")
    return selected


def _normalized(value: object) -> str:
    return "".join(str(value or "").lower().split())


# Relevance is not binary here. Failure analysis showed the retriever landing on
# a sibling of the annotated section — 3.5.2 when 3.5.3 was wanted — which a
# hit/miss metric scores the same as retrieving a different chapter entirely.
EXACT_GRADE = 3
SIBLING_GRADE = 2
CHAPTER_GRADE = 1
_SECTION_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)+)(?![\d.])")
_CHAPTER_NUMBER = re.compile(r"^\s*第\s*(\d+)\s*章")


def _section_number(text: str) -> tuple[str, ...] | None:
    match = _SECTION_NUMBER.match(text)
    return tuple(match[1].split(".")) if match else None


def grade_result(result: dict[str, Any], markers: Sequence[str]) -> int:
    """Grade exact headings, same-depth siblings and explicit chapters.

    Local list numbers such as 1.OS are not chapter identifiers. Without a
    numbered annotation, only an exact heading match can establish relevance.
    """
    path = _normalized(result_section(result))
    if any(_normalized(marker) in path for marker in markers):
        return EXACT_GRADE
    sections = [_section_number(str(result.get(field, "")))
                for field in ("section_h2", "section_h3", "section_h4")]
    chapter_match = _CHAPTER_NUMBER.match(str(result.get("chapter", "")))
    chapter = chapter_match[1] if chapter_match else None
    best = 0
    for marker in markers:
        wanted = _section_number(marker)
        explicit_chapter = _CHAPTER_NUMBER.match(marker)
        if wanted:
            if chapter and chapter != wanted[0]:
                continue
            if any(seen and len(seen) == len(wanted) and seen != wanted
                   and seen[:-1] == wanted[:-1] for seen in sections):
                best = max(best, SIBLING_GRADE)
            elif chapter == wanted[0]:
                best = max(best, CHAPTER_GRADE)
        elif explicit_chapter and chapter == explicit_chapter[1]:
            best = max(best, CHAPTER_GRADE)
    return best


def _dcg(grades: Sequence[int]) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 1)
               for rank, grade in enumerate(grades, 1))


def ndcg_at_k(grades: Sequence[int], ideal_grades: Sequence[int], top_k: int) -> float:
    """Normalize chunk-level gains against the same graded candidate universe."""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if any(grade not in (0, 1, 2, 3) for grade in (*grades, *ideal_grades)):
        raise ValueError("相关性等级必须在 0 到 3 之间")
    ideal = sorted(ideal_grades, reverse=True)[:top_k]
    best = _dcg(ideal)
    actual = _dcg(list(grades)[:top_k])
    if actual > best + 1e-9:
        raise ValueError("实际收益超过候选全集的理想收益，请检查候选全集或重复结果")
    return actual / best if best else 0.0


def result_section(result: dict[str, Any]) -> str:
    """Return the visible heading hierarchy used for relevance matching."""

    fields = ("chapter", "section_h2", "section_h3", "section_h4")
    return " > ".join(str(result.get(field, "")).strip() for field in fields if result.get(field))


def score_ranked_results(
    results: Sequence[dict[str, Any]],
    relevant_sections: Sequence[str],
    *,
    top_k: int = 5,
    candidates: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score one ranked list with section Recall@K and reciprocal rank."""

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    expected = {_normalized(section): section for section in relevant_sections}
    if not expected or "" in expected:
        raise ValueError("relevant_sections 不能为空")

    # Repeated chunk IDs must not manufacture extra relevance gain.
    seen_ids: set[str] = set()
    unique_results = []
    for result in results:
        chunk_id = result.get("chunk_id")
        if chunk_id is not None:
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
        unique_results.append(result)
    results = unique_results
    matched: set[str] = set()
    first_relevant_rank: int | None = None
    top_sections: list[str] = []
    grades: list[int] = []
    for rank, result in enumerate(results[:top_k], 1):
        section = result_section(result)
        top_sections.append(section)
        grades.append(grade_result(result, list(expected.values())))
        normalized_section = _normalized(section)
        current_matches = {
            marker
            for normalized_marker, marker in expected.items()
            if normalized_marker in normalized_section
        }
        if current_matches and first_relevant_rank is None:
            first_relevant_rank = rank
        matched.update(current_matches)

    return {
        "recall_at_k": len(matched) / len(expected),
        "reciprocal_rank": 0.0 if first_relevant_rank is None else 1 / first_relevant_rank,
        "ndcg_at_k": ndcg_at_k(
            grades,
            [grade_result(item, list(expected.values())) for item in candidates]
            if candidates is not None else grades,
            top_k,
        ),
        "ndcg_scope": "corpus" if candidates is not None else "returned_results_only",
        "grades": grades,
        "first_relevant_rank": first_relevant_rank,
        "matched_sections": sorted(matched),
        "top_sections": top_sections,
    }


def evaluate_retrieval(
    questions: Sequence[RetrievalQuestion],
    search: Callable[[RetrievalQuestion, int], Sequence[dict[str, Any]]],
    *,
    top_k: int = 5,
    candidates_by_book: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run a search strategy and return JSON-serializable aggregate metrics."""

    if not questions:
        raise ValueError("检索评估问题不能为空")

    cases: list[dict[str, Any]] = []
    for question in questions:
        started = time.monotonic()
        results = list(search(question, top_k))
        elapsed_seconds = time.monotonic() - started
        score = score_ranked_results(
            results,
            question.relevant_sections,
            top_k=top_k,
            candidates=candidates_by_book[question.book_name] if candidates_by_book else None,
        )
        cases.append(
            {
                "question": question.question,
                "book_name": question.book_name,
                "relevant_sections": list(question.relevant_sections),
                **score,
                "elapsed_seconds": round(elapsed_seconds, 6),
            }
        )

    count = len(cases)
    return {
        "question_count": count,
        "top_k": top_k,
        "mean_recall_at_k": sum(case["recall_at_k"] for case in cases) / count,
        "hit_rate_at_k": sum(case["first_relevant_rank"] is not None for case in cases) / count,
        "mrr": sum(case["reciprocal_rank"] for case in cases) / count,
        "mean_ndcg_at_k": sum(case["ndcg_at_k"] for case in cases) / count,
        "mean_latency_seconds": sum(case["elapsed_seconds"] for case in cases) / count,
        "cases": cases,
    }


def search_with_strategy(
    engine: Any,
    question: RetrievalQuestion,
    top_k: int,
    *,
    strategy: str,
) -> Sequence[dict[str, Any]]:
    """Run one explicit retrieval strategy without invoking an LLM or HyDE."""

    if strategy == "bm25":
        return engine.search_bm25(question.book_name, question.question, top_k)
    if strategy == "embedding":
        return engine.search_embedding(
            question.book_name,
            question.question,
            top_k,
            use_hyde=False,
        )
    if strategy in {"hybrid", "hybrid-rerank"}:
        return engine.search_single_book(
            question.book_name,
            question.question,
            top_k,
            use_hyde=False,
            use_reranker=strategy == "hybrid-rerank",
        )
    raise ValueError(f"未知检索策略: {strategy}")


def run_retrieval_strategies(
    engine: Any,
    questions: Sequence[RetrievalQuestion],
    strategies: Sequence[str],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate multiple retrieval strategies against the same annotations."""

    selected = tuple(dict.fromkeys(strategies))
    if not selected:
        raise ValueError("至少选择一种检索策略")
    unknown = [strategy for strategy in selected if strategy not in RETRIEVAL_STRATEGIES]
    if unknown:
        raise ValueError(f"未知检索策略: {', '.join(unknown)}")

    candidates_by_book = None
    if getattr(engine, "vectorizer", None) is not None:
        candidates_by_book = {}
        for book in sorted({question.book_name for question in questions}):
            collection = engine.vectorizer.client.get_collection(f"textbook_{book}")
            data = collection.get(include=["metadatas"])
            candidates_by_book[book] = [
                {**metadata, "chunk_id": chunk_id}
                for chunk_id, metadata in zip(data["ids"], data["metadatas"])
            ]

    results = {
        strategy: evaluate_retrieval(
            questions,
            partial(search_with_strategy, engine, strategy=strategy),
            top_k=top_k,
            candidates_by_book=candidates_by_book,
        )
        for strategy in selected
    }
    return {
        "schema_version": 2,
        "ndcg_scope": "corpus" if candidates_by_book is not None else "returned_results_only",
        "relevance_policy": "heading_hierarchy_v2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "question_count": len(questions),
        "top_k": top_k,
        "strategies": results,
    }


def save_retrieval_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write a timestamped retrieval report without overwriting earlier runs."""

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"retrieval_{timestamp}.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
