"""Replay saved development retrieval with local context expansion only.

Reads the current Chroma SQLite catalog in a read-only transaction. No engine,
provider, model, dotenv, or network client is created. Saved rankings must match
both the development dataset and the indexed source bodies before replay.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import sqlite3
import statistics
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_textbook_qa.evaluation.retrieval import (
    load_retrieval_questions,
    score_source_evidence_coverage,
    select_split,
)
from rag_textbook_qa.indexing.revision import index_revision
from rag_textbook_qa.json_utils import loads_strict
from rag_textbook_qa.rag.adjacent import (
    append_same_section_neighbours,
    order_indexed_chunks,
)
from rag_textbook_qa.rag.context import select_context
from rag_textbook_qa.rag.supplement import (
    ExcerptBudgetExceeded,
    pack_selected_excerpts,
    paragraph_spans,
)
from rag_textbook_qa.rag.tokenizer import BM25Tokenizer, heading_terms
from scripts.experiment_adjacent_context import _shingles, section_body_shingles

HEADING_FIELDS = ("chapter", "section_h2", "section_h3", "section_h4")
ARMS = (
    "baseline", "same_section", "parent_radius_1", "parent_radius_2",
    "compact_parent", "question_parent", "bridge_parent",
)
_EXERCISE_HEADING = re.compile(r"习题|练习题|思考题|选择题|填空题|自测题|复习题")
_QUERY_STOP_WORDS = {"的", "了", "是", "在", "中", "有", "和", "与", "及", "对",
                     "哪些", "什么", "如何", "怎么", "为什么", "一个", "一种"}


def load_index_corpora(db_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read bodies/metadata from the SQLite schema used by local Chroma indexes."""
    sqlite_path = db_path.resolve() / "chroma.sqlite3"
    with closing(sqlite3.connect(f"{sqlite_path.as_uri()}?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        collections = connection.execute(
            "SELECT id, name FROM collections WHERE substr(name, 1, 9) = 'textbook_'"
        ).fetchall()
        corpora = {}
        for collection_id, name in collections:
            book = name.removeprefix("textbook_")
            rows = connection.execute(
                """SELECT e.embedding_id, m.key, m.string_value, m.int_value,
                          m.float_value, m.bool_value
                   FROM embeddings e JOIN segments s ON s.id = e.segment_id
                   LEFT JOIN embedding_metadata m ON m.id = e.id
                   WHERE s.collection = ? AND s.scope = 'METADATA'
                   ORDER BY e.id, m.key""",
                (collection_id,),
            ).fetchall()
            by_id: dict[str, dict[str, Any]] = {}
            for identifier, key, string, integer, floating, boolean in rows:
                values = by_id.setdefault(identifier, {})
                if key in values:
                    raise ValueError(f"索引片段重复: {book}/{identifier}")
                if key is not None:
                    values[key] = next(
                        (v for v in (string, integer, floating, boolean) if v is not None),
                        None,
                    )
            if not by_id or any(
                not isinstance(values.get("chroma:document"), str)
                or values.get("book_name") != book
                for values in by_id.values()
            ):
                raise ValueError(f"索引正文或教材身份不完整: {book}")
            corpora[book] = order_indexed_chunks(
                {
                    "ids": list(by_id),
                    "documents": [v["chroma:document"] for v in by_id.values()],
                    "metadatas": [
                        {k: value for k, value in v.items() if k != "chroma:document"}
                        for v in by_id.values()
                    ],
                },
                book,
            )
    if not corpora:
        raise ValueError("索引没有教材")
    return corpora


def parent_neighbours(
    results: Sequence[dict[str, Any]],
    corpora: Mapping[str, Sequence[dict[str, Any]]],
    *,
    radius: int,
) -> list[dict[str, Any]]:
    """Experimental append-only expansion across subsections of one named H2.

    Neighbours are visited next, previous at each distance, in anchor rank order.
    Every intervening chunk must share chapter and H2, so this cannot jump over a
    section boundary. All ranked evidence stays first with its original identity.
    """
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius 必须为正整数")
    positions = {
        book: {str(row["chunk_id"]): index for index, row in enumerate(corpus)}
        for book, corpus in corpora.items()
    }
    seen = {(str(r["book_name"]), str(r["chunk_id"])) for r in results}
    expanded = list(results)
    for anchor in results:
        book, identifier = str(anchor["book_name"]), str(anchor["chunk_id"])
        corpus = corpora.get(book)
        index = positions.get(book, {}).get(identifier)
        key = tuple(str(anchor.get(field, "")).strip() for field in HEADING_FIELDS[:2])
        if corpus is None or index is None or not all(key):
            continue
        for distance in range(1, radius + 1):
            for direction, sign in (("next", 1), ("previous", -1)):
                target = index + sign * distance
                if not 0 <= target < len(corpus):
                    continue
                path = [corpus[index + sign * step] for step in range(1, distance + 1)]
                if any(
                    tuple(str(row.get(f, "")).strip() for f in HEADING_FIELDS[:2]) != key
                    for row in path
                ):
                    continue
                neighbour = corpus[target]
                identity = (book, str(neighbour["chunk_id"]))
                if identity in seen:
                    continue
                expanded.append({
                    **neighbour,
                    "rank": anchor.get("rank"),
                    "method": "parent-section-adjacent-experiment",
                    "adjacent_of": identifier,
                    "adjacent_direction": direction,
                    "adjacent_distance": distance,
                })
                seen.add(identity)
    return expanded


def _coverage(sources: Sequence[dict[str, Any]], expected: set[str]) -> float:
    seen: set[str] = set()
    for row in sources:
        seen.update(_shingles(row.get("content", "")))
    return len(seen & expected) / len(expected)


def compact_parent_context(
    ranked: Sequence[dict[str, Any]], corpora: Mapping[str, Sequence[dict[str, Any]]],
    *, context_budget: int = 4000,
) -> tuple[str, list[dict[str, Any]]]:
    """A local experimental arm: at most two introductory prose paragraphs/chunk.

    Excludes exercise headings, code and tables; limits all additional prose to
    1200 characters and eight sources. Keeps complete paragraphs or skips them,
    preserving original evidence and its numbering. This is not query relevance
    scoring and remains a replay option rather than a production policy.
    """
    candidates = parent_neighbours(ranked, corpora, radius=2)[len(ranked):]
    selection: dict[str, Any] = {"selected": []}
    context, sources = pack_selected_excerpts(
        ranked, candidates, selection, context_budget=context_budget,
        max_sources=8, max_excerpt_chars=1200,
    )
    for candidate in candidates:
        if len(selection["selected"]) >= 8:
            break
        if any(_EXERCISE_HEADING.search(str(candidate.get(f, ""))) for f in HEADING_FIELDS):
            continue
        paragraphs = paragraph_spans(candidate["content"])
        eligible = [p for p in paragraphs if p["selectable"]]
        if not eligible:
            continue
        first = last = eligible[0]["index"]
        if first + 1 < len(paragraphs) and paragraphs[first + 1]["selectable"]:
            last += 1
        proposed = {"selected": [*selection["selected"], {
            "chunk_id": candidate["chunk_id"],
            "start_paragraph": first, "end_paragraph": last,
        }]}
        try:
            next_context, next_sources = pack_selected_excerpts(
                ranked, candidates, proposed, context_budget=context_budget,
                max_sources=8, max_excerpt_chars=1200,
            )
        except ExcerptBudgetExceeded:
            continue
        selection, context, sources = proposed, next_context, next_sources
    return context, sources


def question_parent_context(
    query: str, ranked: Sequence[dict[str, Any]],
    corpora: Mapping[str, Sequence[dict[str, Any]]], *, context_budget: int = 4000,
) -> tuple[str, list[dict[str, Any]]]:
    """Experimental query-word selection over all safe one/two-paragraph spans.

    In particular, explanations after a table or image can compete with the
    introduction. Scores use positive local IDF and logarithmic term counts;
    they do not establish that an excerpt explains the question completely.
    All addition/preservation limits are identical to compact_parent_context.
    """
    if not isinstance(query, str):
        raise TypeError("问题必须为文本")
    candidates = [row for row in parent_neighbours(ranked, corpora, radius=2)[len(ranked):]
                  if not any(_EXERCISE_HEADING.search(str(row.get(f, "")))
                             for f in HEADING_FIELDS)]
    selection: dict[str, Any] = {"selected": []}
    context, sources = pack_selected_excerpts(
        ranked, candidates, selection, context_budget=context_budget,
        max_sources=8, max_excerpt_chars=1200,
    )
    tokenizer = BM25Tokenizer()
    terms = {t.lower() for t in tokenizer(query)} - _QUERY_STOP_WORDS
    if not terms:
        return context, sources
    windows, paragraph_terms = [], []
    for candidate_index, candidate in enumerate(candidates):
        paragraphs = paragraph_spans(candidate["content"])
        paragraph_terms.extend(set(tokenizer(p["text"].lower()))
                               for p in paragraphs if p["selectable"])
        for first, paragraph in enumerate(paragraphs):
            if not paragraph["selectable"]:
                continue
            for last in (first, first + 1):
                if last >= len(paragraphs) or not paragraphs[last]["selectable"]:
                    continue
                text = candidate["content"][paragraph["start"]:paragraphs[last]["end"]]
                tokens = tokenizer(text.lower())
                windows.append((candidate_index, first, last, tokens))
    weights = {term: 1 + math.log((1 + len(paragraph_terms)) / (
        1 + sum(term in tokens for tokens in paragraph_terms)
    )) for term in terms}
    scored = [(sum(weights[t] * (1 + math.log(tokens.count(t)))
                   for t in terms & set(tokens)), index, first, last)
              for index, first, last, tokens in windows]
    seen = set()
    for score, index, first, last in sorted(scored, key=lambda row: (-row[0], *row[1:])):
        identifier = candidates[index]["chunk_id"]
        if score <= 0 or identifier in seen:
            continue
        if len(selection["selected"]) >= 8:
            break
        proposed = {"selected": [*selection["selected"], {
            "chunk_id": identifier, "start_paragraph": first, "end_paragraph": last,
        }]}
        try:
            next_context, next_sources = pack_selected_excerpts(
                ranked, candidates, proposed, context_budget=context_budget,
                max_sources=8, max_excerpt_chars=1200,
            )
        except ExcerptBudgetExceeded:
            continue
        selection, context, sources = proposed, next_context, next_sources
        seen.add(identifier)
    return context, sources


def bridge_parent_context(
    query: str, ranked: Sequence[dict[str, Any]],
    corpora: Mapping[str, Sequence[dict[str, Any]]], *, context_budget: int = 4000,
) -> tuple[str, list[dict[str, Any]]]:
    """Experimental question matching with explicitly named concept dependencies.

    Among delivered anchors, the largest query/title token overlap identifies
    a parent section to visit first (ties use retrieval order). A neighbour's
    subsection heading can bridge a missing query term only when that exact
    heading term occurs in the delivered anchor body. Its first two prose
    paragraphs precede question-matched windows of up to three paragraphs in
    that parent. Without a named dependency, the existing two-paragraph query
    selection remains the fallback. A queried concept itself is not a bridge.

    This heuristic does not establish relevance or answer correctness. It uses
    neither diagnostic evidence spans nor question-specific concept lists.
    """
    if not isinstance(query, str):
        raise TypeError("问题必须为文本")
    candidates = [row for row in parent_neighbours(ranked, corpora, radius=2)[len(ranked):]
                  if not any(_EXERCISE_HEADING.search(str(row.get(f, "")))
                             for f in HEADING_FIELDS)]
    selection: dict[str, Any] = {"selected": []}
    context, sources = pack_selected_excerpts(
        ranked, candidates, selection, context_budget=context_budget,
        max_sources=8, max_excerpt_chars=1200,
    )
    tokenizer = BM25Tokenizer()
    terms = {t.lower() for t in tokenizer(query)} - _QUERY_STOP_WORDS
    if not terms:
        return context, sources
    anchors = {(row["book_name"], row["chunk_id"]): row for row in sources}
    overlaps = {
        identity: len(terms & set(tokenizer(" ".join(
            str(row.get(field, "")) for field in HEADING_FIELDS[1:]
        ).lower()))) for identity, row in anchors.items()
    }
    primary = max(overlaps, key=overlaps.get) if overlaps else None
    primary_key = (None if primary is None or overlaps[primary] == 0 else (
        primary[0], *(str(anchors[primary].get(field, "")).strip()
                      for field in HEADING_FIELDS[:2]),
    ))
    bridge_windows = {}
    for index, candidate in enumerate(candidates):
        key = (candidate["book_name"], *(str(candidate.get(field, "")).strip()
                                        for field in HEADING_FIELDS[:2]))
        anchor = anchors.get((candidate["book_name"], candidate["adjacent_of"]))
        paragraphs = paragraph_spans(candidate["content"])
        bridges = [term.lower() for term in heading_terms(
            str(candidate.get(field, "")) for field in HEADING_FIELDS[2:]
        ) if anchor is not None and key == primary_key and not term.isdigit()
            and term.lower() not in query.lower()
            and term.lower() in anchor["content"].lower()]
        # Match the concept at the beginning, not merely a passing mention in
        # a later paragraph. The bridge cannot jump over an introductory image.
        if bridges and paragraphs and paragraphs[0]["selectable"] and any(
            paragraphs[0]["text"].lower().startswith(term) for term in bridges
        ):
            bridge_windows[index] = int(len(paragraphs) > 1 and paragraphs[1]["selectable"])
    bridge_key = primary_key if bridge_windows else None
    windows, paragraph_terms = [], []
    for index, candidate in enumerate(candidates):
        paragraphs = paragraph_spans(candidate["content"])
        paragraph_terms.extend(set(tokenizer(p["text"].lower()))
                               for p in paragraphs if p["selectable"])
        key = (candidate["book_name"], *(str(candidate.get(field, "")).strip()
                                        for field in HEADING_FIELDS[:2]))
        if index in bridge_windows:
            last = bridge_windows[index]
            text = candidate["content"][paragraphs[0]["start"]:paragraphs[last]["end"]]
            windows.append((2, index, 0, last, tokenizer(text.lower())))
        for first, paragraph in enumerate(paragraphs):
            if not paragraph["selectable"]:
                continue
            count = 3 if key == bridge_key else 2
            for last in range(first, min(first + count, len(paragraphs))):
                if not paragraphs[last]["selectable"]:
                    break
                text = candidate["content"][paragraph["start"]:paragraphs[last]["end"]]
                windows.append((int(key == bridge_key), index, first, last,
                                tokenizer(text.lower())))
    weights = {term: 1 + math.log((1 + len(paragraph_terms)) / (
        1 + sum(term in tokens for tokens in paragraph_terms)
    )) for term in terms}
    scored = [(priority, sum(weights[t] * (1 + math.log(tokens.count(t)))
                            for t in terms & set(tokens)), index, first, last)
              for priority, index, first, last, tokens in windows]
    seen = set()
    for priority, score, index, first, last in sorted(
        scored, key=lambda row: (-row[0], -row[1], *row[2:]),
    ):
        identifier = candidates[index]["chunk_id"]
        if (score <= 0 and priority != 2) or identifier in seen:
            continue
        if len(selection["selected"]) >= 8:
            break
        proposed = {"selected": [*selection["selected"], {
            "chunk_id": identifier, "start_paragraph": first, "end_paragraph": last,
        }]}
        try:
            next_context, next_sources = pack_selected_excerpts(
                ranked, candidates, proposed, context_budget=context_budget,
                max_sources=8, max_excerpt_chars=1200,
            )
        except ExcerptBudgetExceeded:
            continue
        selection, context, sources = proposed, next_context, next_sources
        seen.add(identifier)
    return context, sources


def replay(
    *, capture_dir: Path, db_path: Path, questions_path: Path,
    output_dir: Path, evidence_path: Path | None = None,
) -> dict[str, Any]:
    revision = index_revision(db_path)
    protocol_path = capture_dir / "protocol.json"
    protocol = loads_strict(protocol_path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(questions_path.read_bytes()).hexdigest()
    if (
        protocol.get("complete") is not True or protocol.get("split") != "dev"
        or protocol.get("index_revision") != revision
        or protocol.get("dataset_sha256") != dataset_hash
        or protocol.get("top_k") != 5 or protocol.get("context_budget") != 4000
    ):
        raise ValueError("回放协议与当前开发题集、索引或预算不一致")
    questions = select_split(load_retrieval_questions(questions_path), "dev")
    expected_questions = {(q.book_name, q.question): q for q in questions}
    if len(expected_questions) != len(questions):
        raise ValueError("开发题存在重复教材/问题")
    spans = load_retrieval_questions(evidence_path) if evidence_path else []
    if any(
        q.split != "dev" or q.evidence is None
        or (q.book_name, q.question) not in expected_questions for q in spans
    ):
        raise ValueError("证据片段必须对应开发题并声明原文行号和哈希")
    corpora = load_index_corpora(db_path)
    indexed = {book: {r["chunk_id"]: r for r in rows} for book, rows in corpora.items()}
    cases, capture_hashes, seen = [], {}, set()
    for path in sorted(capture_dir.glob("dev-*.json")):
        raw = path.read_bytes()
        captured = loads_strict(raw.decode("utf-8"))
        identity = (captured["book_name"], captured["question"])
        question = expected_questions.get(identity)
        if question is None or identity in seen:
            raise ValueError(f"回放含非开发题或重复问题: {path.name}")
        if tuple(captured["relevant_sections"]) != question.relevant_sections:
            raise ValueError(f"标注章节改变: {path.name}")
        seen.add(identity)
        current = captured["variants"]["current"]
        for row in current["ranked"]:
            source = indexed.get(identity[0], {}).get(row["chunk_id"])
            if (
                source is None or row.get("book_name") != identity[0]
                or row["content"] != source["content"]
                or any(row.get(f, "") != source.get(f, "") for f in HEADING_FIELDS)
            ):
                raise ValueError(f"保存的候选与当前索引原文不一致: {path.name}")
        ranked = current["ranked"][:5]
        baseline_context, baseline_sources = select_context(ranked, 4000)
        if baseline_context != current["context"]:
            raise ValueError(f"上下文打包已改变，不能直接回放: {path.name}")
        expected = section_body_shingles(corpora[identity[0]], question.relevant_sections)
        candidates = {
            "baseline": ranked,
            "same_section": append_same_section_neighbours(ranked, corpora),
            "parent_radius_1": parent_neighbours(ranked, corpora, radius=1),
            "parent_radius_2": parent_neighbours(ranked, corpora, radius=2),
        }
        compact_context, compact_sources = compact_parent_context(ranked, corpora)
        question_context, question_sources = question_parent_context(
            question.question, ranked, corpora,
        )
        bridge_context, bridge_sources = bridge_parent_context(question.question, ranked, corpora)
        arms = {}
        for name in ARMS:
            context, sources = (
                (bridge_context, bridge_sources) if name == "bridge_parent"
                else (question_context, question_sources) if name == "question_parent"
                else (compact_context, compact_sources) if name == "compact_parent"
                else select_context(candidates[name], 4000)
            )
            if sources[:len(baseline_sources)] != baseline_sources:
                raise ValueError(f"原有证据被修改或挤出: {path.name}/{name}")
            evidence = []
            for span in spans:
                if (span.book_name, span.question) == identity:
                    score = score_source_evidence_coverage(
                        ranked, span.evidence, sources=sources,
                    )
                    corpus_score = score_source_evidence_coverage(
                        corpora[identity[0]], span.evidence,
                        top_k=len(corpora[identity[0]]),
                    )
                    score["indexed_corpus_coverage"] = corpus_score[
                        "source_evidence_coverage_at_k"
                    ]
                    evidence.append(score)
            arms[name] = {
                "context": context, "sources": sources,
                "context_chars": len(context),
                "added_sources": len(sources) - len(baseline_sources),
                "added_truncated_sources": sum(
                    bool(s.get("truncated")) for s in sources[len(baseline_sources):]
                ),
                "baseline_preserved": True,
                "section_body_coverage": _coverage(sources, expected),
                "evidence_spans": evidence,
            }
        cases.append({
            "id": captured["id"], "book_name": identity[0], "question": identity[1],
            "relevant_sections": question.relevant_sections, "arms": arms,
        })
        capture_hashes[path.name] = hashlib.sha256(raw).hexdigest()
    if seen != set(expected_questions):
        raise ValueError("回放未完整覆盖开发题集")
    if index_revision(db_path) != revision:
        raise ValueError("读取期间索引改变，请重新运行")
    summary = {}
    for name in ARMS:
        values = [case["arms"][name] for case in cases]
        deltas = [v["section_body_coverage"] - c["arms"]["baseline"]["section_body_coverage"]
                  for v, c in zip(values, cases, strict=True)]
        summary[name] = {
            "questions": len(cases),
            "mean_context_chars": statistics.mean(v["context_chars"] for v in values),
            "mean_section_body_coverage": statistics.mean(
                v["section_body_coverage"] for v in values
            ),
            "questions_improved": sum(d > 1e-12 for d in deltas),
            "questions_regressed": sum(d < -1e-12 for d in deltas),
            "baseline_preserved_questions": sum(v["baseline_preserved"] for v in values),
        }
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "index_revision": revision, "dataset_sha256": dataset_hash,
        "capture_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "capture_sha256": capture_hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted({Path(inspect.getfile(value)).resolve() for value in (
                index_revision, load_retrieval_questions, score_source_evidence_coverage,
                select_context, order_indexed_chunks, pack_selected_excerpts,
                BM25Tokenizer, section_body_shingles,
            )})
        },
        "evidence_manifest_sha256": (
            hashlib.sha256(evidence_path.read_bytes()).hexdigest() if evidence_path else None
        ),
        "indexed_books": {b: len(rows) for b, rows in sorted(corpora.items())},
        "split": "dev", "reuses_saved_rankings": True,
        "model_calls": 0, "paid_calls": 0, "remote_calls": 0,
        "default_changed": False,
        "metric_scope": "Exact source-text coverage; no generated-answer correctness",
        "evidence_review": "Selected by Codex for development diagnosis; not expert gold",
        "summary": summary, "cases": cases,
    }
    if output_dir.resolve().is_relative_to(db_path.resolve()):
        raise ValueError("输出目录不能位于原索引中")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=Path("artifacts/vector_db"))
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/retrieval_questions.json"))
    parser.add_argument("--evidence-spans", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = replay(
        capture_dir=args.capture_dir, db_path=args.db_path,
        questions_path=args.questions, evidence_path=args.evidence_spans,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
