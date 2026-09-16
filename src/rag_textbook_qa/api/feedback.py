"""Opt-in answer feedback storage for the public question-answering service."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FEEDBACK_RATINGS = frozenset({"helpful", "needs_improvement"})
FEEDBACK_REASON_ORDER = (
    "not_answered",
    "irrelevant_sources",
    "unsupported_answer",
    "incomplete",
    "too_slow",
    "other",
)
FEEDBACK_REASONS = frozenset(FEEDBACK_REASON_ORDER)
_REASON_CHECKS = {
    "not_answered": "generation",
    "irrelevant_sources": "retrieval",
    "unsupported_answer": "grounding",
    "incomplete": "generation",
    "too_slow": "performance",
    "other": "manual_review",
}


class AnswerRecordExpiredError(LookupError):
    """The answer is unknown or has aged out of the bounded in-memory registry."""


@dataclass(frozen=True)
class AnswerSnapshot:
    answer_id: str
    answered_at_utc: str
    book_id: str | None
    query: str
    status: str
    answer: str | None
    sources: list[dict[str, Any]]
    conflicts: list[str]
    timing: dict[str, Any]


class AnswerRegistry:
    """Keep recent answers in memory until a user explicitly chooses to save feedback."""

    def __init__(
        self,
        *,
        max_entries: int = 1_000,
        ttl_seconds: float = 3_600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries 必须大于 0")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds 必须大于 0")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, AnswerSnapshot]] = OrderedDict()

    def remember(
        self,
        *,
        query: str,
        book_id: str | None,
        result: Mapping[str, Any],
    ) -> str:
        answer_id = secrets.token_hex(16)
        snapshot = AnswerSnapshot(
            answer_id=answer_id,
            answered_at_utc=datetime.now(UTC).isoformat(),
            book_id=book_id,
            query=query,
            status=str(result.get("status") or "unknown"),
            answer=str(result["answer"]) if result.get("answer") is not None else None,
            sources=copy.deepcopy(list(result.get("sources") or [])),
            conflicts=[str(topic) for topic in result.get("conflicts") or []],
            timing=copy.deepcopy(dict(result.get("timing") or {})),
        )
        now = self._clock()
        with self._lock:
            self._discard_expired(now)
            self._entries[answer_id] = (now, snapshot)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return answer_id

    def resolve(self, answer_id: str) -> AnswerSnapshot:
        now = self._clock()
        with self._lock:
            self._discard_expired(now)
            entry = self._entries.get(answer_id)
        if entry is None:
            raise AnswerRecordExpiredError(answer_id)
        return entry[1]

    def _discard_expired(self, now: float) -> None:
        horizon = now - self.ttl_seconds
        while self._entries:
            _, (recorded_at, _) = next(iter(self._entries.items()))
            if recorded_at > horizon:
                return
            self._entries.popitem(last=False)


class FeedbackStore:
    """Persist explicitly submitted feedback in one cross-platform SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS answer_feedback (
                    answer_id TEXT PRIMARY KEY,
                    answered_at_utc TEXT NOT NULL,
                    feedback_at_utc TEXT NOT NULL,
                    book_id TEXT,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT,
                    sources_json TEXT NOT NULL,
                    conflicts_json TEXT NOT NULL,
                    timing_json TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    reason TEXT,
                    comment TEXT NOT NULL
                )
                """
            )

    def save(
        self,
        snapshot: AnswerSnapshot,
        *,
        rating: str,
        reason: str | None,
        comment: str,
    ) -> None:
        if rating not in FEEDBACK_RATINGS:
            raise ValueError("未知反馈类型")
        if reason is not None and reason not in FEEDBACK_REASONS:
            raise ValueError("未知反馈原因")
        feedback_at = datetime.now(UTC).isoformat()
        values = (
            snapshot.answer_id,
            snapshot.answered_at_utc,
            feedback_at,
            snapshot.book_id,
            snapshot.query,
            snapshot.status,
            snapshot.answer,
            json.dumps(snapshot.sources, ensure_ascii=False, separators=(",", ":")),
            json.dumps(snapshot.conflicts, ensure_ascii=False, separators=(",", ":")),
            json.dumps(snapshot.timing, ensure_ascii=False, separators=(",", ":")),
            rating,
            reason,
            comment,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO answer_feedback (
                    answer_id, answered_at_utc, feedback_at_utc, book_id, query,
                    status, answer, sources_json, conflicts_json, timing_json,
                    rating, reason, comment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(answer_id) DO UPDATE SET
                    feedback_at_utc=excluded.feedback_at_utc,
                    rating=excluded.rating,
                    reason=excluded.reason,
                    comment=excluded.comment
                """,
                values,
            )

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM answer_feedback ORDER BY answered_at_utc, answer_id"
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            for key in ("sources", "conflicts", "timing"):
                record[key] = json.loads(record.pop(f"{key}_json"))
            records.append(record)
        return records

    def export_jsonl(self, output: str | Path, *, overwrite: bool = False) -> int:
        destination = Path(output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "x"
        records = self.records()
        with destination.open(mode, encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(records)

    def export_candidates(self, output: str | Path, *, overwrite: bool = False) -> int:
        destination = Path(output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "x"
        candidates = build_feedback_candidates(self.records())
        payload = {
            "schema_version": 1,
            "notice": (
                "候选项不是正式评测数据；必须人工复核后，才能进入检索/生成评测或性能待办。"
            ),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        with destination.open(mode, encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return len(candidates)


def summarize_feedback(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Build aggregate product signals without exposing question or answer text."""

    total = len(records)
    helpful = sum(record.get("rating") == "helpful" for record in records)
    needs_improvement = sum(
        record.get("rating") == "needs_improvement" for record in records
    )
    reasons = {
        reason: sum(
            record.get("rating") == "needs_improvement" and record.get("reason") == reason
            for record in records
        )
        for reason in FEEDBACK_REASON_ORDER
    }
    negative_by_book: dict[str, int] = {}
    for record in records:
        if record.get("rating") != "needs_improvement":
            continue
        book_id = str(record.get("book_id") or "all_books")
        negative_by_book[book_id] = negative_by_book.get(book_id, 0) + 1

    durations = []
    for record in records:
        timing = record.get("timing")
        value = timing.get("total_seconds") if isinstance(timing, Mapping) else None
        if isinstance(value, int | float) and not isinstance(value, bool):
            duration = float(value)
            if math.isfinite(duration) and duration >= 0:
                durations.append(duration)
    durations.sort()

    return {
        "total": total,
        "ratings": {
            "helpful": helpful,
            "needs_improvement": needs_improvement,
        },
        "helpful_rate": round(helpful / total, 4) if total else None,
        "reasons": {reason: count for reason, count in reasons.items() if count},
        "negative_by_book": dict(sorted(negative_by_book.items())),
        "latency_seconds": {
            "samples": len(durations),
            "average": _rounded(sum(durations) / len(durations)) if durations else None,
            "p50": _rounded(_percentile(durations, 0.50)) if durations else None,
            "p95": _rounded(_percentile(durations, 0.95)) if durations else None,
        },
    }


def build_feedback_candidates(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group negative feedback into human-review candidates, never formal eval cases."""

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("rating") != "needs_improvement":
            continue
        question = str(record.get("query") or "").strip()
        book_name = str(record.get("book_id") or "all_books").strip()
        if not question:
            continue
        normalized_question = re.sub(r"\s+", " ", question).casefold()
        key = (book_name, normalized_question)
        candidate = grouped.get(key)
        if candidate is None:
            digest = hashlib.sha256(f"{book_name}\0{normalized_question}".encode()).hexdigest()
            candidate = {
                "candidate_id": f"feedback-{digest[:12]}",
                "question": question,
                "book_name": book_name,
                "occurrences": 0,
                "feedback_reasons": {},
                "suggested_checks": [],
                "answer_ids": [],
                "review_status": "pending",
                "relevant_sections": [],
                "ground_truth": "",
                "review_notes": "",
            }
            grouped[key] = candidate
        candidate["occurrences"] += 1
        reason = record.get("reason")
        if isinstance(reason, str) and reason in FEEDBACK_REASONS:
            reasons = candidate["feedback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
            check = _REASON_CHECKS[reason]
            if check not in candidate["suggested_checks"]:
                candidate["suggested_checks"].append(check)
        answer_id = str(record.get("answer_id") or "").strip()
        if answer_id and answer_id not in candidate["answer_ids"]:
            candidate["answer_ids"].append(answer_id)

    candidates = list(grouped.values())
    for candidate in candidates:
        candidate["feedback_reasons"] = {
            reason: candidate["feedback_reasons"][reason]
            for reason in FEEDBACK_REASON_ORDER
            if reason in candidate["feedback_reasons"]
        }
        candidate["suggested_checks"].sort()
    candidates.sort(key=lambda item: (-item["occurrences"], item["book_name"], item["question"]))
    return candidates


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _rounded(value: float) -> float:
    return round(value, 3)
