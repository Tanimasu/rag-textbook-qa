"""Opt-in answer feedback storage for the public question-answering service."""

from __future__ import annotations

import copy
import json
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
FEEDBACK_REASONS = frozenset(
    {
        "not_answered",
        "irrelevant_sources",
        "unsupported_answer",
        "incomplete",
        "too_slow",
        "other",
    }
)


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
