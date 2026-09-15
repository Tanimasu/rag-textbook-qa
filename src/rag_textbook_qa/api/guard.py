"""Access and cost controls for the public question-answering API.

Nothing here imports FastAPI, so the policy is testable without a server. Every
counter lives in process memory and a restart forgets it: acceptable for a
single-process demo, wrong for anything running more than one replica.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime

# Bounds memory when many distinct addresses pass through once each.
_MAX_TRACKED_CLIENTS = 10_000


class GuardError(Exception):
    """A refusal the API turns into an HTTP status."""


class AccessDenied(GuardError):
    """The caller sent no access code, or the wrong one."""


class RateLimited(GuardError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("提问太频繁了，请稍后再试")
        self.retry_after_seconds = retry_after_seconds


class Busy(GuardError):
    """Another answer held the generation slot for longer than the queue allows."""


def _positive_number(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是正整数") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return value


@dataclass(frozen=True)
class GuardSettings:
    access_code: str | None = None
    requests_per_window: int = 10
    window_seconds: float = 600
    daily_generations: int = 200
    queue_timeout_seconds: float = 90
    # Enable only behind exactly one trusted proxy, such as a Hugging Face Space.
    trust_proxy: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GuardSettings:
        environ = os.environ if environ is None else environ
        code = environ.get("RAG_QA_ACCESS_CODE") or None
        # Browsers only send ASCII header values, and the value itself must never be
        # echoed back, so the message describes the rule rather than the input.
        if code is not None and (
            not code.isascii()
            or not code.isprintable()
            or any(character.isspace() for character in code)
        ):
            raise ValueError("RAG_QA_ACCESS_CODE 只能包含可打印的 ASCII 字符，且不能含空白")
        return cls(
            access_code=code,
            requests_per_window=_positive_number(environ, "RAG_QA_RATE_LIMIT", 10),
            window_seconds=_positive_number(environ, "RAG_QA_RATE_WINDOW_SECONDS", 600),
            daily_generations=_positive_number(environ, "RAG_QA_DAILY_GENERATIONS", 200),
            queue_timeout_seconds=_positive_number(environ, "RAG_QA_QUEUE_TIMEOUT", 90),
            trust_proxy=environ.get("RAG_QA_TRUST_PROXY", "").strip().lower()
            in {"1", "true", "yes"},
        )


def _utc_today() -> date:
    return datetime.now(UTC).date()


class AccessGuard:
    def __init__(
        self,
        settings: GuardSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], date] = _utc_today,
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._today = today
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}
        self._day = today()
        self._generations = 0
        # One answer at a time: on a two-core CPU host, concurrent reranking only
        # makes every caller wait longer.
        self._slot = threading.BoundedSemaphore(1)

    @property
    def requires_access_code(self) -> bool:
        return self.settings.access_code is not None

    def check_access(self, supplied: str | None) -> None:
        expected = self.settings.access_code
        if expected is None:
            return
        if not supplied or not secrets.compare_digest(
            supplied.encode("utf-8"), expected.encode("utf-8")
        ):
            raise AccessDenied("访问口令不正确")

    def check_rate(self, client: str) -> None:
        now = self._clock()
        horizon = now - self.settings.window_seconds
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            while hits and hits[0] <= horizon:
                hits.popleft()
            if len(hits) >= self.settings.requests_per_window:
                wait = hits[0] + self.settings.window_seconds - now
                raise RateLimited(max(1, int(wait) + 1))
            hits.append(now)
            if len(self._hits) > _MAX_TRACKED_CLIENTS:
                idle = [key for key, times in self._hits.items() if times[-1] <= horizon]
                for key in idle:
                    del self._hits[key]

    def reserve_generation(self) -> bool:
        """Claim one generation from today's budget; False means answer from retrieval."""

        with self._lock:
            self._roll_day()
            if self._generations >= self.settings.daily_generations:
                return False
            self._generations += 1
            return True

    def refund_generation(self) -> None:
        """Return a claim that never reached the model, so the cap counts real calls."""

        with self._lock:
            self._roll_day()
            self._generations = max(0, self._generations - 1)

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._generations = 0

    @contextmanager
    def generation_slot(self) -> Iterator[None]:
        if not self._slot.acquire(timeout=self.settings.queue_timeout_seconds):
            raise Busy("当前排队的人比较多，请稍后再试")
        try:
            yield
        finally:
            self._slot.release()

    def status(self) -> dict[str, object]:
        with self._lock:
            self._roll_day()
            remaining = max(0, self.settings.daily_generations - self._generations)
        return {
            "access_code_required": self.requires_access_code,
            "generations_remaining_today": remaining,
        }
