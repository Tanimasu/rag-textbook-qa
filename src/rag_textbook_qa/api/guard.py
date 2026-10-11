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

# Bounds memory when many distinct addresses and request scopes pass through once each.
_MAX_RATE_BUCKETS = 10_000


class GuardError(Exception):
    """A refusal the API turns into an HTTP status."""


class AccessDenied(GuardError):
    """The caller sent no access code, or the wrong one."""


class RateLimited(GuardError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("提问太频繁了，请稍后再试")
        self.retry_after_seconds = retry_after_seconds


class Busy(GuardError):
    """The request queue is full, or an admitted request waited too long."""


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
    feedback_requests_per_window: int = 30
    window_seconds: float = 600
    daily_generations: int = 200
    queue_timeout_seconds: float = 90
    # Includes the one executing request; waiters use coroutines, not threads.
    max_pending_requests: int = 8
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
            feedback_requests_per_window=_positive_number(
                environ,
                "RAG_QA_FEEDBACK_RATE_LIMIT",
                30,
            ),
            window_seconds=_positive_number(environ, "RAG_QA_RATE_WINDOW_SECONDS", 600),
            daily_generations=_positive_number(environ, "RAG_QA_DAILY_GENERATIONS", 200),
            queue_timeout_seconds=_positive_number(environ, "RAG_QA_QUEUE_TIMEOUT", 90),
            max_pending_requests=_positive_number(environ, "RAG_QA_MAX_PENDING_REQUESTS", 8),
            trust_proxy=environ.get("RAG_QA_TRUST_PROXY", "").strip().lower()
            in {"1", "true", "yes"},
        )


def _utc_today() -> date:
    return datetime.now(UTC).date()


class RequestAdmission:
    """One bounded request, queued until its caller can start the answer thread.

    Cancellation releases a waiting request immediately. An executing request
    keeps its slot and capacity until its worker finishes, even if the reader has
    gone away; otherwise the next request could overlap an uncancelled model call.
    """

    def __init__(self, guard: AccessGuard, admitted_at: float) -> None:
        self._guard = guard
        self._admitted_at = admitted_at
        self._state = "queued"
        self._cancelled = threading.Event()

    def try_start(self) -> bool:
        """Acquire the execution slot without blocking; FIFO among admitted callers."""

        return self._guard._try_start(self)

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._guard._lock:
            if self._state == "queued":
                self._guard._release_request(self)

    def release(self) -> None:
        """Return capacity once; only the executing worker may release an active slot."""

        with self._guard._lock:
            self._guard._release_request(self)


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
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._day = today()
        self._generations = 0
        if settings.max_pending_requests <= 0:
            raise ValueError("max_pending_requests 必须是正整数")
        self._requests: deque[RequestAdmission] = deque()
        self._pending_requests = 0
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

    def check_rate(self, client: str, *, scope: str = "question") -> None:
        limits = {
            "question": self.settings.requests_per_window,
            "feedback": self.settings.feedback_requests_per_window,
        }
        if scope not in limits:
            raise ValueError(f"未知限流范围: {scope}")
        now = self._clock()
        horizon = now - self.settings.window_seconds
        with self._lock:
            hits = self._hits.setdefault((scope, client), deque())
            while hits and hits[0] <= horizon:
                hits.popleft()
            if len(hits) >= limits[scope]:
                wait = hits[0] + self.settings.window_seconds - now
                raise RateLimited(max(1, int(wait) + 1))
            hits.append(now)
            if len(self._hits) > _MAX_RATE_BUCKETS:
                idle = [key for key, times in self._hits.items() if times[-1] <= horizon]
                for key in idle:
                    del self._hits[key]
                while len(self._hits) > _MAX_RATE_BUCKETS:
                    del self._hits[next(iter(self._hits))]

    def reserve_generation(self) -> date | None:
        """Claim a generation and return its UTC day; None means retrieval only."""

        with self._lock:
            self._roll_day()
            if self._generations >= self.settings.daily_generations:
                return None
            self._generations += 1
            return self._day

    def refund_generation(self, reserved_on: date) -> None:
        """Return a claim that never reached the model, so the cap counts real calls."""

        with self._lock:
            self._roll_day()
            if reserved_on == self._day:
                self._generations = max(0, self._generations - 1)

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._generations = 0

    def admit_request(self) -> RequestAdmission:
        """Claim bounded capacity before creating a response or a worker thread."""

        with self._lock:
            if self._pending_requests >= self.settings.max_pending_requests:
                raise Busy("当前排队已满，请稍后再试")
            admission = RequestAdmission(self, self._clock())
            self._requests.append(admission)
            self._pending_requests += 1
            return admission

    def _try_start(self, admission: RequestAdmission) -> bool:
        with self._lock:
            if admission._state != "queued":
                return False
            if self._clock() - admission._admitted_at >= self.settings.queue_timeout_seconds:
                self._release_request(admission)
                raise Busy("等待回答超时，请稍后再试")
            if self._requests[0] is not admission or not self._slot.acquire(blocking=False):
                return False
            self._requests.popleft()
            admission._state = "active"
            return True

    def _release_request(self, admission: RequestAdmission) -> None:
        # The caller holds _lock. Repeated release/cancel calls are harmless.
        if admission._state == "released":
            return
        if admission._state == "queued":
            self._requests.remove(admission)
        else:
            self._slot.release()
        admission._state = "released"
        self._pending_requests -= 1

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
            pending = self._pending_requests
            queued = len(self._requests)
        return {
            "access_code_required": self.requires_access_code,
            "generations_remaining_today": remaining,
            "requests_in_flight": pending,
            "requests_queued": queued,
            "request_capacity": self.settings.max_pending_requests,
        }
