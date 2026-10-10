"""Safe per-HTTP-attempt usage records, including failed and retried calls."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_textbook_qa.token_usage import completion_usage

UsageObserver = Callable[[Mapping[str, Any]], None]


class CallUsageRecordingError(RuntimeError):
    """A paid call finished but its accounting record could not be saved."""


class CallAttemptLimitReached(RuntimeError):
    """The invocation's shared allowance has no remaining HTTP attempts."""


class CallAttemptBudget:
    """Reserve attempts before sending, shared by generators, judges and retries.

    Reservations are conservative: SDK validation failures also consume a slot.
    This is an invocation limit, not a currency or cumulative experiment limit.
    """

    def __init__(self, limit: int) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("请求次数上限必须为正整数")
        self._limit = limit
        self._started = 0
        self._blocked = False
        self._lock = threading.Lock()

    def reserve(self) -> None:
        with self._lock:
            if self._started >= self._limit:
                self._blocked = True
                raise CallAttemptLimitReached("已达到本次运行的请求次数上限")
            self._started += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"scope": "invocation", "limit": self._limit,
                    "attempts_started": self._started, "blocked": self._blocked}


class CallUsageLog:
    """Append usage without prompts, answers, reasoning text or provider errors.

    The experiment's run lock owns the directory; this lock serializes its
    worker threads. A damaged tail is separated and preserved before appending.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def __call__(self, record: Mapping[str, Any]) -> None:
        line = (json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with self._lock, self.path.open("ab+") as handle:
            handle.seek(0, 2)
            if handle.tell():
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write(line)


def observed_completion(
    client: Any, request: Mapping[str, Any], *, role: str,
    on_call: UsageObserver | None,
    call_budget: CallAttemptBudget | None = None,
) -> Any:
    """Observe one attempt; retry owners invoke this again for each HTTP call."""

    if on_call is None:
        if call_budget is not None:
            call_budget.reserve()
        return client.chat.completions.create(**request)
    started = time.monotonic()
    record: dict[str, Any] = {
        "call_id": uuid.uuid4().hex,
        "at_utc": datetime.now(UTC).isoformat(),
        "role": role,
        "model": request["model"],
        "request_sha256": hashlib.sha256(json.dumps(
            request, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "status": "error",
        "tokens": None,
    }
    # A refused attempt never reaches the SDK or receives a usage record.
    if call_budget is not None:
        call_budget.reserve()
    try:
        response = client.chat.completions.create(**request)
        record["tokens"] = completion_usage(response)
        record["status"] = "response"
        choices = getattr(response, "choices", ())
        record["finish_reason"] = getattr(choices[0], "finish_reason", None) if choices else None
        return response
    except Exception as error:
        record["error_type"] = type(error).__name__
        status = getattr(error, "status_code", None)
        if type(status) is int:
            record["http_status"] = status
        raise
    finally:
        record["seconds"] = round(time.monotonic() - started, 3)
        # An accounting write failure must stop further work, not trigger a
        # provider retry that could charge again for a response already received.
        try:
            on_call(record)
        except Exception as error:
            raise CallUsageRecordingError("模型调用用量无法保存，已停止评测") from error
