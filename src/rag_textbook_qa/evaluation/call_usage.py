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

UsageObserver = Callable[[Mapping[str, Any]], None]


class CallUsageRecordingError(RuntimeError):
    """A paid call finished but its accounting record could not be saved."""


class CallUsageLog:
    """Append usage without prompts, answers, reasoning text or provider errors.

    The experiment's run lock owns the directory; this lock serializes its
    worker threads. A damaged tail is separated and preserved before appending.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def __call__(self, record: Mapping[str, Any]) -> None:
        line = (json.dumps(dict(record), ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock, self.path.open("ab+") as handle:
            handle.seek(0, 2)
            if handle.tell():
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write(line)


def _count(value: Any) -> int | None:
    # Missing usage is unknown, not a zero-cost call.
    return value if type(value) is int and value >= 0 else None


def observed_completion(
    client: Any, request: Mapping[str, Any], *, role: str,
    on_call: UsageObserver | None,
) -> Any:
    """Observe one attempt; retry owners invoke this again for each HTTP call."""

    if on_call is None:
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
    try:
        response = client.chat.completions.create(**request)
        usage = getattr(response, "usage", None)
        if usage is not None:
            completion = getattr(usage, "completion_tokens_details", None)
            prompt = getattr(usage, "prompt_tokens_details", None)
            record["tokens"] = {
                "prompt": _count(getattr(usage, "prompt_tokens", None)),
                "completion": _count(getattr(usage, "completion_tokens", None)),
                "total": _count(getattr(usage, "total_tokens", None)),
                "reasoning": _count(getattr(completion, "reasoning_tokens", None)),
                "cached_prompt": _count(getattr(prompt, "cached_tokens", None)),
            }
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
