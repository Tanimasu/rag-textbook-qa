"""Provider token counts shared by runtime answers and evaluation accounting."""

from __future__ import annotations

from typing import Any


def _count(value: Any) -> int | None:
    # Missing or malformed counts do not establish a zero-cost request.
    return value if type(value) is int and value >= 0 else None


def completion_usage(response: Any) -> dict[str, int | None] | None:
    """Retain counts without guessing, coercion, or rejecting a valid answer."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    completion = getattr(usage, "completion_tokens_details", None)
    prompt = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt": _count(getattr(usage, "prompt_tokens", None)),
        "completion": _count(getattr(usage, "completion_tokens", None)),
        "total": _count(getattr(usage, "total_tokens", None)),
        "reasoning": _count(getattr(completion, "reasoning_tokens", None)),
        "cached_prompt": _count(getattr(prompt, "cached_tokens", None)),
    }
