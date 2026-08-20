"""Pure presentation helpers for Web response messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def answer_message(result: Mapping[str, Any]) -> str:
    error = str(result.get("error") or "").strip()
    if error and result.get("success") is False:
        if len(error) > 500:
            error = error[:497] + "..."
        return f"⚠️ 未能生成答案：{error}"

    answer = result.get("answer")
    if answer:
        return str(answer)
    if error:
        return f"⚠️ 未能生成答案：{error}"
    return "抱歉，未能生成答案。"
