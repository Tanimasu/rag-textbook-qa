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


def compute_trace_items(execution: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Turn a safe engine execution summary into compact UI labels."""

    if not execution:
        return []

    items = []
    for key, label, icon in (
        ("embedding", "Embedding", "🧭"),
        ("reranker", "Reranker", "🎯"),
    ):
        stage = execution.get(key)
        if not isinstance(stage, Mapping):
            continue
        backend = str(stage.get("backend") or "unknown")
        fallback_used = bool(stage.get("fallback_used"))
        location = _execution_location(
            backend=backend,
            platform_name=str(stage.get("platform") or "unknown"),
            fallback_used=fallback_used,
        )
        device = str(stage.get("device") or "unknown")
        device_label = device.upper() if device != "unknown" else "未知设备"
        elapsed = _seconds(stage.get("elapsed_seconds"))
        calls = _positive_int(stage.get("calls"))
        call_suffix = f" · {calls} 次" if calls > 1 else ""
        items.append(
            {
                "kind": "fallback" if fallback_used else backend,
                "text": (
                    f"{icon} {label} · {location} · {device_label} · "
                    f"{elapsed:.3f} 秒{call_suffix}"
                ),
            }
        )

    retrieval = _seconds(execution.get("retrieval_seconds"))
    generation = _seconds(execution.get("generation_seconds"))
    total = _seconds(execution.get("total_seconds"))
    items.append(
        {
            "kind": "timing",
            "text": (
                f"⏱️ 检索 {retrieval:.3f} 秒 · 回答 {generation:.3f} 秒 · "
                f"总计 {total:.3f} 秒"
            ),
        }
    )
    return items


def _execution_location(*, backend: str, platform_name: str, fallback_used: bool) -> str:
    platform_label = {
        "Darwin": "macOS",
        "Windows": "Windows",
        "Linux": "Linux",
        "mixed": "多平台",
    }.get(platform_name, "")
    if backend == "remote":
        location = "远程 Worker"
    elif backend == "local":
        location = "本地"
    elif backend == "mixed":
        location = "远程/本地混合"
    else:
        location = "执行位置未知"
    if platform_label:
        location += f"（{platform_label}）"
    if fallback_used:
        location = f"已回退到{location}"
    return location


def _seconds(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
