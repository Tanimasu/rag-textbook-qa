"""Pure presentation helpers for Web response messages."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from rag_textbook_qa.catalog import BOOK_LABELS
from rag_textbook_qa.rag.references import render_source_sections
from rag_textbook_qa.timing import finite_seconds


def source_section_label(source: Mapping[str, Any]) -> str:
    parts = [str(source.get(field) or "").strip()
             for field in ("chapter", "section_h2", "section_h3", "section_h4")]
    return " > ".join(part for part in parts if part) or "未标注章节"


def _literal_block(text: str) -> str:
    # Textbook code can contain fences itself. Preserve it as literal evidence,
    # without letting a shorter fence reinterpret the following source as Markdown.
    longest = max((len(match[0]) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def answer_export_markdown(
    query: str, answer: str, sources: Sequence[Mapping[str, Any]],
) -> str:
    """Save the displayed answer and its actual excerpts, without internal metadata."""
    answer = render_source_sections(answer, sources)
    lines = ["# 教材问答记录", "", "## 问题", "", _literal_block(query), "",
             "## 回答", "", answer, "", "## 参考教材片段", "",
             "以下为本次回答使用的片段；请按资料编号核对回答。", ""]
    for index, source in enumerate(sources, 1):
        book_id = str(source.get("book_name") or "未知教材")
        book = BOOK_LABELS.get(book_id, book_id.replace("_", " ").title())
        citation = source.get("citation_id", index)
        lines.extend([f"### 参考资料 {citation} · {book}", "",
                      source_section_label(source), ""])
        if source.get("truncated"):
            lines.extend(["片段已截断。", ""])
        if source.get("table_compacted"):
            lines.extend(["表格按行整理。", ""])
        lines.extend([_literal_block(str(source.get("content") or "")), ""])
    if not sources:
        lines.extend(["本次记录未包含教材片段。", ""])
    return "\n".join(lines)


def answer_message(result: Mapping[str, Any]) -> str:
    error = str(result.get("error") or "").strip()
    if error and result.get("success") is False:
        if len(error) > 500:
            error = error[:497] + "..."
        return f"⚠️ 未能生成答案：{error}"

    answer = result.get("answer")
    if answer:
        return render_source_sections(str(answer), result.get("context_sources") or [])
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
        device_label = {"unknown": "未知设备", "mixed": "多设备"}.get(device, device.upper())
        elapsed = _duration_label(stage.get("elapsed_seconds"))
        calls = _positive_int(stage.get("calls"))
        call_suffix = f" · {calls} 次" if calls > 1 else ""
        items.append(
            {
                "kind": "fallback" if fallback_used else backend,
                "text": (
                    f"{icon} {label} · {location} · {device_label} · "
                    f"{elapsed}{call_suffix}"
                ),
            }
        )

    retrieval = _duration_label(execution.get("retrieval_seconds"))
    generation = _duration_label(execution.get("generation_seconds"))
    total = _duration_label(execution.get("total_seconds"))
    first_token = execution.get("first_token_seconds")
    first_token_suffix = (
        f"（首字 {_duration_label(first_token)}）"
        if first_token is not None
        else ""
    )
    items.append(
        {
            "kind": "timing",
            "text": (
                f"⏱️ 检索 {retrieval} · "
                f"回答 {generation}{first_token_suffix} · "
                f"总计 {total}"
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
        location = f"{location} · 已发生回退" if backend == "mixed" else f"已回退到{location}"
    return location


def _duration_label(value: Any) -> str:
    seconds = finite_seconds(value)
    return f"{seconds:.3f} 秒" if seconds is not None else "耗时未知"


def _positive_int(value: Any) -> int:
    count = finite_seconds(value)
    return int(count) if count is not None and count.is_integer() else 0
