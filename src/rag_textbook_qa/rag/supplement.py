"""Validate exact, bounded source excerpts selected for a context experiment.

This module creates no model client and is not enabled by the product engine.
The selector names existing paragraphs; it cannot supply replacement prose.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from rag_textbook_qa.rag.context import select_context

_PARAGRAPH = re.compile(r"\S.*?(?=\n[ \t]*\n|\Z)", re.DOTALL)
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$", re.MULTILINE)
_IMAGE_START = re.compile(r"(?:!\[[^\n]*?\]\(|<img\b)", re.IGNORECASE)
_PIPE_TABLE_SEPARATOR = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$",
    re.MULTILINE,
)
_RAW_C_FUNCTION = re.compile(
    r"^[ \t]*(?:[A-Za-z_]\w*[ \t*]+)+[A-Za-z_]\w*[ \t]*"
    r"\([^{}\n]*\)[ \t]*\n[ \t]*\{", re.MULTILINE,
)
_DISPLAY_MATH = re.compile(r"(?:\$\$.*\$\$|\\\[.*\\\])", re.DOTALL)
_BROKEN_IMAGE_START = re.compile(
    r"[a-f0-9]{32,64}\.(?:png|jpe?g|gif|webp|bmp)\s*\)", re.IGNORECASE,
)


class ExcerptBudgetExceeded(ValueError):
    """Valid excerpts exceed a source, body-length, or context allowance."""


def paragraph_spans(content: str) -> list[dict[str, Any]]:
    """Expose exact offsets, excluding recognized code, tables and non-prose blocks."""
    protected = []
    opening = None
    for match in _FENCE.finditer(content):
        marker, rest = match.groups()
        if opening is None:
            if marker[0] == "~" or "`" not in rest:
                opening = (marker, match.start())
        elif marker[0] == opening[0][0] and len(marker) >= len(opening[0]) and not rest.strip():
            protected.append((opening[1], match.end()))
            opening = None
    if opening is not None:
        protected.append((opening[1], len(content)))
    protected.extend((m.start(), m.end()) for m in re.finditer(
        r"<table\b.*?(?:</table\s*>|\Z)", content, re.DOTALL | re.IGNORECASE,
    ))
    paragraphs = []
    for match in _PARAGRAPH.finditer(content):
        text = match.group().rstrip()
        start, end = match.start(), match.start() + len(text)
        paragraphs.append({
            "index": len(paragraphs), "start": start, "end": end, "text": text,
            "selectable": len(text) >= 20 and not _IMAGE_START.match(text)
            and not _PIPE_TABLE_SEPARATOR.search(text) and not _RAW_C_FUNCTION.search(text)
            and not _DISPLAY_MATH.fullmatch(text) and not _BROKEN_IMAGE_START.match(text)
            and not any(
                start < right and end > left for left, right in protected
            ),
        })
    return paragraphs


def pack_selected_excerpts(
    ranked: Sequence[dict[str, Any]], candidates: Sequence[dict[str, Any]],
    selection: Mapping[str, Any], *, context_budget: int = 4000,
    max_sources: int = 2, max_excerpt_chars: int = 1200,
) -> tuple[str, list[dict[str, Any]]]:
    """Append selected contiguous prose only if all excerpts fit without clipping."""
    for name, value in (("context_budget", context_budget), ("max_sources", max_sources),
                        ("max_excerpt_chars", max_excerpt_chars)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} 必须为正整数")
    if not isinstance(selection, Mapping) or set(selection) != {"selected"}:
        raise ValueError("选择结果必须只包含 selected")
    chosen = selection["selected"]
    if not isinstance(chosen, list):
        raise TypeError("选择来源格式错误")
    if len(chosen) > max_sources:
        raise ExcerptBudgetExceeded("选择来源数超过限制")
    by_id = {}
    for candidate in candidates:
        identifier = candidate.get("chunk_id")
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError("候选来源编号缺失或重复")
        by_id[identifier] = candidate
    ranked_ids = {(r["book_name"], r["chunk_id"]) for r in ranked}
    books = {r["book_name"] for r in ranked}
    seen = set()
    excerpts = []
    for item in chosen:
        if not isinstance(item, Mapping) or set(item) != {
            "chunk_id", "start_paragraph", "end_paragraph",
        }:
            raise ValueError("选择项必须声明来源及连续段落范围")
        identifier = item["chunk_id"]
        if not isinstance(identifier, str) or identifier not in by_id or identifier in seen:
            raise ValueError("选择了未知或重复来源")
        source = by_id[identifier]
        identity = (source.get("book_name"), identifier)
        if identity in ranked_ids or identity[0] not in books:
            raise ValueError("选择来源已在基线中或来自其他教材")
        paragraphs = paragraph_spans(source["content"])
        first, last = item["start_paragraph"], item["end_paragraph"]
        if (type(first) is not int or type(last) is not int
                or not 0 <= first <= last < len(paragraphs)):
            raise ValueError("选择段落范围无效")
        selected = paragraphs[first:last + 1]
        if not all(p["selectable"] for p in selected):
            raise ValueError("选择范围含代码、表格、图片或非正文段落")
        start, end = selected[0]["start"], selected[-1]["end"]
        excerpt = source["content"][start:end]
        excerpts.append({
            **source, "content": excerpt, "char_count": len(excerpt),
            "selected_excerpt": True, "excerpt_start": start, "excerpt_end": end,
            "original_char_count": len(source["content"]),
            "selection_method": "validated-paragraph-selection-experiment",
        })
        seen.add(identifier)
    if sum(len(r["content"]) for r in excerpts) > max_excerpt_chars:
        raise ExcerptBudgetExceeded("选取正文超过长度限制")
    baseline_context, baseline_sources = select_context(list(ranked), context_budget)
    context, sources = select_context([*ranked, *excerpts], context_budget)
    if sources[:len(baseline_sources)] != baseline_sources or not context.startswith(baseline_context):
        raise ValueError("原有上下文被改变")
    added = sources[len(baseline_sources):]
    if len(added) != len(excerpts) or any(s["truncated"] for s in added):
        raise ExcerptBudgetExceeded("所选原文无法完整装入剩余预算")
    return context, sources
