"""Formatting and source rendering helpers for the Web interface."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from rag_textbook_qa.catalog import BOOK_LABELS
from rag_textbook_qa.web.messages import compute_trace_items


def format_book_label(book_id: str) -> str:
    return BOOK_LABELS.get(book_id, book_id.replace("_", " ").title())


def format_section_label(source: dict[str, Any]) -> str:
    parts = [
        str(source.get("chapter", "")).strip(),
        str(source.get("section_h2", "")).strip(),
        str(source.get("section_h3", "")).strip(),
    ]
    populated = [part for part in parts if part]
    return " > ".join(populated) if populated else "未标注章节"


def render_source_preview(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    tags = []
    for source in sources[:3]:
        book = html.escape(format_book_label(source.get("book_name", "") or "未知教材"))
        section = html.escape(format_section_label(source))
        tags.append(f"<span class='tag'>📘 {book} · {section}</span>")

    st.markdown(
        "<div class='inline-tags'>" + "".join(tags) + "</div>",
        unsafe_allow_html=True,
    )


def render_sources_expander(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander(f"📚 参考来源（{len(sources)}）", expanded=False):
        for index, source in enumerate(sources, 1):
            score = float(source.get("final_score", source.get("similarity", 0)))
            method = html.escape(str(source.get("method", "hybrid")))
            book = html.escape(format_book_label(source.get("book_name", "") or "未知教材"))
            section = html.escape(format_section_label(source))
            content = str(source.get("content", ""))
            snippet = html.escape(content[:220] + ("..." if len(content) > 220 else ""))

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{index}. {book}</div>
                    <div class="source-meta">{section} · 分数 {score:.3f} · {method}</div>
                    <div class="source-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_compute_trace(execution: dict[str, Any] | None) -> None:
    items = compute_trace_items(execution)
    if not items:
        return
    chips = "".join(
        (
            f"<span class='compute-chip compute-chip--{html.escape(item['kind'])}'>"
            f"{html.escape(item['text'])}</span>"
        )
        for item in items
    )
    st.markdown(
        f"<div class='compute-trace'>{chips}</div>",
        unsafe_allow_html=True,
    )


def render_answer_block(
    answer: str,
    sources: list[dict[str, Any]],
    execution: dict[str, Any] | None = None,
) -> None:
    st.markdown(
        """
        <div class="answer-shell">
            <div class="answer-title">
                <span class="answer-badge">答</span>
                <span>教材回答</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(answer)
    render_compute_trace(execution)
    render_source_preview(sources)
    render_sources_expander(sources)
