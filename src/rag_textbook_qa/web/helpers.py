"""Formatting and source rendering helpers for the Web interface."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from rag_textbook_qa.catalog import BOOK_LABELS
from rag_textbook_qa.rag.references import render_source_sections
from rag_textbook_qa.web.messages import compute_trace_items, source_section_label


def format_book_label(book_id: str) -> str:
    return BOOK_LABELS.get(book_id, book_id.replace("_", " ").title())


def format_section_label(source: dict[str, Any]) -> str:
    return source_section_label(source)


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
        st.caption("逐条核对回答中的参考资料编号与教材原文。")
        for index, source in enumerate(sources, 1):
            book = html.escape(format_book_label(source.get("book_name", "") or "未知教材"))
            section = html.escape(format_section_label(source))
            if source.get("truncated"):
                section += " · 片段已截断"
            if source.get("table_compacted"):
                section += " · 表格按行整理"
            content = str(source.get("content", ""))
            citation_id = html.escape(str(source.get("citation_id", index)))
            snippet = html.escape(content)

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">参考资料 {citation_id} · {book}</div>
                    <div class="source-meta">{section}</div>
                    <div class="source-snippet" tabindex="0" role="region"
                         aria-label="参考资料 {citation_id} 原文">{snippet}</div>
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
    render_answer_header()
    st.markdown(render_source_sections(answer, sources))
    render_answer_details(sources, execution)


def render_answer_header() -> None:
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


def render_answer_details(
    sources: list[dict[str, Any]],
    execution: dict[str, Any] | None = None,
) -> None:
    render_compute_trace(execution)
    render_source_preview(sources)
    render_sources_expander(sources)
