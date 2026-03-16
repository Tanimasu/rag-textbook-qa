import html

import streamlit as st

from config.constants import BOOK_NAME_LABELS


def format_book_label(book_id: str) -> str:
    return BOOK_NAME_LABELS.get(book_id, book_id.replace("_", " ").title())


def format_section_label(source: dict) -> str:
    parts = [
        source.get("chapter", "").strip(),
        source.get("section_h2", "").strip(),
        source.get("section_h3", "").strip(),
    ]
    parts = [part for part in parts if part]
    return " > ".join(parts) if parts else "未标注章节"


def render_source_preview(sources: list[dict]):
    if not sources:
        return

    tags = []
    for source in sources[:3]:
        book = format_book_label(source.get("book_name", "") or "未知教材")
        section = format_section_label(source)
        tags.append(f"<span class='tag'>📘 {book} · {section}</span>")

    st.markdown(
        "<div class='inline-tags'>" + "".join(tags) + "</div>",
        unsafe_allow_html=True,
    )


def render_sources_expander(sources: list[dict]):
    if not sources:
        return

    with st.expander(f"📚 参考来源（{len(sources)}）", expanded=False):
        for i, source in enumerate(sources, 1):
            score = source.get("final_score", source.get("similarity", 0))
            method = source.get("method", "hybrid")
            book = format_book_label(source.get("book_name", "") or "未知教材")
            section = format_section_label(source)
            content = source.get("content", "")
            snippet = html.escape(content[:220] + ("..." if len(content) > 220 else ""))

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{i}. {book}</div>
                    <div class="source-meta">{section} · 分数 {score:.3f} · {method}</div>
                    <div class="source-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_answer_block(answer: str, sources: list[dict]):
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
    render_source_preview(sources)
    render_sources_expander(sources)
