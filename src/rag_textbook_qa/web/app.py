"""Packaged Streamlit entry point for the textbook QA application."""

from __future__ import annotations

import streamlit as st

from rag_textbook_qa.web.chat_page import render_chat_tab
from rag_textbook_qa.web.eval_page import render_eval_tab
from rag_textbook_qa.web.layout import render_hero, render_sidebar
from rag_textbook_qa.web.services import (
    load_available_books,
    load_engine,
    load_ragas_results,
    run_ragas_evaluation,
)
from rag_textbook_qa.web.styles import inject_custom_styles


def main() -> None:
    # Present the packaged app as an end-user UI. This hides Streamlit's
    # developer menu and prevents its bare "C" clear-cache shortcut from
    # colliding with Command+C on macOS.
    st.set_option("client.toolbarMode", "viewer")
    st.set_page_config(
        page_title="CS 教材智能问答系统",
        page_icon="📚",
        layout="wide",
    )

    inject_custom_styles()
    st.session_state.setdefault("messages", [])
    book_options = load_available_books()
    sidebar_state = render_sidebar(book_options)
    render_hero(
        book_id=sidebar_state["book_id"],
        book_count=sidebar_state["book_count"],
        top_k=sidebar_state["top_k"],
    )

    tab_chat, tab_eval = st.tabs(["💬 问答对话", "📊 评估结果"])
    with tab_chat:
        render_chat_tab(
            book_id=sidebar_state["book_id"],
            top_k=sidebar_state["top_k"],
            temperature=sidebar_state["temperature"],
            max_tokens=sidebar_state["max_tokens"],
            load_engine=load_engine,
        )
    with tab_eval:
        render_eval_tab(
            load_ragas_results=load_ragas_results,
            run_ragas_evaluation=run_ragas_evaluation,
        )


if __name__ == "__main__":
    main()
