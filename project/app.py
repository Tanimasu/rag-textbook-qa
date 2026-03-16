"""
app.py — Streamlit UI for CS Textbook RAG Q&A System
Run: cd project && streamlit run app.py
"""
import sys

import streamlit as st

from config.constants import BASE_DIR
from services.app_services import (
    load_available_books,
    load_engine,
    load_ragas_results,
    run_ragas_evaluation,
)
from ui.chat_page import render_chat_tab
from ui.eval_page import render_eval_tab
from ui.layout import render_hero, render_sidebar
from ui.styles import inject_custom_styles


st.set_page_config(
    page_title="CS 教材智能问答系统",
    page_icon="📚",
    layout="wide",
)

sys.path.insert(0, BASE_DIR)
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
