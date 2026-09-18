"""Sidebar and hero layout for the Web interface."""

from __future__ import annotations

from typing import Any

import streamlit as st

from rag_textbook_qa.web.helpers import format_book_label


def render_sidebar(book_options: list[tuple[str, str | None]]) -> dict[str, Any]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand__icon">📚</div>
                <div class="sidebar-brand__text">CS 教材智能问答系统</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.subheader("教材选择")
        book_labels = [label for label, _ in book_options]
        book_mapping = dict(book_options)
        default_index = len(book_labels) - 1 if len(book_labels) == 1 else 0
        book_label = st.radio("选择教材", book_labels, index=default_index)
        book_id = book_mapping[book_label]
        book_count = max(0, len(book_options) - 1)

        st.caption(f"当前可检索教材：{book_count} 本")

        with st.expander("高级参数", expanded=False):
            top_k = st.slider("检索条数 (top_k)", min_value=1, max_value=10, value=5)
            temperature = st.slider(
                "回答发散度 (temperature)",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
            )
            max_tokens = st.slider(
                "最大回答长度 (max_tokens)",
                min_value=500,
                max_value=3000,
                value=2000,
                step=100,
            )
            enable_hyde = st.toggle(
                "启用 HyDE 增强检索",
                value=False,
                help="会在检索前额外调用一次 LLM；默认关闭以降低等待时间和费用。",
            )

        with st.expander("实验功能（默认关闭 · 未通过验收）", expanded=False):
            st.warning(
                "以下功能都**没有通过真实回答质量验收**。开启后结果可能更差；"
                "查询分解和引用核对还会增加模型调用、费用和等待时间。",
                icon="⚠️",
            )
            enable_adjacent_context = st.toggle(
                "补充同小节相邻片段（实验）",
                value=False,
                help=(
                    "保持 Top 5 顺序不变，只用剩余上下文预算追加同小节的相邻片段。"
                    "35 题开发集的正文覆盖有所提升，但尚未证明回答质量提高；"
                    "查询分解实际生效时不会叠加此功能。"
                ),
            )
            enable_decomposition = st.toggle(
                "启用查询分解（实验 · 未验收）",
                value=False,
                help=(
                    "适合包含多个独立问题的提问；额外调用一次模型规划。成功分解时不使用 HyDE，"
                    "失败时恢复普通检索。仅完成小样本回归，未证明整体答案准确率提升。"
                ),
            )

            verify_citations = st.toggle(
                "核对答案引用（实验 · 未验收）",
                value=False,
                help=(
                    "增加两次模型核对；完成后才显示答案，核对失败时不会显示未经核对的草稿。"
                    "模型核对本身也会出错。目前只通过离线流程测试，没有真实效果验收。"
                ),
            )

        st.markdown("---")
        if st.button("清空对话", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    return {
        "book_id": book_id,
        "book_count": book_count,
        "top_k": top_k,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_hyde": enable_hyde,
        "enable_adjacent_context": enable_adjacent_context,
        "enable_decomposition": enable_decomposition,
        "verify_citations": verify_citations,
    }


def render_hero(book_id: str | None, book_count: int, top_k: int) -> None:
    selected_book_text = "全教材检索" if book_id is None else format_book_label(book_id)
    st.markdown(
        f"""
        <div class="hero">
            <h1>教材问答工作台</h1>
            <p>围绕教材原文进行检索、回答与溯源，适合课程演示和论文答辩时展示 RAG 的可解释性。</p>
        </div>
        <div class="status-grid">
            <div class="status-card">
                <div class="status-label">当前检索范围</div>
                <div class="status-value">{selected_book_text}</div>
            </div>
            <div class="status-card">
                <div class="status-label">已加载教材</div>
                <div class="status-value">{book_count} 本</div>
            </div>
            <div class="status-card">
                <div class="status-label">当前检索条数</div>
                <div class="status-value">Top {top_k}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
