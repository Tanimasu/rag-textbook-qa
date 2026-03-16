import streamlit as st

from ui.helpers import format_book_label


def render_sidebar(book_options):
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

        st.markdown("---")
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return {
        "book_id": book_id,
        "book_count": book_count,
        "top_k": top_k,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def render_hero(book_id, book_count, top_k):
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
