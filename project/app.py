"""
app.py — Streamlit UI for CS Textbook RAG Q&A System
Run: cd project && streamlit run app.py
"""
import html
import json
import os
import sqlite3

import streamlit as st

# ─────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CS 教材智能问答系统",
    page_icon="📚",
    layout="wide",
)

RAGAS_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "ragas_evaluation_results.csv")
TEST_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "test_questions.json")
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "vector_db")

BOOK_NAME_LABELS = {
    "os": "操作系统",
    "computer_organization": "计算机组成原理",
    "computer_network": "计算机网络",
    "database": "数据库原理及应用",
    "data_structure": "数据结构",
}


def format_book_label(book_id: str) -> str:
    """将集合后缀转换成适合前端显示的教材名称。"""
    return BOOK_NAME_LABELS.get(book_id, book_id.replace("_", " ").title())


def inject_custom_styles():
    st.markdown(
        """
        <style>
        :root {
            --accent: #e85d5d;
            --accent-soft: #fff1ef;
            --ink: #22304a;
            --muted: #6d7890;
            --line: #e7ebf3;
            --panel: #ffffff;
            --panel-alt: #f7f9fc;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(232, 93, 93, 0.08), transparent 28%),
                linear-gradient(180deg, #f7f8fb 0%, #f3f5f9 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f6f7fb 0%, #eef2f7 100%);
            border-right: 1px solid rgba(34, 48, 74, 0.08);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
            padding-bottom: 1.25rem;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 1.6rem;
            padding-bottom: 5rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
        }

        .stTabs [aria-selected="true"] {
            background: var(--accent-soft);
            border-color: rgba(232, 93, 93, 0.2);
            color: #a54a4a;
        }

        [data-testid="stChatInput"] {
            background: rgba(255, 255, 255, 0.92);
            border-top: 1px solid rgba(34, 48, 74, 0.08);
        }

        .hero {
            background: linear-gradient(135deg, #ffffff 0%, #fff7f4 100%);
            border: 1px solid rgba(232, 93, 93, 0.14);
            border-radius: 24px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 14px 40px rgba(34, 48, 74, 0.06);
            margin-bottom: 1rem;
        }

        .hero h1 {
            color: var(--ink);
            font-size: 2rem;
            line-height: 1.2;
            margin: 0;
        }

        .hero p {
            margin: 0.55rem 0 0;
            color: var(--muted);
            font-size: 0.98rem;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.9rem 0 1.2rem;
        }

        .status-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }

        .status-label {
            color: var(--muted);
            font-size: 0.8rem;
            margin-bottom: 0.35rem;
        }

        .status-value {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 700;
        }

        .answer-shell {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: 0 18px 44px rgba(34, 48, 74, 0.06);
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
        }

        .answer-title {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: var(--ink);
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .answer-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #ffb347 0%, #ff8a3d 100%);
            color: #fff;
            font-size: 1rem;
        }

        .inline-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.75rem 0 0.5rem;
        }

        .tag {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.38rem 0.75rem;
            border-radius: 999px;
            background: var(--accent-soft);
            border: 1px solid rgba(232, 93, 93, 0.14);
            color: #a54a4a;
            font-size: 0.83rem;
            line-height: 1;
        }

        .source-card {
            background: var(--panel-alt);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
        }

        .source-title {
            color: var(--ink);
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .source-meta {
            color: var(--muted);
            font-size: 0.84rem;
            margin-bottom: 0.45rem;
        }

        .source-snippet {
            color: #33415c;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .empty-state {
            background: rgba(255, 255, 255, 0.7);
            border: 1px dashed rgba(109, 120, 144, 0.35);
            border-radius: 22px;
            padding: 1.1rem 1.2rem;
            color: var(--muted);
            margin-top: 0.75rem;
        }

        div[data-testid="stChatMessage"] {
            background: transparent;
        }

        div[data-testid="stChatMessageContent"] {
            width: 100%;
        }

        @media (max-width: 900px) {
            .status-grid {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 1.6rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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

    preview_items = sources[:3]
    tags = []
    for source in preview_items:
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


@st.cache_data(show_spinner=False)
def load_available_books():
    """从 ChromaDB 中读取当前已向量化的教材列表。"""
    sqlite_path = os.path.join(VECTOR_DB_PATH, "chroma.sqlite3")

    if not os.path.exists(sqlite_path):
        return [("全部", None)]

    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM collections
            WHERE name LIKE 'textbook_%'
            ORDER BY name
            """
        ).fetchall()

    book_ids = [row[0].replace("textbook_", "") for row in rows]
    options = [(format_book_label(book_id), book_id) for book_id in book_ids]
    options.append(("全部", None))
    return options


# ─────────────────────────────────────────────────────────────
# Cached resource: RAGEngine (loads embedding model + ChromaDB once)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="正在加载 RAG 引擎，请稍候…")
def load_engine():
    # Keep the import local so Streamlit does not load the full RAG stack until needed.
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from rag_engine import RAGEngine
    return RAGEngine(db_path="./vector_db", verbose=False)


inject_custom_styles()


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 CS 教材智能问答系统")
    st.markdown("---")

    st.subheader("教材选择")
    book_options = load_available_books()
    book_labels = [label for label, _ in book_options]
    book_mapping = dict(book_options)
    default_index = len(book_labels) - 1 if len(book_labels) == 1 else 0
    book_label = st.radio("选择教材", book_labels, index=default_index)
    book_id = book_mapping[book_label]

    st.caption(f"当前可检索教材：{max(0, len(book_options) - 1)} 本")

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


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

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
            <div class="status-value">{max(0, len(book_options) - 1)} 本</div>
        </div>
        <div class="status-card">
            <div class="status-label">当前检索条数</div>
            <div class="status-value">Top {top_k}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 问答对话", "📊 评估结果"])


# ═════════════════════════════════════════════════════════════
# TAB 1: Chat
# ═════════════════════════════════════════════════════════════
with tab_chat:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                可以直接提问概念题、比较题或定义题，例如“什么是进程？”、“线程与进程的区别是什么？”。
                回答会优先依据教材原文生成，并给出参考章节来源。
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer_block(msg["content"], msg.get("sources", []))
            else:
                st.markdown(msg["content"])

    # Chat input
    user_question = st.chat_input("请输入您的问题…")
    if user_question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Get answer from RAG engine
        engine = load_engine()
        with st.chat_message("assistant"):
            with st.spinner("正在检索和生成答案…"):
                result = engine.ask(
                    query=user_question,
                    book_name=book_id,
                    top_k=top_k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            answer = result.get("answer") or "抱歉，未能生成答案。"
            sources = result.get("results", [])

            render_answer_block(answer, sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


# ═════════════════════════════════════════════════════════════
# TAB 2: RAGAS Evaluation
# ═════════════════════════════════════════════════════════════

# RAGAS metric display names
RAGAS_METRIC_LABELS = {
    "faithfulness":       "忠实度",
    "answer_relevancy":   "答案相关性",
    "context_precision":  "上下文精确度",
    "context_recall":     "上下文召回率",
}

with tab_eval:
    st.header("RAGAS 评估结果")

    import pandas as pd

    def load_ragas_results():
        if os.path.exists(RAGAS_RESULTS_PATH):
            return pd.read_csv(RAGAS_RESULTS_PATH, encoding="utf-8-sig")
        return None

    def run_ragas_evaluation():
        from ragas_evaluation import RAGASEvaluator
        engine = load_engine()
        test_questions = json.load(open(TEST_QUESTIONS_PATH, encoding="utf-8"))
        evaluator = RAGASEvaluator()
        dataset = evaluator.prepare_evaluation_data(engine, test_questions)
        result = evaluator.evaluate(dataset)
        df = evaluator.print_results(result)
        if df is not None:
            df.to_csv(RAGAS_RESULTS_PATH, index=False, encoding="utf-8-sig")
        return load_ragas_results()

    df_existing = load_ragas_results()

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_eval = st.button("运行评估", use_container_width=True)
    with col_info:
        if df_existing is not None:
            st.caption(f"已有评估结果（{len(df_existing)} 条），点击「运行评估」重新生成。")
        else:
            st.caption("尚无评估结果，点击「运行评估」开始（需要几分钟）。")

    if run_eval:
        with st.spinner("正在运行 RAGAS 评估，请耐心等待…"):
            df_existing = run_ragas_evaluation()
        if df_existing is not None:
            st.success("评估完成！")
        else:
            st.error("评估完成但未生成结果文件，请检查控制台输出。")

    if df_existing is not None:
        # Identify metric columns present in CSV
        metric_cols = [c for c in RAGAS_METRIC_LABELS if c in df_existing.columns]

        # Metric cards — averages
        if metric_cols:
            st.subheader("汇总指标")
            cols = st.columns(len(metric_cols))
            for col, metric in zip(cols, metric_cols):
                avg = df_existing[metric].mean()
                col.metric(RAGAS_METRIC_LABELS[metric], f"{avg:.3f}")

        # Per-question bar chart (use first metric or all stacked)
        if metric_cols:
            st.subheader("逐题得分")
            # Build chart dataframe: index = short question label
            question_col = next(
                (c for c in df_existing.columns if c in {"user_input", "question"}), None
            )
            if question_col:
                chart_df = df_existing[metric_cols].copy()
                chart_df.index = df_existing[question_col].str[:20] + "…"
            else:
                chart_df = df_existing[metric_cols].copy()
                chart_df.index = [f"Q{i+1}" for i in range(len(chart_df))]
            st.bar_chart(chart_df)

        # Full results table
        st.subheader("详细结果")
        # Rename metric columns to Chinese for display
        display_df = df_existing.copy()
        display_df = display_df.rename(columns=RAGAS_METRIC_LABELS)
        # Round numeric columns
        for col in display_df.select_dtypes(include="number").columns:
            display_df[col] = display_df[col].round(3)
        st.dataframe(display_df, use_container_width=True)
