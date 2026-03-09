"""
app.py — Streamlit UI for CS Textbook RAG Q&A System
Run: cd project && streamlit run app.py
"""
import json
import os
import sys

import streamlit as st

# Ensure project directory is on path so local modules resolve
sys.path.insert(0, os.path.dirname(__file__))

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

BOOK_LABELS = ["操作系统", "计算机组成原理", "全部"]
BOOK_IDS    = ["os", "computer_organization", None]


# ─────────────────────────────────────────────────────────────
# Cached resource: RAGEngine (loads embedding model + ChromaDB once)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="正在加载 RAG 引擎，请稍候…")
def load_engine():
    from rag_engine import RAGEngine
    return RAGEngine(db_path="./vector_db", verbose=False)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 CS 教材智能问答系统")
    st.markdown("---")

    st.subheader("教材选择")
    book_label = st.radio("选择教材", BOOK_LABELS, index=0)
    book_id = BOOK_IDS[BOOK_LABELS.index(book_label)]

    st.markdown("---")
    st.subheader("参数设置")
    top_k       = st.slider("检索数量 (top_k)",   min_value=1,   max_value=10,   value=5)
    temperature = st.slider("生成温度 (temperature)", min_value=0.1, max_value=1.0, value=0.7, step=0.05)
    max_tokens  = st.slider("最大生成长度 (max_tokens)", min_value=500, max_value=3000, value=2000, step=100)

    st.markdown("---")
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 问答对话", "📊 评估结果"])


# ═════════════════════════════════════════════════════════════
# TAB 1: Chat
# ═════════════════════════════════════════════════════════════
with tab_chat:
    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show sources for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 参考来源"):
                    for i, src in enumerate(msg["sources"], 1):
                        score = src.get("final_score", src.get("similarity", 0))
                        method = src.get("method", "")
                        chapter = src.get("chapter", "")
                        h2 = src.get("section_h2", "")
                        h3 = src.get("section_h3", "")
                        section = f"{chapter} > {h2}" + (f" > {h3}" if h3 else "")
                        book = src.get("book_name", "")
                        content = src.get("content", "")

                        st.markdown(
                            f"**{i}. [{book}] {section}**  "
                            f"`得分: {score:.3f}` `方法: {method}`"
                        )
                        st.caption(content[:300] + ("…" if len(content) > 300 else ""))
                        if i < len(msg["sources"]):
                            st.divider()

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

            st.markdown(answer)

            if sources:
                with st.expander("📚 参考来源"):
                    for i, src in enumerate(sources, 1):
                        score = src.get("final_score", src.get("similarity", 0))
                        method = src.get("method", "")
                        chapter = src.get("chapter", "")
                        h2 = src.get("section_h2", "")
                        h3 = src.get("section_h3", "")
                        section = f"{chapter} > {h2}" + (f" > {h3}" if h3 else "")
                        book = src.get("book_name", "")
                        content = src.get("content", "")

                        st.markdown(
                            f"**{i}. [{book}] {section}**  "
                            f"`得分: {score:.3f}` `方法: {method}`"
                        )
                        st.caption(content[:300] + ("…" if len(content) > 300 else ""))
                        if i < len(sources):
                            st.divider()

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
