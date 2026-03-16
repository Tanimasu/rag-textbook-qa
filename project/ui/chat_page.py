import streamlit as st

from ui.helpers import render_answer_block


def render_chat_tab(book_id, top_k, temperature, max_tokens, load_engine):
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

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer_block(msg["content"], msg.get("sources", []))
            else:
                st.markdown(msg["content"])

    user_question = st.chat_input("请输入您的问题…")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        engine = load_engine()
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

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()
