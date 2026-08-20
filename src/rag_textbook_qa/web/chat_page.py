"""Interactive textbook chat tab."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from rag_textbook_qa.web.helpers import render_answer_block
from rag_textbook_qa.web.messages import answer_message


def render_chat_tab(
    book_id: str | None,
    top_k: int,
    temperature: float,
    max_tokens: int,
    load_engine: Callable[[], Any],
) -> None:
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer_block(
                    message["content"],
                    message.get("sources", []),
                    message.get("execution"),
                )
            else:
                st.markdown(message["content"])

    user_question = st.chat_input("请输入您的问题…")
    if not user_question:
        return

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

    answer = answer_message(result)
    sources = result.get("results", [])
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "execution": result.get("execution"),
        }
    )
    st.rerun()
