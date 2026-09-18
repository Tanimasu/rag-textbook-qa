"""Interactive textbook chat tab."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from rag_textbook_qa.providers.base import ProviderError
from rag_textbook_qa.web.helpers import (
    render_answer_block,
    render_answer_details,
    render_answer_header,
)
from rag_textbook_qa.web.messages import answer_message


def render_decomposition(plan: dict[str, Any] | None) -> None:
    if not plan or plan.get("status") == "disabled":
        return
    if plan.get("status") == "active":
        with st.expander("本次查询分解"):
            for index, query in enumerate(plan.get("queries", []), 1):
                st.write(f"{index}. {query}")
            if plan.get("uncovered_query_ids"):
                st.caption("部分子问题没有检索片段进入上下文，回答会提示证据不足。")
    elif plan.get("status") == "not_needed":
        if plan.get("reason") == "local_dependent":
            st.caption("本题包含前后依赖，已使用原问题检索。")
        elif plan.get("reason") == "local_simple":
            st.caption("本题是单一需求，已直接检索，未调用分解模型。")
        else:
            st.caption("本题无需独立拆分，已使用原问题检索。")
    else:
        st.caption("本次查询分解未启用成功，已回退到原问题检索。")


def render_grounding(grounding: dict[str, Any] | None) -> None:
    if not grounding:
        return
    if grounding.get("status") == "checked":
        st.caption("已完成模型引用核对，仍请结合原文判断。")
    elif grounding.get("status") == "blocked":
        st.caption("引用核对未通过，未展示原始草稿。")


def render_context_expansion(expansion: dict[str, Any] | None) -> None:
    if expansion and expansion.get("added_sources"):
        st.caption(
            f"实验性相邻片段补充：本次上下文增加 {expansion['added_sources']} 条教材片段。"
        )


def render_chat_tab(
    book_id: str | None,
    top_k: int,
    temperature: float,
    max_tokens: int,
    enable_hyde: bool,
    load_engine: Callable[[], Any],
    enable_adjacent_context: bool = False,
    enable_decomposition: bool = False,
    verify_citations: bool = False,
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
                render_decomposition(message.get("decomposition"))
                render_grounding(message.get("grounding"))
                render_context_expansion(message.get("context_expansion"))
            else:
                st.markdown(message["content"])

    user_question = st.chat_input("请输入您的问题…")
    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        render_answer_header()
        answer_placeholder = st.empty()
        streamed_chunks: list[str] = []

        def render_chunk(chunk: str) -> None:
            streamed_chunks.append(chunk)
            answer_placeholder.markdown("".join(streamed_chunks) + "▌")

        try:
            with st.spinner("正在检索教材并生成答案…"):
                engine = load_engine()
                result = engine.ask(
                    query=user_question,
                    book_name=book_id,
                    top_k=top_k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_hyde=enable_hyde,
                    use_adjacent_context=enable_adjacent_context,
                    use_decomposition=enable_decomposition,
                    verify_citations=verify_citations,
                    on_answer_chunk=render_chunk,
                )
        except (ProviderError, OSError, RuntimeError, ValueError):
            result = {
                "success": False,
                "answer": "暂时无法完成问答，请检查模型服务或教材索引后重试。",
                "context_sources": [],
            }

        answer = answer_message(result)
        answer_placeholder.markdown(answer)
        sources = result.get("context_sources", result.get("results", []))
        render_answer_details(sources, result.get("execution"))
        render_decomposition(result.get("decomposition"))
        render_grounding(result.get("grounding"))
        render_context_expansion(result.get("context_expansion"))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "execution": result.get("execution"),
            "decomposition": result.get("decomposition"),
            "grounding": result.get("grounding"),
            "context_expansion": result.get("context_expansion"),
        }
    )
