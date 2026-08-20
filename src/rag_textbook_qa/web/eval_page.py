"""RAGAS results tab for the Web interface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from rag_textbook_qa.web.constants import RAGAS_METRIC_LABELS


def render_eval_tab(
    load_ragas_results: Callable[[], Any],
    run_ragas_evaluation: Callable[[], Any],
) -> None:
    st.header("RAGAS 评估结果")

    results = load_ragas_results()
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_eval = st.button("运行评估", width="stretch")
    with col_info:
        if results is not None:
            st.caption(f"已有评估结果（{len(results)} 条），点击「运行评估」重新生成。")
        else:
            st.caption("尚无评估结果，点击「运行评估」开始（需要几分钟）。")

    if run_eval:
        with st.spinner("正在运行 RAGAS 评估，请耐心等待…"):
            results = run_ragas_evaluation()
        if results is not None:
            st.success("评估完成！")
        else:
            st.error("评估完成但未生成结果文件，请检查控制台输出。")

    if results is None:
        return

    metric_cols = [column for column in RAGAS_METRIC_LABELS if column in results.columns]
    question_col = next(
        (column for column in results.columns if column in {"user_input", "question"}),
        None,
    )

    if metric_cols:
        st.subheader("汇总指标")
        columns = st.columns(len(metric_cols))
        for column, metric in zip(columns, metric_cols):
            column.metric(RAGAS_METRIC_LABELS[metric], f"{results[metric].mean():.3f}")

        _render_score_chart(results, metric_cols, question_col)

    _render_results_table(results, metric_cols, question_col)


def _render_score_chart(results: Any, metric_cols: list[str], question_col: str | None) -> None:
    st.subheader("逐题得分")
    controls = st.columns([1.15, 1.15, 1.05, 0.8])
    with controls[0]:
        selected_metric = st.selectbox(
            "显示指标",
            ["全部指标", "平均分", *metric_cols],
            format_func=lambda value: (
                value if value in {"全部指标", "平均分"} else RAGAS_METRIC_LABELS[value]
            ),
            key="eval_chart_metric_mode",
        )
    with controls[1]:
        sort_metric = st.selectbox(
            "排序依据",
            ["平均分", *metric_cols],
            format_func=lambda value: (
                "平均分" if value == "平均分" else RAGAS_METRIC_LABELS[value]
            ),
            key="eval_chart_sort_metric",
        )
    with controls[2]:
        sort_mode = st.selectbox(
            "排序方式",
            ["原始顺序", "最低分优先", "最高分优先"],
            key="eval_chart_sort",
        )
    with controls[3]:
        limit_options = [value for value in [10, 20, 30, 50] if value < len(results)]
        limit_options.append(len(results))
        limit_options = sorted({max(5, value) for value in limit_options})
        default_limit = min(20, len(results))
        display_limit = st.selectbox(
            "显示题数",
            limit_options,
            index=limit_options.index(default_limit) if default_limit in limit_options else 0,
            key="eval_chart_limit",
        )

    chart_df = results.copy()
    chart_df["题号"] = [f"Q{index + 1}" for index in range(len(chart_df))]
    chart_df["平均分"] = chart_df[metric_cols].mean(axis=1).round(3)
    chart_df["问题"] = chart_df[question_col] if question_col else chart_df["题号"]
    if sort_mode != "原始顺序":
        chart_df = chart_df.sort_values(sort_metric, ascending=sort_mode == "最低分优先")
    chart_df = chart_df.head(int(display_limit))

    if selected_metric == "全部指标":
        chart_plot = chart_df.set_index("题号")[["平均分", *metric_cols]].rename(
            columns={"平均分": "平均分", **RAGAS_METRIC_LABELS}
        )
    elif selected_metric == "平均分":
        chart_plot = chart_df.set_index("题号")[["平均分"]]
    else:
        chart_plot = chart_df.set_index("题号")[[selected_metric]].rename(
            columns=RAGAS_METRIC_LABELS
        )
    st.line_chart(chart_plot, height=340)

    with st.expander("题号对照表", expanded=False):
        st.dataframe(
            chart_df[["题号", "问题"]].copy(),
            width="stretch",
            hide_index=True,
            column_config={
                "题号": st.column_config.TextColumn(width="small"),
                "问题": st.column_config.TextColumn(width="large"),
            },
        )


def _render_results_table(
    results: Any,
    metric_cols: list[str],
    question_col: str | None,
) -> None:
    st.subheader("详细结果")
    display_df = results.copy()
    display_df["题号"] = [f"Q{index + 1}" for index in range(len(display_df))]
    if question_col:
        display_df = display_df.rename(columns={question_col: "问题"})

    display_metric_cols = [RAGAS_METRIC_LABELS[column] for column in metric_cols]
    display_df["平均分"] = display_df[metric_cols].mean(axis=1).round(3)
    display_df = display_df.rename(columns=RAGAS_METRIC_LABELS)

    controls = st.columns([1.2, 1, 1, 1])
    with controls[0]:
        search_text = st.text_input("搜索问题", placeholder="输入关键词筛选", key="eval_search")
    with controls[1]:
        sort_column = st.selectbox(
            "排序列",
            ["题号", "平均分", *display_metric_cols],
            key="eval_table_sort_col",
        )
    with controls[2]:
        sort_desc = st.selectbox("排序方向", ["降序", "升序"], key="eval_table_sort_dir")
    with controls[3]:
        score_threshold = st.slider(
            "最高平均分",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="eval_score_threshold",
        )

    if "问题" in display_df.columns and search_text:
        display_df = display_df[
            display_df["问题"].astype(str).str.contains(search_text, case=False, na=False)
        ]
    display_df = display_df[display_df["平均分"] <= score_threshold]
    display_df = display_df.sort_values(sort_column, ascending=sort_desc == "升序")
    for column in display_df.select_dtypes(include="number").columns:
        display_df[column] = display_df[column].round(3)

    preferred_columns = [
        column
        for column in ["题号", "问题", "平均分", *display_metric_cols]
        if column in display_df.columns
    ]
    remaining_columns = [
        column for column in display_df.columns if column not in preferred_columns
    ]
    display_df = display_df[preferred_columns + remaining_columns]
    st.caption(f"当前显示 {len(display_df)} 条结果。")
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "题号": st.column_config.TextColumn(width="small"),
            "问题": st.column_config.TextColumn(width="large"),
            "平均分": st.column_config.NumberColumn(format="%.3f"),
        },
    )
