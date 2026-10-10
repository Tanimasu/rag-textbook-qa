"""RAGAS results tab for the Web interface."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real
from typing import Any

import pandas as pd
import streamlit as st

from rag_textbook_qa.web.constants import RAGAS_METRIC_LABELS


def _question_labels(results: Any) -> list[str]:
    if "question_index" not in results.columns:
        return [f"Q{index}" for index in range(1, len(results) + 1)]
    indices = list(results["question_index"])
    try:
        valid = all(
            isinstance(index, Real) and not pd.api.types.is_bool(index)
            and math.isfinite(index) and index > 0 and int(index) == index
            for index in indices
        ) and len(set(indices)) == len(indices)
    except (ValueError, OverflowError):
        valid = False
    if valid:
        return [f"Q{int(index)}" for index in indices]
    return [f"结果行{index}" for index in range(1, len(results) + 1)]


def render_eval_tab(
    load_ragas_results: Callable[[], Any],
    run_ragas_evaluation: Callable[[], Any],
) -> None:
    st.header("RAGAS 评估结果")

    load_failed = False
    try:
        results = load_ragas_results()
    except Exception:  # noqa: BLE001 - A partial or unreadable CSV must not break chat.
        results, load_failed = None, True
        st.warning("评估结果暂时无法读取，文件可能尚未写完。可以重新读取，原记录会保留。")
        if st.button("重新读取结果", key="reload_eval_results"):
            st.rerun()
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_eval = st.button("运行评估", width="stretch")
    with col_info:
        if results is not None:
            st.caption(f"已有评估结果（{len(results)} 条），点击「运行评估」重新生成。")
        elif load_failed:
            st.caption("重新读取不调用模型；「运行评估」会发起一批新的评估。")
        else:
            st.caption("尚无评估结果，点击「运行评估」开始（需要几分钟）。")

    if run_eval:
        try:
            with st.spinner("正在运行 RAGAS 评估，请耐心等待…"):
                new_results = run_ragas_evaluation()
            if new_results is not None:
                results = new_results
                st.success("评估完成！")
            else:
                st.error("本次评估未生成结果文件；已有结果保留。")
        except Exception:  # noqa: BLE001 - Keep the previous report available after a failed run.
            st.error("本次评估未完成，已有结果保留。请检查模型服务后再试。")

    if results is not None and results.attrs.get("unreadable_newer_results", 0):
        st.warning("较新的评估结果暂时无法读取，当前显示此前可读取的结果；原文件保留。")
        if st.button("重新读取结果", key="reload_eval_results"):
            st.rerun()
    if results is None:
        return

    metric_cols = [column for column in RAGAS_METRIC_LABELS if column in results.columns]
    results = results.copy()
    labels = _question_labels(results)
    if labels and labels[0].startswith("结果行"):
        st.warning("题号缺失、重复或格式异常，图表与明细按结果行号显示；原题号保留，不能据此配对题目。")
    invalid_scores = 0
    for metric in metric_cols:
        original = results[metric]
        numeric = pd.to_numeric(original, errors="coerce")
        numeric = numeric.mask(original.map(pd.api.types.is_bool)).replace(
            [float("inf"), float("-inf")], float("nan")
        )
        invalid_scores += int((original.notna() & numeric.isna()).sum())
        results[metric] = numeric
    if invalid_scores:
        st.warning("部分得分格式异常，按未评分展示；原始结果文件保留。")
    question_col = next(
        (column for column in results.columns if column in {"user_input", "question"}),
        None,
    )

    if metric_cols:
        st.subheader("汇总指标")
        columns = st.columns(len(metric_cols))
        for column, metric in zip(columns, metric_cols):
            values = results[metric].dropna()
            column.metric(RAGAS_METRIC_LABELS[metric], f"{values.mean():.3f}" if len(values) else "—")
            column.caption(f"{len(values)}/{len(results)} 题有得分")

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
        limit_options = sorted(set(limit_options))
        default_limit = min(20, len(results))
        display_limit = st.selectbox(
            "显示题数",
            limit_options,
            index=limit_options.index(default_limit) if default_limit in limit_options else 0,
            key="eval_chart_limit",
        )

    chart_df = results.copy()
    chart_df["题号"] = _question_labels(results)
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
    display_df["题号"] = _question_labels(results)
    if len(display_df) and display_df["题号"].iloc[0].startswith("结果行"):
        display_df = display_df.rename(columns={"question_index": "原题号"})
        display_df["原题号"] = display_df["原题号"].map(
            lambda value: "未提供" if pd.api.types.is_scalar(value) and pd.isna(value)
            else str(value)
        )
    else:
        display_df = display_df.drop(columns=["question_index"], errors="ignore")
    if question_col:
        display_df = display_df.rename(columns={question_col: "问题"})

    display_metric_cols = [RAGAS_METRIC_LABELS[column] for column in metric_cols]
    display_df["平均分"] = display_df[metric_cols].mean(axis=1).round(3)
    display_df["有效指标"] = display_df[metric_cols].notna().sum(axis=1).map(
        lambda count: f"{count}/{len(metric_cols)}"
    )
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
            display_df["问题"].astype(str).str.contains(search_text, case=False, na=False, regex=False)
        ]
    display_df = display_df[display_df["平均分"].isna() | (display_df["平均分"] <= score_threshold)]
    display_df = display_df.sort_values(
        sort_column,
        ascending=sort_desc == "升序",
        key=lambda values: values.str.extract(r"(\d+)$", expand=False).map(int)
        if values.name == "题号" else values,
    )
    for column in display_df.select_dtypes(include="number").columns:
        display_df[column] = display_df[column].round(3)

    preferred_columns = [
        column
        for column in ["题号", "问题", "平均分", "有效指标", *display_metric_cols]
        if column in display_df.columns
    ]
    remaining_columns = [
        column for column in display_df.columns if column not in preferred_columns
    ]
    display_df = display_df[preferred_columns + remaining_columns]
    st.caption(f"当前显示 {len(display_df)} 条结果。")
    st.caption("未评分题目保留展示；平均分按本题可用指标计算，缺失得分不记为零。")
    # Streamlit's numeric cells render null as "None" even with Styler na_rep.
    # Format only the final table; filtering, ordering and stored scores stay numeric.
    for column in ("平均分", *display_metric_cols):
        display_df[column] = display_df[column].map(
            lambda value: "未评分" if pd.isna(value) else f"{value:.3f}"
        )
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "题号": st.column_config.TextColumn(width="small"),
            "问题": st.column_config.TextColumn(width="large"),
            "平均分": st.column_config.TextColumn(),
        },
    )
