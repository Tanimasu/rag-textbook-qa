import streamlit as st

from config.constants import RAGAS_METRIC_LABELS


def render_eval_tab(load_ragas_results, run_ragas_evaluation):
    st.header("RAGAS 评估结果")

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

    if df_existing is None:
        return

    metric_cols = [c for c in RAGAS_METRIC_LABELS if c in df_existing.columns]
    question_col = next((c for c in df_existing.columns if c in {"user_input", "question"}), None)

    if metric_cols:
        st.subheader("汇总指标")
        cols = st.columns(len(metric_cols))
        for col, metric in zip(cols, metric_cols):
            avg = df_existing[metric].mean()
            col.metric(RAGAS_METRIC_LABELS[metric], f"{avg:.3f}")

    if metric_cols:
        st.subheader("逐题得分")
        chart_controls = st.columns([1.2, 1.2, 0.8])
        with chart_controls[0]:
            selected_metric = st.selectbox(
                "查看指标",
                metric_cols,
                format_func=lambda x: RAGAS_METRIC_LABELS[x],
                key="eval_chart_metric",
            )
        with chart_controls[1]:
            sort_mode = st.selectbox(
                "排序方式",
                ["原始顺序", "最低分优先", "最高分优先"],
                key="eval_chart_sort",
            )
        with chart_controls[2]:
            limit_options = [10, 20, 30, 50]
            valid_options = [value for value in limit_options if value < len(df_existing)]
            valid_options.append(len(df_existing))
            valid_options = sorted(set(max(5, value) for value in valid_options))
            default_limit = min(20, len(df_existing))
            display_limit = st.selectbox(
                "显示题数",
                valid_options,
                index=valid_options.index(default_limit) if default_limit in valid_options else 0,
                key="eval_chart_limit",
            )

        chart_df = df_existing.copy()
        chart_df["题号"] = [f"Q{i+1}" for i in range(len(chart_df))]
        if question_col:
            chart_df["问题"] = chart_df[question_col]
            chart_df["问题简称"] = chart_df[question_col].str.slice(0, 22)
            chart_df["问题简称"] = chart_df["问题简称"].where(
                chart_df[question_col].str.len() <= 22,
                chart_df["问题简称"] + "…",
            )
        else:
            chart_df["问题"] = chart_df["题号"]
            chart_df["问题简称"] = chart_df["题号"]

        if sort_mode == "最低分优先":
            chart_df = chart_df.sort_values(selected_metric, ascending=True)
        elif sort_mode == "最高分优先":
            chart_df = chart_df.sort_values(selected_metric, ascending=False)

        chart_df = chart_df.head(int(display_limit))
        chart_plot = chart_df.set_index("题号")[[selected_metric]]
        st.bar_chart(chart_plot, height=340)

        with st.expander("题号对照表", expanded=False):
            question_legend = chart_df[["题号", "问题"]].copy()
            st.dataframe(
                question_legend,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "题号": st.column_config.TextColumn(width="small"),
                    "问题": st.column_config.TextColumn(width="large"),
                },
            )

    st.subheader("详细结果")
    display_df = df_existing.copy()
    display_df["题号"] = [f"Q{i+1}" for i in range(len(display_df))]
    if question_col:
        display_df = display_df.rename(columns={question_col: "问题"})

    display_metric_cols = [RAGAS_METRIC_LABELS[col] for col in metric_cols]
    display_df["平均分"] = display_df[metric_cols].mean(axis=1).round(3)
    display_df = display_df.rename(columns=RAGAS_METRIC_LABELS)

    table_controls = st.columns([1.2, 1, 1, 1])
    with table_controls[0]:
        search_text = st.text_input("搜索问题", placeholder="输入关键词筛选", key="eval_search")
    with table_controls[1]:
        sort_column = st.selectbox(
            "排序列",
            ["题号", "平均分", *display_metric_cols],
            key="eval_table_sort_col",
        )
    with table_controls[2]:
        sort_desc = st.selectbox(
            "排序方向",
            ["降序", "升序"],
            key="eval_table_sort_dir",
        )
    with table_controls[3]:
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
    display_df = display_df.sort_values(sort_column, ascending=(sort_desc == "升序"))

    for col in display_df.select_dtypes(include="number").columns:
        display_df[col] = display_df[col].round(3)

    preferred_columns = [
        column for column in ["题号", "问题", "平均分", *display_metric_cols]
        if column in display_df.columns
    ]
    other_columns = [column for column in display_df.columns if column not in preferred_columns]
    display_df = display_df[preferred_columns + other_columns]

    st.caption(f"当前显示 {len(display_df)} 条结果。")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "题号": st.column_config.TextColumn(width="small"),
            "问题": st.column_config.TextColumn(width="large"),
            "平均分": st.column_config.NumberColumn(format="%.3f"),
        },
    )
