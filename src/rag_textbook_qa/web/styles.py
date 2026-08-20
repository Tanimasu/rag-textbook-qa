"""Custom styling for the packaged Streamlit interface."""

import streamlit as st


def inject_custom_styles() -> None:
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

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            color: var(--ink);
            margin-bottom: 0.45rem;
        }

        .sidebar-brand__icon {
            font-size: 1.55rem;
            line-height: 1;
        }

        .sidebar-brand__text {
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.05;
            white-space: nowrap;
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

        .compute-trace {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.8rem 0 0.4rem;
        }

        .compute-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: #eef7ff;
            border: 1px solid #cfe5f7;
            color: #2c6688;
            font-size: 0.8rem;
            line-height: 1.2;
        }

        .compute-chip--fallback {
            background: #fff7e8;
            border-color: #f4d7a2;
            color: #94631c;
        }

        .compute-chip--timing {
            background: #f5f3ff;
            border-color: #ded8f8;
            color: #655b91;
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

            .sidebar-brand__text {
                font-size: 1.35rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
