"""Lazy application services for the packaged Streamlit interface."""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag_textbook_qa.config import Settings
from rag_textbook_qa.web.helpers import format_book_label


def _settings() -> Settings:
    return Settings.load()


@st.cache_data(show_spinner=False)
def load_available_books() -> list[tuple[str, str | None]]:
    sqlite_path = _settings().paths.vector_db / "chroma.sqlite3"
    if not sqlite_path.exists():
        return [("全部", None)]

    with sqlite3.connect(sqlite_path) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM collections
            WHERE name LIKE 'textbook_%'
            ORDER BY name
            """
        ).fetchall()

    book_ids = [row[0].removeprefix("textbook_") for row in rows]
    options = [(format_book_label(book_id), book_id) for book_id in book_ids]
    options.append(("全部", None))
    return options


@st.cache_resource(show_spinner="正在加载 RAG 引擎，请稍候…")
def load_engine() -> Any:
    from rag_textbook_qa.rag import RAGEngine

    paths = _settings().paths
    load_dotenv(paths.root / "project" / ".env", override=False)
    return RAGEngine(db_path=str(paths.vector_db), verbose=False)


def load_ragas_results() -> Any | None:
    results_path = _settings().paths.evaluations / "ragas_evaluation_results.csv"
    if not results_path.exists():
        return None
    return pd.read_csv(results_path, encoding="utf-8-sig")


def run_ragas_evaluation() -> Any | None:
    from rag_textbook_qa.evaluation import load_test_questions, run_evaluation

    paths = _settings().paths
    engine = load_engine()
    test_questions = load_test_questions(
        paths.evaluation_data / "test_questions.json"
    )
    run_evaluation(engine, test_questions, paths.evaluations)
    return load_ragas_results()
