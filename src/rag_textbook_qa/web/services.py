"""Lazy application services for the packaged Streamlit interface."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag_textbook_qa.config import Settings
from rag_textbook_qa.indexing.revision import index_catalog
from rag_textbook_qa.web.helpers import format_book_label


def _settings() -> Settings:
    return Settings.load()


def load_available_books() -> list[tuple[str, str | None]]:
    # This is a small catalog read, so a Streamlit rerun sees new/deleted books.
    rows = index_catalog(_settings().paths.vector_db)
    book_ids = sorted({row[0].removeprefix("textbook_") for row in rows})
    options = [(format_book_label(book_id), book_id) for book_id in book_ids]
    options.append(("全部", None))
    return options


@st.cache_resource(show_spinner="正在加载 RAG 引擎，请稍候…")
def _cached_engine(db_path: str) -> Any:
    from rag_textbook_qa.rag import RAGEngine

    paths = _settings().paths
    load_dotenv(paths.root / "project" / ".env", override=False)
    return RAGEngine(db_path=db_path, verbose=False)


def load_engine() -> Any:
    engine = _cached_engine(str(_settings().paths.vector_db))
    engine.refresh_index_if_changed()
    return engine


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
