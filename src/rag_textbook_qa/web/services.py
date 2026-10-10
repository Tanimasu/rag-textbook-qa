"""Lazy application services for the packaged Streamlit interface."""

from __future__ import annotations

import csv
import tempfile
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
    directory = _settings().paths.evaluations
    candidates = list((directory / "ragas-runs").glob("ragas-*/ragas_evaluation_results.csv"))
    legacy = directory / "ragas_evaluation_results.csv"
    if legacy.is_file():
        candidates.append(legacy)
    if not candidates:
        return None
    last_error = None
    available = []
    unavailable = 0
    for path in candidates:
        try:
            available.append((path.stat().st_mtime_ns, str(path), path))
        except OSError as error:
            # A file can disappear between discovery and sorting. Its age is
            # unknown, so keep this separate from unreadable newer reports.
            unavailable += 1
            last_error = error
    ordered = [row[2] for row in sorted(available, reverse=True)]
    for skipped, results_path in enumerate(ordered):
        try:
            with results_path.open(encoding="utf-8-sig", newline="") as handle:
                headers = next(csv.reader(handle, strict=True), [])
                if len(headers) != len(set(headers)):
                    raise pd.errors.ParserError("评估 CSV 表头存在重复字段")
                handle.seek(0)
                results = pd.read_csv(handle)
        except (OSError, UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            last_error = error
            continue
        results.attrs["unreadable_newer_results"] = skipped
        results.attrs["unavailable_result_candidates"] = unavailable
        return results
    # Let the page offer its existing reload action when no report is readable.
    raise last_error


def run_ragas_evaluation() -> Any | None:
    from rag_textbook_qa.evaluation import load_test_questions, run_evaluation

    paths = _settings().paths
    test_questions = load_test_questions(
        paths.evaluation_data / "test_questions.json"
    )
    runs = paths.evaluations / "ragas-runs"
    runs.mkdir(parents=True, exist_ok=True)
    output_dir = tempfile.mkdtemp(prefix="ragas-", dir=runs)
    engine = load_engine()
    return run_evaluation(engine, test_questions, output_dir)
