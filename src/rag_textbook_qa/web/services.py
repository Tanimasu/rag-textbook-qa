"""Lazy application services for the packaged Streamlit interface."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
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


def _load_legacy_ragas_evaluator(workspace: Path) -> type[Any]:
    """Load the still-legacy evaluator only when the user starts an evaluation."""

    project_path = workspace / "project"
    module_path = project_path / "ragas_evaluation.py"
    if not module_path.is_file():
        raise RuntimeError(f"找不到 RAGAS 兼容入口: {module_path}")

    spec = importlib.util.spec_from_file_location(
        "_rag_textbook_qa_legacy_ragas_evaluation",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 RAGAS 兼容入口: {module_path}")

    module = importlib.util.module_from_spec(spec)
    project_directory = str(project_path)
    added_to_path = project_directory not in sys.path
    if added_to_path:
        sys.path.insert(0, project_directory)
    try:
        spec.loader.exec_module(module)
    finally:
        if added_to_path:
            sys.path.remove(project_directory)
    return module.RAGASEvaluator


def run_ragas_evaluation() -> Any | None:
    paths = _settings().paths
    evaluator_type = _load_legacy_ragas_evaluator(paths.root)
    engine = load_engine()
    with (paths.evaluation_data / "test_questions.json").open(encoding="utf-8") as file:
        test_questions = json.load(file)

    evaluator = evaluator_type()
    dataset = evaluator.prepare_evaluation_data(engine, test_questions)
    result = evaluator.evaluate(dataset)
    dataframe = evaluator.print_results(result)
    if dataframe is not None:
        paths.evaluations.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(
            paths.evaluations / "ragas_evaluation_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return load_ragas_results()
