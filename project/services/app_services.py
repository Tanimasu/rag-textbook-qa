import json
import os
import sqlite3

import pandas as pd
import streamlit as st

from config.constants import RAGAS_RESULTS_PATH, TEST_QUESTIONS_PATH, VECTOR_DB_PATH
from ui.helpers import format_book_label


@st.cache_data(show_spinner=False)
def load_available_books():
    sqlite_path = os.path.join(VECTOR_DB_PATH, "chroma.sqlite3")

    if not os.path.exists(sqlite_path):
        return [("全部", None)]

    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM collections
            WHERE name LIKE 'textbook_%'
            ORDER BY name
            """
        ).fetchall()

    book_ids = [row[0].replace("textbook_", "") for row in rows]
    options = [(format_book_label(book_id), book_id) for book_id in book_ids]
    options.append(("全部", None))
    return options


@st.cache_resource(show_spinner="正在加载 RAG 引擎，请稍候…")
def load_engine():
    from rag_engine import RAGEngine

    return RAGEngine(db_path="./vector_db", verbose=False)


def load_ragas_results():
    if os.path.exists(RAGAS_RESULTS_PATH):
        return pd.read_csv(RAGAS_RESULTS_PATH, encoding="utf-8-sig")
    return None


def run_ragas_evaluation():
    from ragas_evaluation import RAGASEvaluator

    engine = load_engine()
    with open(TEST_QUESTIONS_PATH, encoding="utf-8") as f:
        test_questions = json.load(f)

    evaluator = RAGASEvaluator()
    dataset = evaluator.prepare_evaluation_data(engine, test_questions)
    result = evaluator.evaluate(dataset)
    df = evaluator.print_results(result)
    if df is not None:
        df.to_csv(RAGAS_RESULTS_PATH, index=False, encoding="utf-8-sig")
    return load_ragas_results()
