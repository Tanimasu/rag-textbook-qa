"""Compatibility imports for packaged Streamlit application services."""

from rag_textbook_qa.web.services import (
    load_available_books,
    load_engine,
    load_ragas_results,
    run_ragas_evaluation,
)

__all__ = [
    "load_available_books",
    "load_engine",
    "load_ragas_results",
    "run_ragas_evaluation",
]
