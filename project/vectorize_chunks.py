"""Compatibility entry point for the packaged textbook vectorizer."""

from pathlib import Path

from rag_textbook_qa.indexing.vectorizer import (
    MultiBookVectorizer,
    interactive_main,
    parse_selection,
)

DEFAULT_VECTOR_DB_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "vector_db"
_parse_selection = parse_selection
main = interactive_main

__all__ = ["DEFAULT_VECTOR_DB_PATH", "MultiBookVectorizer", "main"]


if __name__ == "__main__":
    main()
