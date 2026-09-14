"""Textbook vector indexing backed by a local Chroma database."""

from rag_textbook_qa.indexing.vectorizer import (
    MultiBookVectorizer,
    fetch_indexed_chunks,
    list_indexed_books,
)

__all__ = ["MultiBookVectorizer", "fetch_indexed_chunks", "list_indexed_books"]
