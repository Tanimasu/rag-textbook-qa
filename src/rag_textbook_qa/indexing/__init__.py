"""Textbook vector indexing backed by a local Chroma database."""

from rag_textbook_qa.indexing.vectorizer import (
    MultiBookVectorizer,
    list_indexed_books,
)

__all__ = ["MultiBookVectorizer", "list_indexed_books"]
