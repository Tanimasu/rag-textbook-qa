"""Compatibility entry point for the packaged RAG engine."""

from pathlib import Path

from rag_textbook_qa.rag import RAGEngine, interactive_main

DEFAULT_VECTOR_DB_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "vector_db"
main = interactive_main

__all__ = ["DEFAULT_VECTOR_DB_PATH", "RAGEngine", "main"]


if __name__ == "__main__":
    main()
