"""Compatibility entry point for the packaged RAGAS evaluator."""

from rag_textbook_qa.cli import main as cli_main
from rag_textbook_qa.evaluation import RAGASEvaluator, create_test_dataset

__all__ = ["RAGASEvaluator", "create_test_dataset", "main"]


def main() -> int:
    """Preserve the legacy script's RAG plus baseline behavior."""

    return cli_main(["evaluate", "--baseline"])


if __name__ == "__main__":
    raise SystemExit(main())
