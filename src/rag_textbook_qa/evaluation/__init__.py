"""RAGAS evaluation entry points."""

from rag_textbook_qa.evaluation.ragas import (
    RAGASEvaluator,
    build_evaluation_plan,
    create_test_dataset,
    load_test_questions,
    render_evaluation_plan,
    run_evaluation,
    validate_evaluation_output_dir,
)
from rag_textbook_qa.evaluation.retrieval import (
    RETRIEVAL_STRATEGIES,
    RetrievalQuestion,
    SourceEvidence,
    evaluate_retrieval,
    load_retrieval_questions,
    run_retrieval_strategies,
    save_retrieval_report,
    score_context_retention,
    score_ranked_results,
    score_source_evidence_coverage,
    search_with_strategy,
    select_split,
)

__all__ = [
    "RETRIEVAL_STRATEGIES",
    "RAGASEvaluator",
    "RetrievalQuestion",
    "SourceEvidence",
    "build_evaluation_plan",
    "create_test_dataset",
    "evaluate_retrieval",
    "load_retrieval_questions",
    "load_test_questions",
    "render_evaluation_plan",
    "run_evaluation",
    "run_retrieval_strategies",
    "save_retrieval_report",
    "score_context_retention",
    "score_ranked_results",
    "score_source_evidence_coverage",
    "search_with_strategy",
    "select_split",
    "validate_evaluation_output_dir",
]
