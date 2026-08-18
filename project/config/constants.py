from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = str(REPOSITORY_ROOT / "project")
DATA_DIR = REPOSITORY_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARSED_DATA_DIR = DATA_DIR / "parsed"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
CHUNKS_DIR = DATA_DIR / "chunks"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"
ARTIFACTS_DIR = REPOSITORY_ROOT / "artifacts"
EVALUATIONS_DIR = ARTIFACTS_DIR / "evaluations"
VECTOR_DB_PATH = ARTIFACTS_DIR / "vector_db"
TEST_QUESTIONS_PATH = EVALUATION_DATA_DIR / "test_questions.json"
RAGAS_RESULTS_PATH = EVALUATIONS_DIR / "ragas_evaluation_results.csv"

BOOK_NAME_LABELS = {
    "os": "操作系统",
    "computer_organization": "计算机组成原理",
    "computer_network": "计算机网络",
    "database": "数据库原理及应用",
    "data_structure": "数据结构",
}

RAGAS_METRIC_LABELS = {
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "context_precision": "上下文精确度",
    "context_recall": "上下文召回率",
}
