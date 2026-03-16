import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAGAS_RESULTS_PATH = os.path.join(BASE_DIR, "ragas_evaluation_results.csv")
TEST_QUESTIONS_PATH = os.path.join(BASE_DIR, "test_questions.json")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

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
