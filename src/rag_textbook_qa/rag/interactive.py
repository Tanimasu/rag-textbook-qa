"""Interactive terminal entry point kept separate from the RAG engine."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from rag_textbook_qa.config import Settings
from rag_textbook_qa.rag.engine import RAGEngine

TEST_QUERIES = (
    {"query": "什么是进程？", "book": "os"},
    {"query": "CPU 的功能是什么？", "book": "computer_organization"},
    {"query": "什么是死锁？如何预防？", "book": "os"},
)


def interactive_main(
    *,
    workspace: str | Path | None = None,
    db_path: str | Path | None = None,
    enable_llm: bool = True,
    enable_reranker: bool = True,
    enable_hyde: bool = True,
) -> None:
    settings = Settings.load(workspace)
    load_dotenv(settings.paths.root / "project" / ".env", override=False)
    engine = RAGEngine(
        db_path=db_path or settings.paths.vector_db,
        enable_llm=enable_llm,
        enable_reranker=enable_reranker,
        enable_hyde=enable_hyde,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("计算机课程 AI 助教系统")
    print("=" * 70)
    engine.vectorizer.list_books()
    print("输入 test 运行测试，输入 quit 退出。")

    while True:
        user_input = input("\n你的问题 > ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("再见！")
            return
        if user_input.lower() == "test":
            for index, case in enumerate(TEST_QUERIES, 1):
                print(f"\n测试用例 {index}/{len(TEST_QUERIES)}")
                engine.ask(
                    query=case["query"],
                    book_name=case["book"],
                    top_k=5,
                )
            continue
        engine.ask(query=user_input, top_k=5)
