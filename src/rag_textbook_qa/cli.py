"""Unified command-line entry point for the cross-platform package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rag_textbook_qa import __version__
from rag_textbook_qa.config import Settings, WorkspaceNotFoundError
from rag_textbook_qa.diagnostics.doctor import (
    diagnostics_as_dict,
    render_diagnostics,
)
from rag_textbook_qa.providers.base import (
    DEFAULT_QUERY_INSTRUCTION,
    PROTOCOL_VERSION,
    ModelIdentity,
    ModelMismatchError,
    ProviderError,
    ProviderProtocolError,
)
from rag_textbook_qa.providers.config import ComputeSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-qa")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--workspace",
        help="项目工作区路径；默认读取 RAG_QA_HOME 或自动发现源码工作区",
    )

    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="执行不会加载模型的环境检查")
    doctor.add_argument("--json", action="store_true", help="输出 JSON")
    doctor.add_argument(
        "--index",
        action="store_true",
        help="额外检查向量库内容与已登记的冲突规则；会导入 chromadb，但不加载模型",
    )

    ingest = commands.add_parser("ingest", help="清洗、分块和检查教材中间产物")
    ingest_commands = ingest.add_subparsers(dest="ingest_command", required=True)

    clean = ingest_commands.add_parser("clean", help="清洗一个 Markdown 文件")
    clean.add_argument("input", type=Path, help="输入 Markdown")
    clean.add_argument("--output", required=True, type=Path, help="输出 Markdown")
    clean.add_argument("--force", action="store_true", help="允许覆盖已有输出")

    chunk = ingest_commands.add_parser("chunk", help="将 Markdown 按标题结构分块")
    chunk.add_argument("input", type=Path, help="输入 Markdown；--batch 时为目录")
    chunk.add_argument("--output", required=True, type=Path, help="输出 JSON 或目录")
    chunk.add_argument("--batch", action="store_true", help="批量处理 *_cleaned.md")
    chunk.add_argument("--max-chunk-size", type=int, default=800)
    chunk.add_argument("--min-chunk-size", type=int, default=100)
    chunk.add_argument("--overlap-size", type=int, default=50)
    chunk.add_argument("--force", action="store_true", help="允许覆盖已有输出")
    chunk.add_argument("--no-preview", action="store_true", help="不生成文本预览")

    check = ingest_commands.add_parser("check", help="检查 Markdown 或 chunks 质量")
    check.add_argument("input", type=Path, help="输入 Markdown 或 chunks JSON")
    check.add_argument(
        "--kind",
        choices=("auto", "markdown", "chunks"),
        default="auto",
        help="检查类型；auto 根据 .json 后缀判断",
    )
    check.add_argument("--json", action="store_true", help="输出结构化 JSON")

    index = commands.add_parser("index", help="构建和检查本地 Chroma 教材索引")
    index_commands = index.add_subparsers(dest="index_command", required=True)
    index_build = index_commands.add_parser("build", help="向量化一个 chunks JSON")
    index_build.add_argument("input", type=Path, help="输入 *_chunks.json")
    index_build.add_argument(
        "--book",
        help="稳定教材标识，如 database；默认根据文件名推断",
    )
    index_build.add_argument("--db-path", type=Path, help="覆盖 artifacts/vector_db")
    index_build.add_argument("--batch-size", type=int, default=32)
    index_build.add_argument("--model", help="覆盖 embedding 模型")
    index_build.add_argument(
        "--append",
        action="store_true",
        help="追加到现有集合；默认完成后原子替换",
    )
    index_list = index_commands.add_parser("list", help="列出已索引教材，不加载模型")
    index_list.add_argument("--db-path", type=Path, help="覆盖 artifacts/vector_db")
    index_list.add_argument("--json", action="store_true", help="输出结构化 JSON")
    index_check = index_commands.add_parser(
        "check",
        help="核对已登记的冲突规则在当前索引中是否仍然命中，不加载模型",
    )
    index_check.add_argument("--db-path", type=Path, help="覆盖 artifacts/vector_db")
    index_check.add_argument("--json", action="store_true", help="输出结构化 JSON")

    chat = commands.add_parser("chat", help="启动交互式教材问答")
    chat.add_argument("--db-path", type=Path, help="覆盖 artifacts/vector_db")
    chat.add_argument("--no-llm", action="store_true", help="只检索，不调用 LLM")
    chat.add_argument("--no-reranker", action="store_true", help="禁用重排序")
    chat.add_argument("--no-hyde", action="store_true", help="禁用 HyDE")
    chat.add_argument(
        "--context-budget",
        type=int,
        help="送入模型的上下文字符预算；默认 4000",
    )

    evaluate = commands.add_parser("evaluate", help="运行 RAGAS 质量评估")
    evaluate.add_argument("--questions", type=Path, help="覆盖评估问题 JSON")
    evaluate.add_argument("--db-path", type=Path, help="覆盖 artifacts/vector_db")
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        help="覆盖 artifacts/evaluations，避免覆盖已有评估结果",
    )
    evaluate.add_argument(
        "--baseline",
        action="store_true",
        help="同时运行无 RAG baseline（会增加 API 调用）",
    )
    evaluate.add_argument(
        "--hyde",
        action="store_true",
        help="显式启用实验性 HyDE（额外调用 LLM；默认关闭以匹配公开产品）",
    )
    evaluate.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="送入上下文选择的检索条数；默认 5，与公开产品一致",
    )
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查题集、模型、计算后端和费用影响，不加载模型或调用 API",
    )

    retrieval_evaluate = commands.add_parser(
        "evaluate-retrieval",
        help="对比检索策略，不调用 LLM",
    )
    retrieval_evaluate.add_argument(
        "--questions",
        type=Path,
        help="覆盖检索评估问题 JSON",
    )
    retrieval_evaluate.add_argument(
        "--db-path",
        type=Path,
        help="覆盖 artifacts/vector_db",
    )
    retrieval_evaluate.add_argument(
        "--output-dir",
        type=Path,
        help="覆盖 artifacts/evaluations/retrieval",
    )
    retrieval_evaluate.add_argument(
        "--strategy",
        choices=("all", "bm25", "embedding", "hybrid", "hybrid-rerank"),
        default="all",
        help="检索策略；默认依次评测全部策略",
    )
    retrieval_evaluate.add_argument("--top-k", type=int, default=5)
    retrieval_evaluate.add_argument(
        "--context-budget",
        type=int,
        default=4000,
        help="按该字符预算装入上下文并统计证据保留率；默认与问答一致，设为 0 只评测检索",
    )
    retrieval_evaluate.add_argument(
        "--split",
        choices=("dev", "holdout", "all"),
        default="dev",
        help="评测划分；默认只用 dev，留出集不参与调参",
    )

    generation_evaluate = commands.add_parser(
        "evaluate-generation",
        help="固定上下文重复采样，逐条核对回答陈述（会调用生成与评判 API）",
    )
    generation_evaluate.add_argument(
        "--cases", type=Path, required=True, help="冻结了实际上下文的题目 JSON"
    )
    generation_evaluate.add_argument(
        "--arm",
        action="append",
        required=True,
        help="方案：名称=上下文@温度 或 名称=上下文@stored；第一个方案作为对照基准",
    )
    generation_evaluate.add_argument(
        "--output-dir", type=Path, required=True, help="实验目录；中断后用同样参数重跑即可续跑"
    )
    generation_evaluate.add_argument("--samples", type=int, default=5, help="每题每个方案的采样次数")
    generation_evaluate.add_argument("--concurrency", type=int, default=3)
    generation_evaluate.add_argument("--seed", type=int, default=0)
    generation_evaluate.add_argument("--max-tokens", type=int, default=2000)

    app = commands.add_parser("app", help="启动 Streamlit 教材问答界面")
    app.add_argument(
        "--backend",
        choices=("local", "remote"),
        help="仅本次启动覆盖 RAG_QA_COMPUTE_BACKEND",
    )
    app.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        help="仅本次启动覆盖 RAG_QA_DEVICE",
    )
    app.add_argument("--host", default="127.0.0.1", help="Web 界面监听地址")
    app.add_argument("--port", type=int, default=8501, help="Web 界面监听端口")
    app.add_argument("--no-browser", action="store_true", help="启动时不自动打开浏览器")

    serve_api = commands.add_parser(
        "serve",
        help="启动对外问答服务：聊天页、REST API 与接口文档",
    )
    serve_api.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址；对外开放需设置 RAG_QA_ACCESS_CODE，或显式加 --public",
    )
    serve_api.add_argument("--port", type=int, default=8000)
    serve_api.add_argument("--db-path", type=Path, help="覆盖 artifacts/vector_db")
    serve_api.add_argument(
        "--public",
        action="store_true",
        help="确认在没有访问口令时对外开放；限流和每日生成上限仍然生效",
    )
    serve_api.add_argument("--no-warmup", action="store_true", help="跳过启动时的模型预热")

    feedback = commands.add_parser("feedback", help="管理用户主动提交的问答反馈")
    feedback_commands = feedback.add_subparsers(dest="feedback_command", required=True)
    feedback_summary = feedback_commands.add_parser("summary", help="汇总反馈和响应耗时")
    feedback_summary.add_argument(
        "--database",
        type=Path,
        help="覆盖 artifacts/product/feedback.sqlite3",
    )
    feedback_summary.add_argument("--json", action="store_true", help="输出结构化 JSON")
    feedback_candidates = feedback_commands.add_parser(
        "candidates",
        help="把负面反馈整理成人工审核候选，不修改正式评测集",
    )
    feedback_candidates.add_argument(
        "--database",
        type=Path,
        help="覆盖 artifacts/product/feedback.sqlite3",
    )
    feedback_candidates.add_argument(
        "--output",
        required=True,
        type=Path,
        help="候选 JSON 路径（建议放在已忽略的 artifacts/product/）",
    )
    feedback_candidates.add_argument("--force", action="store_true", help="允许覆盖已有候选文件")
    feedback_export = feedback_commands.add_parser("export", help="将本地反馈导出为 JSONL")
    feedback_export.add_argument(
        "--database",
        type=Path,
        help="覆盖 artifacts/product/feedback.sqlite3",
    )
    feedback_export.add_argument(
        "--output",
        required=True,
        type=Path,
        help="导出 JSONL 路径（建议放在已忽略的 artifacts/product/）",
    )
    feedback_export.add_argument("--force", action="store_true", help="允许覆盖已有导出文件")

    worker = commands.add_parser("worker", help="运行远程 embedding/reranker Worker")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    serve = worker_commands.add_parser("serve", help="启动模型 Worker HTTP 服务")
    serve.add_argument("--host", default="127.0.0.1", help="监听 IP；远程时建议使用 Tailscale IP")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--embedding-model")
    serve.add_argument("--reranker-model")
    serve.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    serve.add_argument(
        "--warmup",
        action="store_true",
        help="启动监听前加载 embedding 和 reranker 模型",
    )
    worker_check = worker_commands.add_parser(
        "check",
        help="只请求 /health，安全检查远程 Worker 配置",
    )
    worker_check.add_argument("--url", help="覆盖 RAG_QA_REMOTE_URL")
    worker_check.add_argument("--timeout", type=float, help="覆盖连接超时秒数")
    worker_check.add_argument("--json", action="store_true", help="输出结构化 JSON")
    return parser


def _run_ingest(args: argparse.Namespace) -> int:
    if args.ingest_command == "clean":
        from rag_textbook_qa.ingestion.cleaner import clean_markdown

        clean_markdown(args.input, args.output, overwrite=args.force)
        return 0

    if args.ingest_command == "chunk":
        from rag_textbook_qa.ingestion.chunker import (
            batch_chunk_markdown,
            chunk_markdown,
        )

        options = {
            "max_chunk_size": args.max_chunk_size,
            "min_chunk_size": args.min_chunk_size,
            "overlap_size": args.overlap_size,
            "overwrite": args.force,
            "write_preview": not args.no_preview,
        }
        if args.batch:
            result = batch_chunk_markdown(args.input, args.output, **options)
            print(
                f"批量分块完成：新建 {len(result.created)}，"
                f"跳过 {len(result.skipped_existing)}"
            )
        else:
            chunk_markdown(args.input, args.output, **options)
        return 0

    if args.ingest_command == "check":
        from rag_textbook_qa.ingestion.quality import (
            analyze_chunks,
            analyze_markdown,
            render_chunks_report,
            render_markdown_report,
        )

        kind = args.kind
        if kind == "auto":
            kind = "chunks" if args.input.suffix.lower() == ".json" else "markdown"
        if kind == "chunks":
            report = analyze_chunks(args.input)
            rendered = render_chunks_report(report)
        else:
            report = analyze_markdown(args.input)
            rendered = render_markdown_report(report)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(rendered)
        return 0

    raise ValueError(f"未知 ingest 命令: {args.ingest_command}")


def _load_project_environment(env_path: Path) -> None:
    """Load project configuration while making dotenv precedence visible."""

    from dotenv import dotenv_values, load_dotenv

    process_token = os.environ.get("RAG_QA_WORKER_TOKEN")
    file_token = dotenv_values(env_path).get("RAG_QA_WORKER_TOKEN")
    if (
        process_token is not None
        and file_token is not None
        and process_token != file_token
    ):
        print(
            "警告: 进程环境变量 RAG_QA_WORKER_TOKEN 与 project/.env 不一致；"
            "本次将使用进程环境变量。",
            file=sys.stderr,
        )
    load_dotenv(env_path, override=False)


def _conflict_pin_problems(
    db_path: Path, book_name: str | None = None
) -> tuple[list[dict[str, str]], int]:
    """Resolve every registered conflict pin against the live index."""

    from rag_textbook_qa.indexing import fetch_indexed_chunks
    from rag_textbook_qa.rag.conflicts import CONFLICT_RULES, validate_conflict_rules

    rules = [
        rule
        for rule in CONFLICT_RULES
        if book_name is None or rule.book_name == book_name
    ]
    # One fetch per book, because a rule pins several ids in the same collection.
    resolved: dict[str, dict[str, str]] = {}

    def lookup(book: str, chunk_id: str) -> str | None:
        if book not in resolved:
            wanted = [
                pin.chunk_id
                for rule in rules
                if rule.book_name == book
                for pin in rule.pins()
            ]
            resolved[book] = fetch_indexed_chunks(db_path, book, wanted)
        return resolved[book].get(chunk_id)

    problems = validate_conflict_rules(lookup, book_name=book_name)
    return problems, sum(len(list(rule.pins())) for rule in rules)


def _index_health(db_path: Path) -> dict[str, Any]:
    """Inspect what is actually indexed. Imports chromadb, so it stays out of doctor.py.

    `collect_diagnostics` is asserted to import no heavy runtime, and reporting
    `artifact:vector-db: ok` for a directory that exists says nothing about whether
    the collections inside it hold anything.
    """

    from rag_textbook_qa.indexing import list_indexed_books

    books = list_indexed_books(db_path)
    problems, pinned = _conflict_pin_problems(db_path)
    return {
        "db_path": str(db_path),
        "books": books,
        "empty_books": [book["book_name"] for book in books if not book["count"]],
        "conflict_pins": pinned,
        "conflict_problems": problems,
    }


def _render_index_health(report: dict[str, Any]) -> str:
    lines = ["", "向量库检查", "=" * 40, f"路径: {report['db_path']}"]
    if not report["books"]:
        lines.append("[WARNING ] 没有任何 textbook_* 集合，索引尚未构建")
    for book in report["books"]:
        status = "OK      " if book["count"] else "WARNING "
        model = book["embedding_model"] or "未知模型"
        lines.append(f"[{status}] {book['book_name']}: {book['count']} chunks（{model}）")

    pinned, problems = report["conflict_pins"], report["conflict_problems"]
    if not pinned:
        lines.append("[OK      ] 没有登记的教材冲突规则")
    elif problems:
        lines.append(
            f"[WARNING ] {pinned} 条冲突原文引用中有 {len(problems)} 条无法命中；"
            "详情运行 rag-qa index check"
        )
    else:
        lines.append(f"[OK      ] {pinned} 条冲突原文引用仍能在索引中找到")
    lines.append("=" * 40)
    return "\n".join(lines)


def _run_index(args: argparse.Namespace, settings: Settings) -> int:
    from rag_textbook_qa.catalog import book_id_from_chunk_stem
    from rag_textbook_qa.indexing import MultiBookVectorizer, list_indexed_books

    db_path = args.db_path or settings.paths.vector_db
    if args.index_command == "build":
        _load_project_environment(settings.paths.root / "project" / ".env")
        compute = ComputeSettings.from_env()
        vectorizer = MultiBookVectorizer(
            model_name=args.model,
            db_path=db_path,
            compute_settings=compute,
        )
        book_id = args.book or book_id_from_chunk_stem(args.input.stem)
        try:
            collection_name = vectorizer.vectorize_book(
                args.input,
                book_id,
                batch_size=args.batch_size,
                clear_existing=not args.append,
            )
        finally:
            vectorizer.close()
        print(f"索引已就绪: {collection_name}")
        # Re-chunking rewrites chunk ids, which is exactly when a pinned rule dies.
        problems, pinned = _conflict_pin_problems(db_path, book_id)
        if problems:
            print(
                f"警告: 本教材已登记的 {pinned} 条冲突原文引用中有 {len(problems)} 条无法命中；"
                "运行 rag-qa index check 查看详情。",
                file=sys.stderr,
            )
        return 0

    if args.index_command == "list":
        books = list_indexed_books(db_path)
        if args.json:
            print(json.dumps(books, ensure_ascii=False, indent=2))
        elif not books:
            print(f"尚无教材索引: {Path(db_path).expanduser().resolve()}")
        else:
            for book in books:
                model = book["embedding_model"] or "未知模型"
                print(f"{book['book_name']}: {book['count']} chunks ({model})")
        return 0

    if args.index_command == "check":
        problems, pinned = _conflict_pin_problems(db_path)
        if args.json:
            payload = {"pinned": pinned, "problems": problems}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif not problems:
            print(f"已登记冲突规则核对通过：{pinned} 条原文引用仍能在索引中找到。")
        else:
            print(f"已登记冲突规则已失效：{pinned} 条引用中有 {len(problems)} 条无法命中。")
            for problem in problems:
                reason = "片段不在索引中" if problem["status"] == "missing" else "原文已改变"
                print(f"  {problem['rule']} / {problem['chunk_id']}: {reason}")
            print("重新分块会改变片段ID；须按原文重新核实后再更新 rag/conflicts.py。")
        return 1 if problems else 0

    raise ValueError(f"未知 index 命令: {args.index_command}")


def _run_chat(args: argparse.Namespace, settings: Settings) -> int:
    from rag_textbook_qa.rag import interactive_main
    from rag_textbook_qa.rag.context import DEFAULT_CONTEXT_BUDGET

    _load_project_environment(settings.paths.root / "project" / ".env")
    interactive_main(
        workspace=settings.paths.root,
        db_path=args.db_path or settings.paths.vector_db,
        enable_llm=not args.no_llm,
        enable_reranker=not args.no_reranker,
        enable_hyde=not args.no_hyde,
        context_budget=args.context_budget or DEFAULT_CONTEXT_BUDGET,
    )
    return 0


def _run_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    if args.top_k <= 0:
        raise ValueError("--top-k 必须大于 0")

    _load_project_environment(settings.paths.root / "project" / ".env")
    from rag_textbook_qa.evaluation import (
        build_evaluation_plan,
        create_test_dataset,
        load_test_questions,
        render_evaluation_plan,
        run_evaluation,
    )

    questions_path = args.questions or (
        settings.paths.evaluation_data / "test_questions.json"
    )
    if args.questions is None and not questions_path.is_file():
        print(f"{questions_path} 不存在，使用内置测试集")
        questions = create_test_dataset()
    else:
        questions = load_test_questions(questions_path)
    if args.dry_run:
        plan = build_evaluation_plan(
            questions,
            output_dir=args.output_dir or settings.paths.evaluations,
            top_k=args.top_k,
            enable_hyde=args.hyde,
            include_baseline=args.baseline,
        )
        print(render_evaluation_plan(plan))
        return 0
    from rag_textbook_qa.rag import RAGEngine

    with RAGEngine(
        db_path=args.db_path or settings.paths.vector_db,
        enable_llm=True,
        verbose=False,
        enable_hyde=args.hyde,
    ) as engine:
        run_evaluation(
            engine,
            questions,
            args.output_dir or settings.paths.evaluations,
            include_baseline=args.baseline,
            top_k=args.top_k,
        )
    return 0


def _run_retrieval_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    _load_project_environment(settings.paths.root / "project" / ".env")
    from rag_textbook_qa.evaluation import (
        RETRIEVAL_STRATEGIES,
        load_retrieval_questions,
        run_retrieval_strategies,
        save_retrieval_report,
        select_split,
    )
    from rag_textbook_qa.rag import RAGEngine

    if args.top_k <= 0:
        raise ValueError("--top-k 必须大于 0")
    questions_path = args.questions or (
        settings.paths.evaluation_data / "retrieval_questions.json"
    )
    questions = select_split(load_retrieval_questions(questions_path), args.split)
    strategies = RETRIEVAL_STRATEGIES if args.strategy == "all" else (args.strategy,)

    # A benchmark must fail visibly instead of silently mixing remote and local results.
    compute = replace(
        ComputeSettings.from_env(),
        query_fallback_to_local=False,
    )
    with RAGEngine(
        db_path=args.db_path or settings.paths.vector_db,
        enable_llm=False,
        enable_reranker="hybrid-rerank" in strategies,
        enable_hyde=False,
        verbose=False,
        compute_settings=compute,
    ) as engine:
        report = run_retrieval_strategies(
            engine,
            questions,
            strategies,
            top_k=args.top_k,
            context_budget=args.context_budget or None,
        )

    report_path = save_retrieval_report(
        report,
        args.output_dir or settings.paths.evaluations / "retrieval",
    )
    print(f"检索评测完成：{args.split} 划分 {report['question_count']} 题，Top {report['top_k']}")
    for strategy, result in report["strategies"].items():
        line = (
            f"{strategy}: "
            f"Recall@{args.top_k}={result['mean_recall_at_k']:.3f}，"
            f"Hit@{args.top_k}={result['hit_rate_at_k']:.3f}，"
            f"MRR={result['mrr']:.3f}，"
            f"nDCG@{args.top_k}={result['mean_ndcg_at_k']:.3f}，"
            f"平均检索={result['mean_latency_seconds']:.3f} 秒"
        )
        retention = result.get("mean_context_retention")
        if retention is not None:
            line += (
                f"，证据保留={retention:.3f}"
                f"（丢弃 {result['relevant_dropped_total']} 条，"
                f"截断 {result['relevant_truncated_total']} 条）"
            )
        source_coverage = result.get("mean_source_evidence_coverage_at_k")
        if source_coverage is not None:
            line += f"，标注正文覆盖={source_coverage:.3f}"
            context_coverage = result.get("mean_source_evidence_context_coverage")
            if context_coverage is not None:
                line += f"，正文送达={context_coverage:.3f}"
        print(line)
    print(f"报告: {report_path}")
    return 0


def _run_generation_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    _load_project_environment(settings.paths.root / "project" / ".env")
    import hashlib
    from urllib.parse import urlsplit

    from rag_textbook_qa.evaluation.generation import load_generation_cases, parse_arm
    from rag_textbook_qa.evaluation.generation_runner import (
        llm_pair_from_env,
        openai_generator,
        openai_judge,
        run_generation_experiment,
    )

    for option, value in (
        ("--samples", args.samples),
        ("--concurrency", args.concurrency),
        ("--max-tokens", args.max_tokens),
    ):
        if value <= 0:
            raise ValueError(f"{option} 必须大于 0")
    arms = [parse_arm(spec) for spec in args.arm]
    cases = load_generation_cases(args.cases)
    generator_llm, judge_llm, extra = llm_pair_from_env()
    protocol = {
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "generator": {
            "host": urlsplit(generator_llm.base_url).hostname,
            "model": generator_llm.default_model,
            "max_tokens": args.max_tokens,
        },
        "judge": {
            "host": urlsplit(judge_llm.base_url).hostname,
            "model": judge_llm.default_model,
            "extra": extra,
        },
    }
    report = run_generation_experiment(
        cases,
        arms,
        output_dir=args.output_dir,
        generator=openai_generator(
            generator_llm.client, generator_llm.default_model, max_tokens=args.max_tokens
        ),
        judge=openai_judge(judge_llm.client, judge_llm.default_model, extra=extra),
        samples=args.samples,
        seed=args.seed,
        concurrency=args.concurrency,
        protocol=protocol,
    )

    def shown(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    print(
        f"生成评测：计划 {report['planned']} 份回答，"
        f"已生成 {report['generated']}，已评判 {report['judged']}"
    )
    for name, arm in report["arms"].items():
        print(
            f"{name}: 实质问题={shown(arm['problem_claims'])} 条/份，"
            f"实质问题比例={shown(arm['problem_rate'])}，严格口径={shown(arm['strict_rate'])}，"
            f"覆盖={shown(arm['coverage'])}，样本相似度={shown(arm['overlap'])}"
        )
    for name, comparison in report["comparisons"].items():
        primary = comparison["problem_claims"]
        if primary:
            low, high = primary["ci95"]
            print(
                f"{name} 相对 {report['reference_arm']}："
                f"实质问题条数差 {primary['mean_difference']:+.3f}"
                f"（95% CI {low:+.3f} ~ {high:+.3f}，p={primary['p_value']:.3f}，"
                f"{primary['cases']} 题）"
            )
    print(f"报告: {args.output_dir / 'report.json'}")
    return 0


def _require_app_dependencies(compute: ComputeSettings) -> None:
    required = {"streamlit": "ui"}
    if compute.backend == "local" or compute.query_fallback_to_local:
        required.update(
            {
                "sentence_transformers": "local-models",
                "torch": "local-models",
            }
        )

    missing = [
        extra
        for module, extra in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return

    extras = " ".join(f"--extra {extra}" for extra in dict.fromkeys(missing))
    raise RuntimeError(
        "缺少 Web 启动依赖；请按 README 配置当前 Conda 环境后运行: "
        f"uv sync --inexact {extras}"
    )


def _web_app_path() -> Path:
    return Path(__file__).resolve().parent / "web" / "app.py"


def _run_app(args: argparse.Namespace, settings: Settings) -> int:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port 必须在 1 到 65535 之间")
    if not args.host.strip():
        raise ValueError("--host 不能为空")

    env_path = settings.paths.root / "project" / ".env"
    _load_project_environment(env_path)
    environment = dict(os.environ)
    if args.backend:
        environment["RAG_QA_COMPUTE_BACKEND"] = args.backend
    if args.device:
        environment["RAG_QA_DEVICE"] = args.device
    environment["RAG_QA_HOME"] = str(settings.paths.root)

    compute = ComputeSettings.from_env(environment)
    _require_app_dependencies(compute)

    app_path = _web_app_path()
    if not app_path.is_file():
        raise RuntimeError(f"找不到 Streamlit 入口: {app_path}")

    if compute.backend == "remote":
        fallback = "local" if compute.query_fallback_to_local else "关闭"
        print(
            f"计算后端: remote ({compute.remote_url})；"
            f"查询回退: {fallback}"
        )
    else:
        print(f"计算后端: local；device: {compute.device}")
    print(f"工作区: {settings.paths.root}")
    print(f"Web 地址: http://{args.host}:{args.port}")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.address={args.host}",
        f"--server.port={args.port}",
        f"--server.headless={'true' if args.no_browser else 'false'}",
        "--browser.gatherUsageStats=false",
    ]
    completed = subprocess.run(
        command,
        cwd=settings.paths.root,
        env=environment,
        check=False,
    )
    return completed.returncode


def _validated_health_summary(
    payload: dict[str, Any],
    *,
    compute: ComputeSettings,
) -> dict[str, Any]:
    if payload.get("status") != "ok":
        raise ProviderProtocolError("远程 Worker /health 状态不是 ok")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ProviderProtocolError("远程 Worker 协议版本与客户端不一致")

    device = payload.get("device")
    models = payload.get("models")
    if not isinstance(device, str) or not device:
        raise ProviderProtocolError("远程 Worker /health 缺少 device")
    if not isinstance(models, dict):
        raise ProviderProtocolError("远程 Worker /health 缺少 models")

    expected = {
        "embedding": ModelIdentity(
            task="embedding",
            model=compute.embedding_model,
            normalized=True,
            query_instruction=DEFAULT_QUERY_INSTRUCTION,
        ),
        "reranker": ModelIdentity(task="reranker", model=compute.reranker_model),
    }
    model_names: dict[str, str] = {}
    for task, identity in expected.items():
        remote_identity = models.get(task)
        if not isinstance(remote_identity, dict):
            raise ModelMismatchError(f"远程 Worker 未提供 {task} 模型")
        remote_model = remote_identity.get("model")
        if remote_identity.get("fingerprint") != identity.fingerprint:
            raise ModelMismatchError(
                f"远程 Worker {task} 模型不一致："
                f"期望 {identity.model}，实际 {remote_model or '未知'}"
            )
        model_names[task] = str(remote_model)

    return {
        "remote_url": compute.remote_url,
        "http_status": 200,
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "device": device,
        "models": model_names,
        "token_configured": compute.remote_token is not None,
    }


def _run_worker_check(args: argparse.Namespace) -> int:
    from rag_textbook_qa.providers.remote import RemoteWorkerClient

    environment = dict(os.environ)
    environment["RAG_QA_COMPUTE_BACKEND"] = "remote"
    if args.url:
        environment["RAG_QA_REMOTE_URL"] = args.url
    compute = ComputeSettings.from_env(environment)
    if args.timeout is not None:
        if args.timeout <= 0:
            raise ProviderError("--timeout 必须大于 0")
        compute = replace(compute, remote_timeout_seconds=args.timeout)

    client = RemoteWorkerClient(
        compute.remote_url or "",
        token=compute.remote_token,
        timeout=compute.remote_timeout_seconds,
    )
    summary = _validated_health_summary(client.request("/health"), compute=compute)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("远程 Worker 健康检查通过")
        print(f"URL: {summary['remote_url']}")
        print(f"HTTP: {summary['http_status']}")
        print(f"device: {summary['device']}")
        print(f"embedding: {summary['models']['embedding']}")
        print(f"reranker: {summary['models']['reranker']}")
        print("token: 已配置" if summary["token_configured"] else "token: 未配置")
    return 0


def _run_serve(args: argparse.Namespace, settings: Settings) -> int:
    _load_project_environment(settings.paths.root / "project" / ".env")
    from rag_textbook_qa.api.app import run_api_server

    run_api_server(
        host=args.host,
        port=args.port,
        db_path=args.db_path or settings.paths.vector_db,
        public=args.public,
        warmup=not args.no_warmup,
    )
    return 0


def _run_feedback(args: argparse.Namespace, settings: Settings) -> int:
    from rag_textbook_qa.api.feedback import FeedbackStore, summarize_feedback

    database = args.database or settings.paths.artifacts / "product" / "feedback.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(f"还没有反馈数据库: {database}")
    store = FeedbackStore(database)
    if args.feedback_command == "summary":
        summary = summarize_feedback(store.records())
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            _print_feedback_summary(summary)
        return 0
    if args.feedback_command == "candidates":
        count = store.export_candidates(args.output, overwrite=args.force)
        print(f"已生成 {count} 条待人工审核候选: {args.output.expanduser().resolve()}")
        return 0
    if args.feedback_command == "export":
        count = store.export_jsonl(args.output, overwrite=args.force)
        print(f"已导出 {count} 条反馈: {args.output.expanduser().resolve()}")
        return 0
    raise ValueError(f"未知 feedback 命令: {args.feedback_command}")


def _print_feedback_summary(summary: dict[str, Any]) -> None:
    from rag_textbook_qa.catalog import BOOK_LABELS

    total = summary["total"]
    ratings = summary["ratings"]
    rate = summary["helpful_rate"]
    print(f"反馈总数: {total}")
    print(f"有帮助: {ratings['helpful']}")
    print(f"需要改进: {ratings['needs_improvement']}")
    print(f"好评率: {rate * 100:.1f}%" if rate is not None else "好评率: 暂无数据")

    reason_labels = {
        "not_answered": "没有回答问题",
        "irrelevant_sources": "检索资料不相关",
        "unsupported_answer": "答案与教材不一致",
        "incomplete": "回答不完整",
        "too_slow": "速度太慢",
        "other": "其他",
    }
    print("问题原因:")
    if summary["reasons"]:
        for reason, count in summary["reasons"].items():
            print(f"  {reason_labels[reason]}: {count}")
    else:
        print("  暂无负面反馈")

    print("需要改进的教材:")
    if summary["negative_by_book"]:
        for book_id, count in summary["negative_by_book"].items():
            label = "全部教材" if book_id == "all_books" else BOOK_LABELS.get(book_id, book_id)
            print(f"  {label}: {count}")
    else:
        print("  暂无负面反馈")

    latency = summary["latency_seconds"]
    if latency["samples"]:
        print(
            "总耗时: "
            f"平均 {latency['average']:.3f} 秒 · "
            f"P50 {latency['p50']:.3f} 秒 · "
            f"P95 {latency['p95']:.3f} 秒"
        )
    else:
        print("总耗时: 暂无数据")


def _run_worker(args: argparse.Namespace, settings: Settings) -> int:
    _load_project_environment(settings.paths.root / "project" / ".env")

    if args.worker_command == "check":
        return _run_worker_check(args)
    if args.worker_command != "serve":
        raise ValueError(f"未知 worker 命令: {args.worker_command}")

    from rag_textbook_qa.worker import run_worker_server

    compute = ComputeSettings.from_env()
    compute = replace(
        compute,
        embedding_model=args.embedding_model or compute.embedding_model,
        reranker_model=args.reranker_model or compute.reranker_model,
        device=args.device or compute.device,
    )
    run_worker_server(
        host=args.host,
        port=args.port,
        embedding_model=compute.embedding_model,
        reranker_model=compute.reranker_model,
        device=compute.device,
        token=compute.remote_token,
        warmup=args.warmup,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "doctor":
        try:
            settings = Settings.load(args.workspace)
        except WorkspaceNotFoundError as exc:
            parser.error(str(exc))
        from dotenv import load_dotenv

        load_dotenv(settings.paths.root / "project" / ".env")
        index_report = _index_health(settings.paths.vector_db) if args.index else None
        if args.json:
            payload = diagnostics_as_dict(settings)
            if index_report is not None:
                payload["index"] = index_report
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_diagnostics(settings))
            if index_report is not None:
                print(_render_index_health(index_report))
        return 0

    if args.command == "ingest":
        try:
            return _run_ingest(args)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    if args.command == "index":
        try:
            settings = Settings.load(args.workspace)
            return _run_index(args, settings)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    if args.command == "chat":
        try:
            settings = Settings.load(args.workspace)
            return _run_chat(args, settings)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    if args.command == "evaluate":
        try:
            settings = Settings.load(args.workspace)
            return _run_evaluate(args, settings)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    if args.command == "evaluate-retrieval":
        try:
            settings = Settings.load(args.workspace)
            return _run_retrieval_evaluate(args, settings)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    if args.command == "evaluate-generation":
        try:
            settings = Settings.load(args.workspace)
            return _run_generation_evaluate(args, settings)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    if args.command == "app":
        try:
            settings = Settings.load(args.workspace)
            return _run_app(args, settings)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    if args.command == "serve":
        try:
            settings = Settings.load(args.workspace)
            return _run_serve(args, settings)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    if args.command == "feedback":
        try:
            settings = Settings.load(args.workspace)
            return _run_feedback(args, settings)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    if args.command == "worker":
        try:
            settings = Settings.load(args.workspace)
            return _run_worker(args, settings)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    parser.error(f"未知命令: {args.command}")
    return 2
