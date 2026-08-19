"""Unified command-line entry point for the cross-platform package."""

from __future__ import annotations

import argparse
import json
import os
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

    worker = commands.add_parser("worker", help="运行远程 embedding/reranker Worker")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    serve = worker_commands.add_parser("serve", help="启动模型 Worker HTTP 服务")
    serve.add_argument("--host", default="127.0.0.1", help="监听 IP；远程时建议使用 Tailscale IP")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--embedding-model")
    serve.add_argument("--reranker-model")
    serve.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
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
        collection_name = vectorizer.vectorize_book(
            args.input,
            args.book or book_id_from_chunk_stem(args.input.stem),
            batch_size=args.batch_size,
            clear_existing=not args.append,
        )
        print(f"索引已就绪: {collection_name}")
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

    raise ValueError(f"未知 index 命令: {args.index_command}")


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
        if args.json:
            print(json.dumps(diagnostics_as_dict(settings), ensure_ascii=False, indent=2))
        else:
            print(render_diagnostics(settings))
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

    if args.command == "worker":
        try:
            settings = Settings.load(args.workspace)
            return _run_worker(args, settings)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"错误: {exc}\n")

    parser.error(f"未知命令: {args.command}")
    return 2
