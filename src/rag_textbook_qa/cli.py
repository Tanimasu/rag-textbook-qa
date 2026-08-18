"""Unified command-line entry point for the cross-platform package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rag_textbook_qa import __version__
from rag_textbook_qa.config import Settings, WorkspaceNotFoundError
from rag_textbook_qa.diagnostics.doctor import (
    diagnostics_as_dict,
    render_diagnostics,
)


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

    parser.error(f"未知命令: {args.command}")
    return 2
