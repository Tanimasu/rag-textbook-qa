"""Unified command-line entry point for the cross-platform package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        settings = Settings.load(args.workspace)
    except WorkspaceNotFoundError as exc:
        parser.error(str(exc))

    if args.command == "doctor":
        if args.json:
            print(json.dumps(diagnostics_as_dict(settings), ensure_ascii=False, indent=2))
        else:
            print(render_diagnostics(settings))
        return 0

    parser.error(f"未知命令: {args.command}")
    return 2
