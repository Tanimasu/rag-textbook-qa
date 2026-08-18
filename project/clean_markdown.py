"""Compatibility entry point for the migrated Markdown cleaner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rag_textbook_qa.ingestion.cleaner import clean_markdown  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用旧版 V4 规则清洗一个 Markdown 文件。",
    )
    parser.add_argument("input", type=Path, help="输入 Markdown 文件")
    parser.add_argument("output", type=Path, help="输出 Markdown 文件")
    parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.is_file():
        print(f"错误：找不到输入文件: {args.input}", file=sys.stderr)
        return 2

    try:
        if args.input.resolve() == args.output.resolve():
            print("错误：输入和输出不能是同一个文件。", file=sys.stderr)
            return 2

        clean_markdown(args.input, args.output, overwrite=args.force)
    except (OSError, ValueError) as error:
        print(f"错误：清洗失败: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
