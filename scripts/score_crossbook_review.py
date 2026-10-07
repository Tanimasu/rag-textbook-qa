"""Validate a reviewed export copy or score its fixed candidate union.

Keep the original export unchanged and edit a copy of review.json. Pending
ratings can be validated but cannot produce a quality comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_textbook_qa.evaluation.crossbook import score_review, validate_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and args.output is None:
        parser.error("生成对照报告必须指定 --output")
    if args.output is not None and args.output.exists():
        parser.error("输出文件已存在，请换一个路径")
    try:
        report = validate_review(args.template_dir, args.review) if args.validate_only else score_review(
            args.template_dir, args.review, top_k=args.top_k,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
