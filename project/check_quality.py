"""Compatibility wrapper for the packaged chunks quality checker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from rag_textbook_qa.ingestion.quality import check_chunks_quality  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 chunks JSON 质量")
    parser.add_argument("chunks", type=Path, help="待检查的 chunks JSON 文件")
    args = parser.parse_args()
    check_chunks_quality(args.chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
