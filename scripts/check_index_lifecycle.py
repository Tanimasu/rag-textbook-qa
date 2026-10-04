"""Repeat the Chroma paths that failed intermittently on a Windows CI runner.

Uses the offline regression fixtures. Failures are reported immediately, with no
retry that turns a failed acceptance into a passing one.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

TARGETS = (
    "test_index_revision.IndexRevisionTests.test_failed_replace_and_append_keep_the_published_revision",
    "test_index_revision.IndexRevisionTests.test_staging_is_invisible_and_a_publication_gap_is_retried",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations 必须大于 0")
    if args.output.exists():
        parser.error("输出文件已存在，请换一个路径")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    report = {"started_at_utc": datetime.now(UTC).isoformat(),
              "python": platform.python_version(), "platform": platform.system(),
              "iterations": args.iterations, "completed_iterations": 0,
              "targets": list(TARGETS), "complete": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.iterations):
        suite = unittest.defaultTestLoader.loadTestsFromNames(TARGETS)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        report["completed_iterations"] = index + 1
        report["success"] = result.wasSuccessful()
        report["complete"] = result.wasSuccessful() and index + 1 == args.iterations
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not result.wasSuccessful():
            raise SystemExit(1)
        print(f"Index lifecycle {index + 1}/{args.iterations} passed", flush=True)


if __name__ == "__main__":
    main()
