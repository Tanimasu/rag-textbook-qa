"""Repeat the Chroma paths that failed intermittently on a Windows CI runner.

Uses the offline regression fixtures. Failures are reported immediately, with no
retry that turns a failed acceptance into a passing one.
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
import tempfile
import time
import unittest
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from rag_textbook_qa.indexing import MultiBookVectorizer

TARGETS = (
    "test_index_revision.IndexRevisionTests.test_failed_replace_and_append_keep_the_published_revision",
    "test_index_revision.IndexRevisionTests.test_staging_is_invisible_and_a_publication_gap_is_retried",
)


def diagnose_copy_error(original):
    """Inspect an already failed read without changing the test's outcome."""

    def copy(source, target, batch_size):
        try:
            return original(source, target, batch_size)
        except Exception:
            diagnostics = {}
            try:
                db = Path(source._client._system.settings.persist_directory)
                diagnostics["source_collection"] = {"id": str(source.id), "name": source.name,
                                                     "model_dimension": getattr(source._model, "dimension", None)}
                diagnostics["native_files"] = {str(path.relative_to(db)): path.stat().st_size
                                               for path in db.glob("*/*") if path.is_file()}
                with closing(sqlite3.connect(f"{db.as_uri()}/chroma.sqlite3?mode=ro", uri=True)) as connection:
                    diagnostics["catalog"] = connection.execute("SELECT id,name,dimension FROM collections").fetchall()
                attempts = []
                for _ in range(2):
                    started = time.monotonic()
                    try:
                        rows = source.get(limit=1, include=["embeddings"])
                        attempts.append({"success": True, "rows": len(rows["ids"]),
                                         "seconds": time.monotonic() - started})
                    except Exception as error:  # noqa: BLE001 - keep the original failing outcome
                        attempts.append({"success": False, "error_type": type(error).__name__,
                                         "seconds": time.monotonic() - started})
                diagnostics["post_failure_reads"] = attempts
            except Exception as error:  # noqa: BLE001 - diagnostics must not replace the test error
                diagnostics["diagnostic_error_type"] = type(error).__name__
            # The fixtures redirect stdout/stderr while the writer is running.
            print("Chroma failed-read diagnostics: " + json.dumps(diagnostics),
                  file=sys.__stderr__, flush=True)
            raise
    return copy


def diagnose_cleanup_error(original):
    """Measure delayed native release while preserving the first failure."""

    def cleanup(temporary):
        try:
            return original(temporary)
        except PermissionError:
            started = time.monotonic()
            released = False
            for _ in range(20):
                time.sleep(0.01)
                try:
                    original(temporary)
                    released = True
                    break
                except PermissionError:
                    pass
            print("Chroma cleanup diagnostics: " + json.dumps({
                "released_after_failure": released,
                "seconds": time.monotonic() - started,
            }), file=sys.__stderr__, flush=True)
            raise
    return cleanup


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
    MultiBookVectorizer._copy_collection = staticmethod(diagnose_copy_error(MultiBookVectorizer._copy_collection))
    tempfile.TemporaryDirectory.cleanup = diagnose_cleanup_error(tempfile.TemporaryDirectory.cleanup)
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
