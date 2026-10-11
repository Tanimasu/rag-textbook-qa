"""Recompute saved judge/reviewer agreement offline, without producing new labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag_textbook_qa.evaluation.judge_review import compare_judge_review


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 字段重复：{key}")
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出已存在，请使用新文件")
    try:
        raw = {name: path.read_bytes() for name, path in (
            ("extracted", args.extracted), ("labels", args.labels), ("verified", args.verified))}
        payload = {name: json.loads(value, object_pairs_hook=_unique_object) for name, value in raw.items()}
        report = compare_judge_review(payload["extracted"], payload["labels"], payload["verified"])
        report["input_sha256"] = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
        report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"question_counts", "disagreements"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
