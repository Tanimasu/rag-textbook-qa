"""Prepare unlabelled dev candidates from a pinned before/after retrieval report.

No models are loaded. Rankings remain in a separate manifest; the review template
has no route labels or inferred grades. This pool is not a new independent test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb

from rag_textbook_qa.config import Settings
from rag_textbook_qa.indexing.snapshot import TemporaryIndexDirectory, copy_index

ROUTES = ("all_before", "all_after")


def prepare_review(report_path: Path, db_path: Path, output: Path, *, seed: int = 17) -> dict[str, int]:
    if output.exists():
        raise ValueError("输出目录已存在，请换一个路径")
    raw = report_path.read_bytes()
    report = json.loads(raw)
    if report.get("complete") is not True or report.get("split") != "dev":
        raise ValueError("只接受已完成的 dev 对照报告")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("报告必须包含非空 cases")
    wanted: dict[str, set[str]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("报告缺少问题正文")
        for route in ROUTES:
            ranking = case.get("routes", {}).get(route, {}).get("ranking")
            if not isinstance(ranking, list) or not ranking:
                raise ValueError(f"报告缺少 {route} 排名")
            for row in ranking:
                if (not isinstance(row, list) or len(row) != 2
                        or not all(isinstance(value, str) and value.strip() for value in row)):
                    raise ValueError("排名必须由教材与片段编号对组成")
                wanted.setdefault(row[0], set()).add(row[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryIndexDirectory(prefix="crossbook-review-", dir=output.parent) as temporary:
        directory = Path(temporary)
        revision = copy_index(db_path, directory / "db")
        if revision != report.get("index_revision"):
            raise ValueError("当前索引与对照报告的版本不同，请重新生成对照报告")
        found = {}
        client = chromadb.PersistentClient(path=str(directory / "db"))
        try:
            for book, identifiers in sorted(wanted.items()):
                collection = client.get_collection("textbook_" + book)
                rows = collection.get(ids=sorted(identifiers), include=["documents", "metadatas"])
                for identifier, content, metadata in zip(rows["ids"], rows["documents"], rows["metadatas"], strict=True):
                    if content is None:
                        raise ValueError("索引中的候选没有正文")
                    found[(book, identifier)] = (content, metadata or {})
                if identifiers - {key[1] for key in found if key[0] == book}:
                    raise ValueError(f"{book} 存在已经缺失的候选")
        finally:
            client.close()
        randomizer = random.Random(seed)
        questions, provenance = [], []
        for index, case in enumerate(cases, 1):
            question_id = f"crossbook-dev-{index:03d}"
            pool = sorted({tuple(row) for route in ROUTES for row in case["routes"][route]["ranking"]})
            randomizer.shuffle(pool)
            candidates = []
            identifiers = {}
            for book, identifier in pool:
                candidate_id = hashlib.sha256(json.dumps([book, identifier]).encode()).hexdigest()
                identifiers[(book, identifier)] = candidate_id
                content, metadata = found[(book, identifier)]
                candidates.append({
                    "candidate_id": candidate_id, "book_name": book, "chunk_id": identifier,
                    "section": {field: metadata.get(field, "") for field in ("chapter", "section_h2", "section_h3", "section_h4")},
                    "content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "grade": None, "evidence_quotes": [], "rationale": "",
                })
            questions.append({"question_id": question_id, "question": case["question"], "candidates": candidates})
            provenance.append({"question_id": question_id, "rankings": {
                route: [identifiers[tuple(row)] for row in case["routes"][route]["ranking"]]
                for route in ROUTES
            }})
        template: dict[str, Any] = {
            "schema_version": 1, "review_status": "pending", "reviewer": None,
            "reviewed_at_utc": None, "split": "dev",
            "scope": "before/after top-k candidate union; not an independent cross-book test set",
            "grades": {"3": "直接提供回答所需的核心证据", "2": "提供部分有效证据",
                       "1": "只有相关背景，不能回答问题", "0": "无关或误导"},
            "instructions": "仅依据候选正文与问题打分，并给出原文证据及理由。不要把未入池片段当作零分。",
            "questions": questions,
        }
        manifest = {"prepared_at_utc": datetime.now(UTC).isoformat(), "seed": seed,
                    "source_report_sha256": hashlib.sha256(raw).hexdigest(), "index_revision": revision,
                    "source_dataset_sha256": report.get("dataset_sha256"), "routes": list(ROUTES),
                    "questions": provenance}
        prepared = directory / "export"
        prepared.mkdir()
        for name, payload in (("review.json", template), ("manifest.json", manifest)):
            (prepared / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        prepared.rename(output)
    return {"questions": len(questions), "candidates": sum(len(question["candidates"]) for question in questions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    db = args.db_path or Settings.load().paths.vector_db
    print(json.dumps(prepare_review(args.report, db, args.output_dir, seed=args.seed), ensure_ascii=False))


if __name__ == "__main__":
    main()
