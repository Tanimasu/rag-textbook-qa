"""Inspect saved dev budget rankings against a copied, version-matched index.

This loads no models and assigns no body relevance grades. Heading metrics are
proxies; exact source-span coverage is available only for existing annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb

from rag_textbook_qa.evaluation.retrieval import (
    load_retrieval_questions,
    score_source_evidence_coverage,
)
from rag_textbook_qa.indexing.snapshot import TemporaryIndexDirectory, copy_index


def audit(report_path: Path, dataset_path: Path, db_path: Path) -> dict[str, Any]:
    raw = report_path.read_bytes()
    source = json.loads(raw)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if (not isinstance(source, dict) or source.get("complete") is not True or source.get("split") != "dev"
            or source.get("dataset_sha256") != dataset_hash or source.get("top_k") != 5):
        raise ValueError("需要已完成、数据集哈希一致、top_k=5 的 dev 预算报告")
    questions = [question for question in load_retrieval_questions(dataset_path) if question.split == "dev"]
    cases = source.get("cases")
    if not isinstance(cases, list) or len(cases) != len(questions):
        raise ValueError("预算报告的问题集合与 dev 数据集不一致")
    wanted: dict[str, set[str]] = {}
    for case, question in zip(cases, questions, strict=True):
        if (not isinstance(case, dict)
                or (case.get("question"), case.get("book_name")) != (question.question, question.book_name)
                or not isinstance(case.get("budgets"), dict)):
            raise ValueError("预算报告的问题顺序、正文或教材不一致")
        for factor in ("2", "3"):
            route = case["budgets"].get(factor, {})
            if not isinstance(route, dict):
                raise TypeError("预算路线必须是对象")
            ranking = route.get("ranking")
            if (type(route.get("pairs")) is not int or route["pairs"] != int(factor) * 25
                    or not isinstance(ranking, list)
                    or len(ranking) != 5 or any(
                        not isinstance(row, list) or len(row) != 2
                        or not all(isinstance(value, str) and value.strip() for value in row)
                        for row in ranking
                    ) or len({tuple(row) for row in ranking}) != len(ranking)):
                raise ValueError("预算排名必须包含五个不同的教材与片段编号对")
            for book, identifier in ranking:
                wanted.setdefault(book, set()).add(identifier)
    found = {}
    with TemporaryIndexDirectory(prefix="candidate-body-audit-") as temporary:
        db = Path(temporary) / "db"
        revision = copy_index(db_path, db)
        if revision != source.get("index_revision"):
            raise ValueError("当前索引与预算报告版本不同")
        client = chromadb.PersistentClient(path=str(db))
        try:
            for book, identifiers in sorted(wanted.items()):
                rows = client.get_collection("textbook_" + book).get(
                    ids=sorted(identifiers), include=["documents", "metadatas"],
                )
                for identifier, content, metadata in zip(
                    rows["ids"], rows["documents"], rows["metadatas"], strict=True,
                ):
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("候选正文缺失")
                    found[(book, identifier)] = {
                        "book_name": book, "chunk_id": identifier, "content": content,
                        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                        "section": {field: (metadata or {}).get(field, "") for field in (
                            "chapter", "section_h2", "section_h3", "section_h4",
                        )},
                    }
                if identifiers - {identifier for found_book, identifier in found if found_book == book}:
                    raise ValueError("索引中存在缺失候选")
        finally:
            client.close()
    inspected = []
    for index, (case, question) in enumerate(zip(cases, questions, strict=True), 1):
        smaller = [tuple(row) for row in case["budgets"]["2"]["ranking"]]
        larger = [tuple(row) for row in case["budgets"]["3"]["ranking"]]
        coverage = None
        if question.evidence is not None:
            coverage = {str(budget): score_source_evidence_coverage(
                [found[key] for key in ranking], question.evidence, top_k=5,
            ) for budget, ranking in ((50, smaller), (75, larger))}
        inspected.append({
            "question_number": index, "question": question.question, "book_name": question.book_name,
            "ranking_unchanged": smaller == larger, "membership_unchanged": set(smaller) == set(larger),
            "lost_from_top5_at_50": [found[key] for key in larger if key not in smaller],
            "gained_in_top5_at_50": [found[key] for key in smaller if key not in larger],
            "rankings": {"50": [list(key) for key in smaller], "75": [list(key) for key in larger]},
            "shared_candidate_positions": [
                {"book_name": key[0], "chunk_id": key[1], "rank_at_50": smaller.index(key) + 1,
                 "rank_at_75": larger.index(key) + 1} for key in larger if key in smaller
            ],
            "annotated_source_coverage": coverage,
        })
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(), "complete": True, "split": "dev",
        "source_report_sha256": hashlib.sha256(raw).hexdigest(), "dataset_sha256": dataset_hash,
        "index_revision": revision, "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "saved 50/75 top-five union on dev; no models or new relevance grades",
        "body_evidence_policy": "existing reviewed source spans only; null means no span annotation",
        "questions": len(inspected),
        "ranking_unchanged": sum(row["ranking_unchanged"] for row in inspected),
        "membership_unchanged": sum(row["membership_unchanged"] for row in inspected),
        "lost_top5_occurrences": sum(len(row["lost_from_top5_at_50"]) for row in inspected),
        "gained_top5_occurrences": sum(len(row["gained_in_top5_at_50"]) for row in inspected),
        "questions_with_annotated_source_spans": sum(row["annotated_source_coverage"] is not None for row in inspected),
        "cases": inspected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出已存在，请换一个路径")
    try:
        report = audit(args.report, args.dataset, args.db_path)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
