"""Validate reviewer-supplied ratings against an untouched candidate export.

Scores cover the exported candidate union only. They do not establish corpus
recall, independent test performance, or the identity of the declared reviewer.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_textbook_qa.evaluation.retrieval import ndcg_at_k

_RATING_FIELDS = {"grade", "evidence_quotes", "rationale"}
_REVIEW_FIELDS = {"questions", "review_status", "reviewer", "reviewed_at_utc"}
_SCOPE = "exported before/after candidate union on dev; unpooled candidates are unknown"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _indexed(rows: Any, key: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} 必须是非空数组")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{label} 缺少有效的 {key}")
        if row[key] in indexed:
            raise ValueError(f"{label} 的 {key} 重复")
        indexed[row[key]] = row
    return indexed


def _fixed(row: dict[str, Any], editable: set[str]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in editable}


def _same(left: Any, right: Any) -> bool:
    # Python equality treats True, 1 and 1.0 as equal; JSON field types are fixed.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 对象中存在重复字段: {key}")
        result[key] = value
    return result


def _load_validated(
    template_dir: Path, review_path: Path, *, require_complete: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    template_path, manifest_path = template_dir / "review.json", template_dir / "manifest.json"
    if require_complete and template_path.resolve() == review_path.resolve():
        raise ValueError("请评分原模板的副本，保留导出的 review.json 不变")
    raw = {name: path.read_bytes() for name, path in (
        ("template", template_path), ("manifest", manifest_path), ("review", review_path),
    )}
    template, manifest, review = (json.loads(raw[name], object_pairs_hook=_unique_object)
                                  for name in ("template", "manifest", "review"))
    if not all(isinstance(value, dict) for value in (template, manifest, review)):
        raise ValueError("模板、审计清单与评分文件必须是对象")
    if (type(template.get("schema_version")) is not int or template.get("schema_version") != 1 or template.get("split") != "dev"
            or template.get("review_status") != "pending" or template.get("reviewer") is not None
            or template.get("reviewed_at_utc") is not None):
        raise ValueError("需要未经评分的 schema_version=1 dev 原模板")
    if set(template) != set(review) or not _same(_fixed(template, _REVIEW_FIELDS), _fixed(review, _REVIEW_FIELDS)):
        raise ValueError("评分文件改变了模板的固定说明或评测范围")
    for field in ("source_report_sha256", "source_dataset_sha256", "index_revision"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(letter not in "0123456789abcdef" for letter in value):
            raise ValueError(f"审计清单缺少有效的 {field}")
    status = review.get("review_status")
    if not isinstance(status, str) or status not in {"pending", "in_progress", "complete"}:
        raise ValueError("review_status 必须是 pending、in_progress 或 complete")
    if require_complete and status != "complete":
        raise ValueError("评分尚未完成，不能生成质量对照报告")
    reviewer, reviewed_at = review.get("reviewer"), review.get("reviewed_at_utc")
    if reviewer is not None and (not isinstance(reviewer, str) or not reviewer.strip()):
        raise ValueError("reviewer 必须是非空字符串或 null")
    if reviewed_at is not None:
        try:
            timestamp = datetime.fromisoformat(reviewed_at)
            if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("reviewed_at_utc 必须是带 UTC 时区的 ISO 日期") from exc
    if status == "complete" and (reviewer is None or reviewed_at is None):
        raise ValueError("完整评分必须填写 reviewer 和 reviewed_at_utc")
    originals = _indexed(template.get("questions"), "question_id", "原模板问题")
    reviewed = _indexed(review.get("questions"), "question_id", "评分问题")
    provenance = _indexed(manifest.get("questions"), "question_id", "审计清单问题")
    if originals.keys() != reviewed.keys() or originals.keys() != provenance.keys():
        raise ValueError("评分、模板与审计清单的问题集合不一致")
    routes = manifest.get("routes")
    if (not isinstance(routes, list) or not routes
            or not all(isinstance(route, str) and route.strip() for route in routes)
            or len(set(routes)) != len(routes)):
        raise ValueError("审计清单的 routes 必须是不同的非空字符串")
    total = scored = 0
    for question_id, original in originals.items():
        actual = reviewed[question_id]
        if not isinstance(original.get("question"), str) or not original["question"].strip():
            raise ValueError(f"{question_id} 缺少问题正文")
        if not _same(_fixed(original, {"candidates"}), _fixed(actual, {"candidates"})):
            raise ValueError(f"{question_id} 改变了问题正文或编号")
        candidates = _indexed(original.get("candidates"), "candidate_id", "原候选")
        ratings = _indexed(actual.get("candidates"), "candidate_id", "评分候选")
        if candidates.keys() != ratings.keys():
            raise ValueError(f"{question_id} 的候选集合不一致")
        rankings = provenance[question_id].get("rankings")
        if not isinstance(rankings, dict) or rankings.keys() != set(routes):
            raise ValueError(f"{question_id} 的排名路线不一致")
        pooled = set()
        for ranking in rankings.values():
            if (not isinstance(ranking, list) or not ranking
                    or not all(isinstance(value, str) for value in ranking)
                    or len(set(ranking)) != len(ranking) or not set(ranking) <= candidates.keys()):
                raise ValueError(f"{question_id} 的排名存在重复或未知候选")
            pooled.update(ranking)
        if pooled != candidates.keys():
            raise ValueError(f"{question_id} 的候选池与排名并集不一致")
        for candidate_id, candidate in candidates.items():
            rating = ratings[candidate_id]
            if (not _RATING_FIELDS <= candidate.keys()
                    or candidate.get("grade") is not None or candidate.get("evidence_quotes") != []
                    or candidate.get("rationale") != ""):
                raise ValueError("原模板已被评分，无法作为固定参照")
            if not _same(_fixed(candidate, _RATING_FIELDS), _fixed(rating, _RATING_FIELDS)):
                raise ValueError(f"{question_id}/{candidate_id} 改变了候选正文、来源或编号")
            book, chunk, content = (candidate.get(key) for key in ("book_name", "chunk_id", "content"))
            if (not all(isinstance(value, str) and value.strip() for value in (book, chunk, content))
                    or _sha256(json.dumps([book, chunk]).encode()) != candidate_id
                    or _sha256(content.encode()) != candidate.get("content_sha256")):
                raise ValueError(f"{question_id}/{candidate_id} 的原候选身份或正文哈希不一致")
            if set(rating) != set(candidate):
                raise ValueError(f"{question_id}/{candidate_id} 的评分字段不完整")
            grade, quotes, rationale = (rating[key] for key in ("grade", "evidence_quotes", "rationale"))
            if grade is not None and (type(grade) is not int or grade not in range(4)):
                raise ValueError(f"{question_id}/{candidate_id} 的 grade 必须是 0 到 3 的整数或 null")
            if not isinstance(quotes, list) or not all(
                isinstance(quote, str) and quote.strip() and quote in content for quote in quotes
            ):
                raise ValueError(f"{question_id}/{candidate_id} 的证据引文必须逐字来自候选正文")
            if not isinstance(rationale, str) or (grade is not None and not rationale.strip()):
                raise ValueError(f"{question_id}/{candidate_id} 缺少评分理由")
            if grade is not None and grade > 0 and not quotes:
                raise ValueError(f"{question_id}/{candidate_id} 的非零评分缺少正文证据")
            total += 1
            scored += grade is not None
    if status == "complete" and total != scored:
        raise ValueError("完整评分中仍有未评分候选")
    metadata = {
        "review_status": status, "questions": len(originals), "candidates": total,
        "scored": scored, "unscored": total - scored, "reviewer": reviewer,
        "reviewed_at_utc": reviewed_at, "scope": _SCOPE,
        "reviewer_verification": "reviewer field is self-declared; identity and independence are not verified",
        "input_sha256": {name: _sha256(value) for name, value in raw.items()},
        "source_report_sha256": manifest.get("source_report_sha256"),
        "source_dataset_sha256": manifest.get("source_dataset_sha256"),
        "index_revision": manifest.get("index_revision"),
    }
    return metadata, template, manifest, review


def validate_review(template_dir: Path, review_path: Path) -> dict[str, Any]:
    """Report completeness without assigning grades or scoring pending reviews."""

    return _load_validated(template_dir, review_path, require_complete=False)[0]


def score_review(template_dir: Path, review_path: Path, *, top_k: int = 5) -> dict[str, Any]:
    """Score a complete reviewed copy against its exported candidate union only."""

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")
    metadata, template, manifest, review = _load_validated(template_dir, review_path, require_complete=True)
    reviewed = _indexed(review["questions"], "question_id", "评分问题")
    provenance = _indexed(manifest["questions"], "question_id", "审计清单问题")
    cases = []
    for original in template["questions"]:
        question_id = original["question_id"]
        ratings = _indexed(reviewed[question_id]["candidates"], "candidate_id", "评分候选")
        useful = {identifier for identifier, row in ratings.items() if row["grade"] >= 2}
        routes = {}
        for route, ranking in provenance[question_id]["rankings"].items():
            if top_k > len(ranking):
                raise ValueError("top_k 不能超过任一路线已记录排名的长度")
            selected = ranking[:top_k]
            grades = [ratings[identifier]["grade"] for identifier in selected]
            first = next((rank for rank, grade in enumerate(grades, 1) if grade >= 2), None)
            routes[route] = {
                "candidate_ids": selected, "grades": grades,
                "pooled_ndcg_at_k": ndcg_at_k(grades, [row["grade"] for row in ratings.values()], top_k),
                "useful_hit_at_k": first is not None,
                "useful_reciprocal_rank_at_k": 1 / first if first else 0.0,
                "useful_coverage_of_pool_at_k": len(set(selected) & useful) / len(useful) if useful else None,
            }
        cases.append({"question_id": question_id, "useful_candidates_in_pool": len(useful), "routes": routes})
    summary = {}
    for route in manifest["routes"]:
        rows = [case["routes"][route] for case in cases]
        coverage = [row["useful_coverage_of_pool_at_k"] for row in rows
                    if row["useful_coverage_of_pool_at_k"] is not None]
        summary[route] = {
            "questions": len(rows),
            "mean_pooled_ndcg_at_k": sum(row["pooled_ndcg_at_k"] for row in rows) / len(rows),
            "useful_hit_rate_at_k": sum(row["useful_hit_at_k"] for row in rows) / len(rows),
            "mean_useful_reciprocal_rank_at_k": sum(row["useful_reciprocal_rank_at_k"] for row in rows) / len(rows),
            "questions_with_useful_candidates_in_pool": len(coverage),
            "mean_useful_coverage_of_pool_at_k": sum(coverage) / len(coverage) if coverage else None,
        }
    report = {**metadata, "generated_at_utc": datetime.now(UTC).isoformat(),
              "top_k": top_k, "useful_minimum_grade": 2, "summary": summary, "cases": cases}
    if len(manifest["routes"]) == 2:
        before, after = manifest["routes"]
        differences = [case["routes"][after]["pooled_ndcg_at_k"]
                       - case["routes"][before]["pooled_ndcg_at_k"] for case in cases]
        report["comparison"] = {
            "before_route": before, "after_route": after,
            "mean_pooled_ndcg_difference": sum(differences) / len(differences),
            "pooled_ndcg_improved_questions": sum(value > 1e-12 for value in differences),
            "pooled_ndcg_worsened_questions": sum(value < -1e-12 for value in differences),
            "pooled_ndcg_unchanged_questions": sum(abs(value) <= 1e-12 for value in differences),
        }
    return report
