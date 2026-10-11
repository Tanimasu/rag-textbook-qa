"""Offline agreement checks for saved claim extraction, review and judgments.

Agreement with supplied labels is not an estimate of expert accuracy. Reviewer
identity, independence and whether labels were frozen before judgment are not
established by this comparison.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag_textbook_qa.evaluation.generation import CLAIM_STATUSES, PROBLEM_STATUSES


def _index(rows: Any, field: str, label: str) -> dict[Any, dict[str, Any]]:
    if not isinstance(rows, list) or (not rows and field == "case_id"):
        raise ValueError(f"{label} 必须是数组，且题集不能为空")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"{label} 必须包含对象")
        key = row.get(field)
        if field == "case_id":
            valid = isinstance(key, str) and bool(key.strip())
        else:
            valid = type(key) is int
        if not valid or key in indexed:
            raise ValueError(f"{label} 中 {field} 无效或重复")
        indexed[key] = row
    return indexed


def compare_judge_review(
    extracted: Mapping[str, Any], labels: Mapping[str, Any], verified: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare every fact claim, refusing missing, extra or substituted rows."""
    if not isinstance(extracted, Mapping) or not isinstance(labels, Mapping):
        raise TypeError("陈述拆分与复核标签必须是对象")
    cases = _index(extracted.get("rows"), "case_id", "陈述拆分")
    judgments = _index(verified, "case_id", "已存判定")
    if set(cases) != set(labels) or cases.keys() != judgments.keys():
        raise ValueError("拆分、复核与判定的题号集合不一致")
    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    disagreements, missed, question_counts = [], 0, []
    for identifier, case in cases.items():
        claims = _index(case.get("claims"), "id", f"{identifier} 陈述")
        if any(row.get("type") not in {"fact", "meta"}
               or not isinstance(row.get("text"), str) or not row["text"].strip()
               for row in claims.values()):
            raise ValueError(f"{identifier} 陈述类型或正文无效")
        facts = {key: row for key, row in claims.items() if row["type"] == "fact"}
        reviewed = labels[identifier]
        if not isinstance(reviewed, Mapping):
            raise TypeError(f"{identifier} 复核标签必须是对象")
        given = reviewed.get("labels")
        if (not isinstance(given, Mapping) or set(given) != {str(key) for key in facts}
                or any(type(value) is not int or value not in (0, 1) for value in given.values())):
            raise ValueError(f"{identifier} 标签必须逐条覆盖事实陈述，且为整数 0 或 1")
        omissions = reviewed.get("missed")
        if not isinstance(omissions, list) or not all(isinstance(item, str) and item.strip() for item in omissions):
            raise ValueError(f"{identifier} missed 必须明确提供文本数组")
        score = judgments[identifier].get("score")
        if not isinstance(score, Mapping):
            raise TypeError(f"{identifier} 缺少评分对象")
        decisions = _index(score.get("claims"), "id", f"{identifier} 判定")
        if decisions.keys() != facts.keys():
            raise ValueError(f"{identifier} 判定必须逐条覆盖且仅包含事实陈述")
        judged_problems = 0
        for claim_id, fact in facts.items():
            decision = decisions[claim_id]
            status = decision.get("status")
            if status not in CLAIM_STATUSES or decision.get("text") != fact["text"]:
                raise ValueError(f"{identifier}/{claim_id} 判定标签无效或陈述正文不一致")
            judged, reviewed_problem = status in PROBLEM_STATUSES, given[str(claim_id)] == 1
            category = ("tp" if judged else "fn") if reviewed_problem else ("fp" if judged else "tn")
            confusion[category] += 1
            judged_problems += int(judged)
            if judged != reviewed_problem:
                disagreements.append({"case_id": identifier, "claim_id": claim_id,
                                      "text": fact["text"], "reviewed_problem": reviewed_problem,
                                      "judge_status": status})
        missed += len(omissions)
        question_counts.append({"case_id": identifier, "fact_claims": len(facts),
                                "reviewed_problem_claims": sum(given.values()),
                                "judged_problem_claims": judged_problems,
                                "missed_by_extraction": len(omissions)})
    tp, fp, fn, tn = (confusion[key] for key in ("tp", "fp", "fn", "tn"))
    total = tp + fp + fn + tn
    agreement = (tp + tn) / total if total else None
    chance = (((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / total**2) if total else None
    kappa = (agreement - chance) / (1 - chance) if chance is not None and chance < 1 else None
    return {
        "scope": "agreement with supplied historical labels; not independent human accuracy or current-judge validation",
        "reviewer_verification": "identity, independence and blind-label timing are not verified",
        "extraction_judge_version": extracted.get("judge_version"),
        "questions": len(cases), "fact_claims": total, "confusion": confusion,
        "agreement": agreement, "kappa": kappa,
        "judge_recall": tp / (tp + fn) if tp + fn else None,
        "judge_precision": tp / (tp + fp) if tp + fp else None,
        "missed_by_extraction": missed,
        "question_counts": question_counts, "disagreements": disagreements,
    }
