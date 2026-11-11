"""Offline claim/citation review packets. No semantic scores or model clients.

Reviewers split claims and assign labels. Validation binds those labels to exact
answer spans, original citation markers and frozen source quotes; it cannot
certify atomicity, preserved meaning, reviewer identity or semantic accuracy.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_textbook_qa.evaluation.generation import generation_cases_from_payload
from rag_textbook_qa.json_utils import loads_strict
from rag_textbook_qa.rag.references import answer_body

SUPPORT = {"supported", "partial", "unsupported", "contradicted", "insufficient_materials"}
_MARKER = re.compile(r"【参考资料([^】]*)】")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空文本")
    return value


def citation_markers(body: str) -> list[dict[str, Any]]:
    """Record original offsets, including malformed/unknown markers; skip no prose."""
    code_ranges, opening, offset = [], None, 0
    for line in body.splitlines(keepends=True):
        match = _FENCE.match(line)
        if match:
            marker, rest = match.groups()
            if opening is None:
                opening = (marker, offset)
            elif marker[0] == opening[0][0] and len(marker) >= len(opening[0]) and not rest.strip():
                code_ranges.append((opening[1], offset + len(line)))
                opening = None
        offset += len(line)
    if opening:
        code_ranges.append((opening[1], len(body)))
    code_ranges.extend((m.start(), m.end()) for m in re.finditer(r"(`+)[^`\n]*?\1", body))
    rows = []
    for match in _MARKER.finditer(body):
        raw = unicodedata.normalize("NFKC", match[1]).strip()
        raw = re.sub(r"\s*([-–—])\s*", r"\1", raw)
        identifiers, valid = [], bool(raw)
        for token in re.split(r"[,、，\s]+", raw):
            if not re.fullmatch(r"\d+(?:[-–—]\d+)?", token):
                valid = False
                continue
            limits = [int(s) for s in re.split(r"[-–—]", token)]
            first, last = limits[0], limits[-1]
            if not 1 <= first <= last <= 1000:
                valid = False
                continue
            identifiers.extend(range(first, last + 1))
        rows.append({"start": match.start(), "end": match.end(), "raw": match[0],
                     "citation_ids": sorted(set(identifiers)) if valid else [],
                     "syntax_valid": valid,
                     "in_code": any(left <= match.start() < right for left, right in code_ranges)})
    return rows


def prepare_packet(batch_dir: Path, *, run_label: str) -> dict[str, Any]:
    """Read a saved paragraph-selection batch; preserve planned but missing answers."""
    _text(run_label, "运行说明")
    inputs: dict[str, str] = {}

    def read(path: Path) -> Any:
        raw = path.read_bytes()
        inputs[str(path.resolve())] = digest(raw)
        return loads_strict(raw.decode("utf-8"))

    cases_payload = read(batch_dir / "generation-cases.json")
    cases = {c.case_id: c for c in generation_cases_from_payload(cases_payload)}
    requests = read(batch_dir / "generation-requests.json")
    if not isinstance(requests, list) or not requests:
        raise ValueError("请求必须是非空数组")
    responses = {}
    for path in sorted((batch_dir / "responses").glob("gen-*.json")):
        response = read(path)
        if not isinstance(response, dict) or response.get("label") != path.stem:
            raise ValueError("响应标签与文件名不一致")
        responses[path.stem] = response
    rows, seen_ordinals, seen_samples, matched = [], set(), set(), set()
    for job in requests:
        if not isinstance(job, Mapping):
            raise TypeError("请求记录必须是对象")
        ordinal, repeat = job.get("ordinal"), job.get("repeat")
        key = job.get("case_id"), job.get("arm"), repeat
        if (type(ordinal) is not int or ordinal < 0 or ordinal in seen_ordinals
                or type(repeat) is not int or repeat < 0 or key in seen_samples):
            raise ValueError("请求序号或样本键无效/重复")
        case = cases.get(key[0])
        if case is None or key[1] not in case.variants:
            raise ValueError("请求不属于冻结题目/方案")
        variant = case.variants[key[1]]
        request = job.get("request")
        if not isinstance(request, Mapping) or digest(encoded(request)) != job.get("request_sha256"):
            raise ValueError("实际请求哈希不一致")
        messages = request.get("messages")
        if (not isinstance(messages, list) or len(messages) != 1
                or not isinstance(messages[0], Mapping) or messages[0].get("role") != "user"):
            raise ValueError("仅支持本工具已保存的单条 user 提示词请求")
        prompt = _text(messages[0].get("content"), "实际提示词")
        if case.question not in prompt or variant.context not in prompt:
            raise ValueError("实际请求不含冻结问题/上下文")
        label = f"gen-{ordinal:02d}-{key[0]}-{key[1]}-{repeat}"
        if Path(label).name != label:
            raise ValueError("样本键不能含路径")
        response = responses.get(label)
        if response is not None:
            matched.add(label)
            if (response.get("role") != "generator"
                    or response.get("request_sha256") != job["request_sha256"]
                    or response.get("model") != request.get("model")):
                raise ValueError("响应与实际请求身份不一致")
            if not isinstance(response.get("content"), str):
                raise ValueError("已存回答正文必须是字符串")
        answer = response["content"] if response else None
        body = answer_body(answer) if answer is not None else None
        rows.append({
            "id": batch_dir.name + "/" + label, "case_id": case.case_id, "arm": key[1],
            "repeat": repeat, "question": case.question, "requirements": list(case.requirements),
            "request_sha256": job["request_sha256"], "prompt": prompt,
            "model_config": {k: v for k, v in request.items() if k in
                             {"model", "temperature", "max_tokens", "stream", "extra_body"}},
            "context": variant.context, "sources": list(variant.sources),
            "answer": answer, "answer_body": body,
            "answer_sha256": digest(answer.encode()) if answer is not None else None,
            "recorded_finish_reason": response.get("finish_reason") if response else None,
            "recorded_complete": response.get("complete") if response else None,
            "eligible_complete_answer": bool(response and response.get("complete") is True
                                             and response.get("finish_reason") == "stop" and answer.strip()),
            "citation_markers": citation_markers(body) if body is not None else [],
        })
        seen_ordinals.add(ordinal)
        seen_samples.add(key)
    return {"schema_version": 1, "run_label": run_label, "input_sha256": inputs,
            "scope": "saved development answers; no new generation, semantic labels or expert gold",
            "unmatched_generator_responses": sorted(set(responses) - matched), "records": rows}


def review_template(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "packet_sha256": digest(encoded(packet)),
            "review_status": "pending", "reviewer": None, "reviewer_kind": None,
            "reviewed_at_utc": None, "blind": False,
            "instructions": [
                "另存复核副本，不改 packet。手工拆主张，保留条件、数值、符号和适用范围。",
                "start/end 是 answer_body 的精确字符偏移，不能借用参考章节列表中的编号。",
                "citation_marker_offsets 只能引用同一原文段落中的实际标记；歧义写进 reason。",
                "上下文支持和指定引用支持分开判；每个原文摘录逐字复制并声明 source_id。",
                "删除或未展开的说法只记省略，不能补造不存在的主张或当作引用修复。",
                "结构校验不能认证拆分完整、语义正确、盲审或专家身份。模型复核填 model。",
            ],
            "records": [{"id": r["id"], "status": "pending", "extraction_complete": False,
                         "notes": "", "claims": []} for r in packet["records"]]}


def validate_review(packet: Mapping[str, Any], reviewed: Mapping[str, Any]) -> dict[str, Any]:
    """Validate review bindings and report missingness, never an acceptance pass."""
    if packet.get("schema_version") != 1 or reviewed.get("schema_version") != 1:
        raise ValueError("不支持的复核材料版本")
    if type(reviewed.get("blind")) is not bool:
        raise ValueError("必须明确记录盲审状态；本工具不认证盲审时序")
    if reviewed.get("packet_sha256") != digest(encoded(packet)):
        raise ValueError("复核副本与冻结 packet 不一致")
    if reviewed.get("review_status") not in {"partial", "complete"}:
        raise ValueError("未开始的复核不能生成结果")
    _text(reviewed.get("reviewer"), "复核者")
    if reviewed.get("reviewer_kind") not in {"model", "human"}:
        raise ValueError("必须记录 model/human 复核身份")
    try:
        timestamp = datetime.fromisoformat(reviewed["reviewed_at_utc"])
        if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("需要 UTC 时间")
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        raise ValueError("必须记录 UTC 复核时间") from e
    rows = reviewed.get("records")
    originals = {r["id"]: r for r in packet["records"]}
    if (not isinstance(rows, list) or any(not isinstance(r, Mapping) for r in rows)
            or len(rows) != len(originals) or {r.get("id") for r in rows} != set(originals)):
        raise ValueError("复核样本必须唯一、完整对应 packet")
    results, statuses, record_summaries = [], Counter(), []
    for row in rows:
        original = originals[row["id"]]
        status, claims = row.get("status"), row.get("claims")
        if status not in {"pending", "targeted", "complete"} or not isinstance(claims, list):
            raise ValueError("样本复核状态/陈述格式无效")
        if type(row.get("extraction_complete")) is not bool:
            raise ValueError("必须明确声明是否完整拆分")
        if (status == "pending" and claims) or (status == "complete" and (
                not claims or row["extraction_complete"] is not True)):
            raise ValueError("空白/未完整拆分不能标为完整复核")
        if reviewed["review_status"] == "complete" and status != "complete":
            raise ValueError("整批完整复核不能含待办/定向复核")
        statuses[status] += 1
        seen = set()
        sources = {s["citation_id"]: s["content"] for s in original["sources"]}
        markers = {m["start"]: m for m in original["citation_markers"]}
        body = original["answer_body"]
        record_start = len(results)
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise TypeError("陈述必须是对象")
            if set(claim) != {"id", "start", "end", "statement", "answer_qualifiers",
                              "citation_marker_offsets", "context_support", "citation_support",
                              "context_evidence", "citation_evidence", "reason"}:
                raise ValueError("陈述字段不完整或包含可替换原文/引用的额外字段")
            identifier = _text(claim.get("id"), "陈述编号")
            if identifier in seen:
                raise ValueError("陈述编号重复")
            seen.add(identifier)
            start, end = claim.get("start"), claim.get("end")
            if (not isinstance(body, str) or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(body) or not body[start:end].strip()
                    or "\n\n" in body[start:end]):
                raise ValueError("陈述必须绑定同一原文段落中的精确跨度")
            _text(claim.get("statement"), "独立主张")
            qualifiers = claim.get("answer_qualifiers")
            if not isinstance(qualifiers, list) or any(
                    not isinstance(q, str) or not q or q not in body[start:end] for q in qualifiers):
                raise ValueError("答案条件/符号摘录不属于该原文跨度")
            offsets = claim.get("citation_marker_offsets")
            if (not isinstance(offsets, list) or any(type(n) is not int for n in offsets)
                    or len(offsets) != len(set(offsets))):
                raise ValueError("引用标记偏移无效/重复")
            left = body.rfind("\n\n", 0, start) + 2 if "\n\n" in body[:start] else 0
            right = body.find("\n\n", end)
            right = len(body) if right < 0 else right
            cited = set()
            for offset in offsets:
                marker = markers.get(offset)
                if (marker is None or marker["in_code"] or not marker["syntax_valid"]
                        or not left <= offset < right):
                    raise ValueError("指定引用不是同段实际标记，不能从别段/章节/代码借用")
                cited.update(marker["citation_ids"])
            for field in ("context_support", "citation_support"):
                if claim.get(field) not in SUPPORT:
                    raise ValueError("上下文/指定引用必须分别给出有效标签")
            if claim["citation_support"] == "supported" and (not cited or not cited <= sources.keys()):
                raise ValueError("无引用或未知引用不能标为支持")
            if (claim["citation_support"] in {"supported", "partial"}
                    and claim["context_support"] not in {"supported", "partial"}):
                raise ValueError("指定来源有支持时上下文不能判为无支持")
            for name, allowed in (("context_evidence", set(sources)), ("citation_evidence", cited)):
                evidence = claim.get(name)
                if not isinstance(evidence, list):
                    raise TypeError("必须显式给出两类证据数组")
                for item in evidence:
                    if (not isinstance(item, Mapping) or type(item.get("source_id")) is not int
                            or item["source_id"] not in allowed or item["source_id"] not in sources
                            or not isinstance(item.get("quote"), str) or not item["quote"].strip()
                            or item["quote"] not in sources[item["source_id"]]):
                        raise ValueError("摘录不属于声明来源，或从非指定引用借用了支持")
                support = claim["context_support" if name == "context_evidence" else "citation_support"]
                if support in {"supported", "partial"} and not evidence:
                    raise ValueError("支持/部分支持标签必须附对应原文")
            _text(claim.get("reason"), "复核理由/条件与范围说明")
            results.append({"record_id": row["id"], "id": identifier, "answer_quote": body[start:end],
                            **dict(claim), "original_citation_ids": sorted(cited),
                            "eligible_complete_answer": original["eligible_complete_answer"]})
        selected = results[record_start:]
        record_summaries.append({
            "record_id": row["id"], "case_id": original["case_id"], "question": original["question"],
            "arm": original["arm"], "repeat": original["repeat"], "review_status": status,
            "eligible_complete_answer": original["eligible_complete_answer"],
            "reviewed_claims": len(selected),
            "context_support_counts": dict(Counter(r["context_support"] for r in selected)),
            "citation_support_counts": dict(Counter(r["citation_support"] for r in selected)),
        })
    return {"scope": "reviewer-supplied targeted labels; not model accuracy or independent acceptance",
            "reviewer": reviewed["reviewer"], "reviewer_kind": reviewed["reviewer_kind"],
            "identity_atomicity_semantics_independence_verified": False,
            "planned_answers": len(originals),
            "complete_generated_answers": sum(r["eligible_complete_answer"] for r in originals.values()),
            "missing_or_incomplete_answers": sum(not r["eligible_complete_answer"] for r in originals.values()),
            "review_status_counts": dict(statuses), "reviewed_claims": len(results),
            "context_support_counts": dict(Counter(r["context_support"] for r in results)),
            "citation_support_counts": dict(Counter(r["citation_support"] for r in results)),
            "per_answer": record_summaries, "claims": results}
