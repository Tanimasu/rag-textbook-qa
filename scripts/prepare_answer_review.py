"""Export saved answers and pinned textbook evidence for offline review.

No generation, model loading, automatic grading or dotenv reads. A historical
answer export is not a quality acceptance result for the current checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 字段重复：{key}")
        result[key] = value
    return result


def _rows(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} 必须是非空对象数组")
    return rows


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空文本")
    return value


def prepare_review(
    questions_path: Path, answers_path: Path, workspace: Path, output: Path, *, run_label: str,
) -> dict[str, Any]:
    """Join by exact question text and validate all evidence before publication."""
    _text(run_label, "历史运行说明")
    if output.exists():
        raise ValueError("输出目录已存在，请使用新目录并保留原始复核材料")
    raw_questions, raw_answers = questions_path.read_bytes(), answers_path.read_bytes()
    questions, answers = _rows(raw_questions, "题集"), _rows(raw_answers, "已存回答")
    by_question = {}
    for answer in answers:
        question = _text(answer.get("question"), "已存回答的问题")
        if question in by_question:
            raise ValueError("已存回答的问题重复，无法确定对应样本")
        if not isinstance(answer.get("answer"), str):
            raise TypeError("已存回答的 answer 必须是文本")
        by_question[question] = answer

    root, seen_ids, seen_questions = workspace.resolve(), set(), set()
    source_cache: dict[Path, tuple[str, list[str]]] = {}
    rows, matched = [], set()
    for question in questions:
        identifier = _text(question.get("id"), "题号")
        text = _text(question.get("question"), "问题")
        book = _text(question.get("book_name"), "教材编号")
        truth = _text(question.get("ground_truth"), "答案要点")
        if identifier in seen_ids or text in seen_questions:
            raise ValueError("题集中的题号或问题重复")
        seen_ids.add(identifier)
        seen_questions.add(text)
        evidence = question.get("evidence")
        if not isinstance(evidence, dict):
            raise TypeError(f"{identifier} 缺少冻结教材证据")
        relative = Path(_text(evidence.get("path"), "证据路径"))
        source = (root / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(root):
            raise ValueError("教材证据必须位于工作区内")
        if source not in source_cache:
            content = source.read_bytes()
            source_cache[source] = _digest(content), content.decode("utf-8").splitlines()
        digest, lines = source_cache[source]
        if digest != evidence.get("sha256"):
            raise ValueError(f"{identifier} 教材证据哈希不匹配")
        start, end = evidence.get("start_line"), evidence.get("end_line")
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
            raise ValueError(f"{identifier} 教材证据行段无效")
        saved = by_question.get(text)
        if saved is not None:
            matched.add(text)
            if saved.get("question_id") not in (None, identifier):
                raise ValueError(f"{identifier} 已存回答的题号不同")
            if saved.get("ground_truth") != truth:
                raise ValueError(f"{identifier} 已存回答使用了不同的答案要点")
            if saved.get("book_name", book) != book:
                raise ValueError(f"{identifier} 已存回答的教材编号不同")
            contexts = saved.get("contexts")
            if contexts is not None and (
                not isinstance(contexts, list) or not all(isinstance(item, str) for item in contexts)
            ):
                raise ValueError(f"{identifier} 已存 contexts 必须是文本数组")
            actual_context = saved.get("context")
            if actual_context is not None and (
                not isinstance(actual_context, str)
                or (contexts is not None and "".join(contexts) != actual_context)
            ):
                raise ValueError(f"{identifier} 实际上下文与已存片段不一致")
            sources = saved.get("context_sources")
            if sources is not None:
                if (not isinstance(sources, list) or not all(isinstance(source, dict) for source in sources)
                        or any(not isinstance(source.get("context_text"), str) for source in sources)):
                    raise ValueError(f"{identifier} 引用来源缺少 context_text")
                if sources and (actual_context is None or "".join(source["context_text"] for source in sources) != actual_context):
                    raise ValueError(f"{identifier} 引用来源与实际上下文不一致")
            finish = saved.get("finish_reason")
            if finish is not None and not isinstance(finish, str):
                raise ValueError(f"{identifier} finish_reason 必须是文本或 null")
        else:
            contexts, finish, actual_context, sources = None, None, None, None
        answer = saved["answer"] if saved else None
        rows.append({
            "question_id": identifier, "question": text, "book_name": book,
            "split": question.get("split"), "ground_truth": truth,
            "answer": answer, "answer_sha256": _digest(answer.encode()) if answer is not None else None,
            "answer_status": "missing" if saved is None else ("nonempty" if answer.strip() else "empty"),
            "recorded_finish_reason": finish, "recorded_contexts": contexts,
            "recorded_context": actual_context, "recorded_context_sources": sources,
            "citation_ids_in_answer": sorted({int(value) for value in re.findall(r"【参考资料\s*(\d+)】", answer or "")}),
            "citation_verification": "pending; textbook evidence is not the recorded generation context",
            "evidence": dict(evidence), "evidence_excerpt": "\n".join(lines[start - 1:end]),
            "review": {"completeness": None, "grounding": None, "citation_accuracy": None,
                       "issues": [], "notes": ""},
        })
    review = {
        "schema_version": 1, "review_status": "pending", "reviewer": None,
        "reviewer_kind": None, "reviewed_at_utc": None, "source_run_label": run_label,
        "scope": "saved historical answers; not current-version acceptance or independent gold labels",
        "instructions": [
            "核对答案要点、实际回答和教材证据；所有判定初始为空。",
            "教材证据不等于生成时的检索上下文；缺失上下文时不要认定引用正确或无依据。",
            "没有 finish_reason 不能认定生成正常完成。",
            "保留原模板，另存复核副本；如由模型复核，明确填写 reviewer_kind=model。",
            "holdout 题仅用于验收和问题归类，不用于调参或修改评判规则。",
        ],
        "questions": rows,
    }
    summary = {
        "questions": len(rows), "matched_answers": len(matched),
        "missing_answers": sum(row["answer_status"] == "missing" for row in rows),
        "empty_answers": sum(row["answer_status"] == "empty" for row in rows),
        "missing_finish_reason": sum(row["recorded_finish_reason"] is None for row in rows),
        "missing_contexts": sum(not row["recorded_contexts"] for row in rows),
        "unmatched_saved_answers": len(by_question.keys() - matched),
    }
    rendered = json.dumps(review, ensure_ascii=False, indent=2) + "\n"
    manifest = {
        "schema_version": 1, "prepared_at_utc": datetime.now(UTC).isoformat(),
        "source_run_label": run_label, "summary": summary,
        "questions_sha256": _digest(raw_questions), "saved_answers_sha256": _digest(raw_answers),
        "review_template_sha256": _digest(rendered.encode()),
        "evidence_files": {str(path.relative_to(root)): digest for path, (digest, _) in source_cache.items()},
    }
    markdown = ["# 已存回答离线复核", "", f"来源说明：{run_label}", "",
                "历史材料；所有质量判定待复核，不代表当前版本通过验收。", ""]
    for row in rows:
        markdown.extend([
            f"## {row['question_id']} · {row['book_name']}", "", row["question"], "",
            "### 答案要点", "", row["ground_truth"], "", "### 历史回答", "",
            row["answer"] if row["answer"] is not None else "（缺少已存回答）", "",
            "### 冻结教材证据", "",
            f"{row['evidence']['path']}:{row['evidence']['start_line']}–{row['evidence']['end_line']}", "",
            row["evidence_excerpt"], "", "### 待核对", "",
            "- 答案要点是否完整回应？", "- 关键事实与适用范围是否有证据？",
            "- 引用是否有生成时的上下文可供核验？", "",
        ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".answer-review-", dir=output.parent) as temporary:
        prepared = Path(temporary) / "export"
        prepared.mkdir()
        (prepared / "review.json").write_text(rendered, encoding="utf-8")
        (prepared / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (prepared / "review.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
        if output.exists():
            raise ValueError("输出目录已存在")
        prepared.rename(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--saved-answers", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    try:
        summary = prepare_review(args.questions, args.saved_answers, args.workspace, args.output_dir,
                                 run_label=args.run_label)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
