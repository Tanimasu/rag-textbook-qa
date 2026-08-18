"""Structured quality checks for parsed Markdown and textbook chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def analyze_markdown(markdown_path: str | Path) -> dict[str, Any]:
    """Return the legacy Markdown statistics together with structured issues."""

    path = Path(markdown_path).expanduser().resolve()
    content = path.read_text(encoding="utf-8")
    stats = {
        "总字符数": len(content),
        "总行数": len(content.split("\n")),
        "二级标题（##）": content.count("\n## "),
        "三级标题（###）": content.count("\n### "),
        "图片占位符": content.count("<!-- image -->"),
        "代码块": content.count("```"),
        "空行数": content.count("\n\n"),
    }

    issues = []
    if stats["图片占位符"] > 0:
        issues.append(
            f"有 {stats['图片占位符']} 个图片占位符（<!-- image -->），需人工处理或过滤"
        )
    if stats["总字符数"] < 10000:
        issues.append("总字符数不足 10000，文档可能解析不完整")

    return {"path": str(path), "stats": stats, "issues": issues}


def render_markdown_report(report: dict[str, Any]) -> str:
    path = Path(report["path"])
    lines = ["=" * 50, f"Markdown 质量报告: {path.name}", "=" * 50]
    lines.extend(f"  {key}: {value}" for key, value in report["stats"].items())
    lines.append("")
    if report["issues"]:
        lines.append("发现问题:")
        lines.extend(f"  - {issue}" for issue in report["issues"])
    else:
        lines.append("未发现明显问题，文档质量良好。")
    lines.append("=" * 50)
    return "\n".join(lines)


def check_markdown_quality(markdown_path: str | Path) -> dict[str, int]:
    """Print the legacy report and return its statistics for compatibility."""

    report = analyze_markdown(markdown_path)
    print(render_markdown_report(report))
    return report["stats"]


def analyze_chunks(chunks_path: str | Path) -> dict[str, Any]:
    """Return the same chunk-quality decisions as the legacy checker."""

    path = Path(chunks_path).expanduser().resolve()
    chunks = json.loads(path.read_text(encoding="utf-8"))
    issues: dict[str, list[str]] = {
        "code_truncated": [],
        "has_line_numbers": [],
        "too_small": [],
        "too_large": [],
        "empty_code": [],
    }

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        char_count = chunk["char_count"]

        if chunk["has_code"] and (
            content.strip().startswith("```\n\n}")
            or content.strip().startswith("```\n\n)")
        ):
            issues["code_truncated"].append(chunk_id)

        if re.search(r"^\s*\d+\s+[a-zA-Z_]\w*\s*\(", content, re.MULTILINE):
            issues["has_line_numbers"].append(chunk_id)

        if char_count < 100:
            issues["too_small"].append(chunk_id)
        elif char_count > 2000:
            issues["too_large"].append(chunk_id)

        if chunk["has_code"]:
            code_content = re.search(r"```.*?```", content, re.DOTALL)
            if code_content and len(code_content.group(0)) < 50:
                issues["empty_code"].append(chunk_id)

    total = len(chunks)
    total_issues = sum(len(chunk_ids) for chunk_ids in issues.values())
    issue_rate = (total_issues / total * 100) if total else 0.0
    if issue_rate < 5:
        grade = "excellent"
    elif issue_rate < 15:
        grade = "good"
    else:
        grade = "poor"

    return {
        "path": str(path),
        "total": total,
        "issues": issues,
        "issue_counts": {name: len(chunk_ids) for name, chunk_ids in issues.items()},
        "total_issues": total_issues,
        "issue_rate": issue_rate,
        "grade": grade,
    }


def render_chunks_report(report: dict[str, Any]) -> str:
    lines = ["=" * 70, "🔍 分块质量检查", "=" * 70, f"\n总块数: {report['total']}\n"]
    lines.extend(["问题统计:", "-" * 70])
    for issue_type, chunk_ids in report["issues"].items():
        count = len(chunk_ids)
        percentage = (count / report["total"] * 100) if report["total"] else 0
        status = "✅" if percentage < 5 else "⚠️" if percentage < 15 else "❌"
        lines.append(f"{status} {issue_type:20s}: {count:4d} 个 ({percentage:5.1f}%)")
        if 0 < count <= 3:
            lines.extend(f"      - {chunk_id}" for chunk_id in chunk_ids)

    lines.extend(["\n" + "=" * 70, "📋 总体评估:", "-" * 70])
    messages = {
        "excellent": "✅ 质量优秀！问题率 < 5%，可以直接使用",
        "good": "⚠️  质量良好。问题率 < 15%，可以使用，建议关注特定问题块",
        "poor": "❌ 质量较差。问题率 >= 15%，建议修复后再使用",
    }
    lines.append(messages[report["grade"]])
    return "\n".join(lines)


def check_chunks_quality(chunks_path: str | Path) -> str:
    """Print the legacy report and return its grade for compatibility."""

    report = analyze_chunks(chunks_path)
    print(render_chunks_report(report))
    return report["grade"]
