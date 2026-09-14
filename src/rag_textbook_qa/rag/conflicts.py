"""Conservative guards for reviewed textbook inconsistencies, not semantic detection."""

from __future__ import annotations

from typing import Any

# Match both the reviewed source identity and a complete, still-visible quote.
# Rebuilt/edited sources require review before adding new identities here.
_UNIQUE_NULL_SIDES = (
    (("ch7_s7_3_p467", "唯一索引允许所在列包含多个NULL值。"),),
    (
        ("ch5_p281", "唯一码允许为空，但系统为保证其唯一性，最多只允许出现一个NULL值。"),
        ("ch5_p285", "对于UNIQUE所约束的唯一码，则允许为NULL，但是只能有一个NULL值。"),
    ),
)


def find_source_conflicts(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a known conflict only when both sides survive context selection."""
    sides = []
    for alternatives in _UNIQUE_NULL_SIDES:
        matches = []
        for source in sources:
            if source.get("book_name") != "database":
                continue
            identifier = source.get("citation_id")
            if type(identifier) is not int or identifier < 1:
                continue
            for chunk_id, quote in alternatives:
                if source.get("chunk_id") == chunk_id and quote in source.get("content", ""):
                    matches.append(
                        {"citation_id": identifier, "chunk_id": chunk_id, "quote": quote}
                    )
        if not matches:
            return []
        sides.append(matches)
    return [
        {
            "id": "database_unique_null_count",
            "topic": "UNIQUE／唯一索引允许的NULL数量",
            "sides": sides,
        }
    ]


def render_source_conflicts(conflicts: list[dict[str, Any]]) -> str:
    """Show attributed excerpts without letting a model resolve the disagreement."""
    blocks = ["本次检索到的教材片段存在已核实的表述冲突："]
    for conflict in conflicts:
        blocks.append(f"关于{conflict['topic']}：")
        for side in conflict["sides"]:
            for evidence in side:
                blocks.append(
                    f"- 原文：“{evidence['quote']}”【参考资料 {evidence['citation_id']}】"
                )
    blocks.append(
        "这些资料的说法不一致，仅凭本次片段无法确定适用条件或作出统一结论。"
        "请结合教材勘误和对应数据库版本的说明核实。本次仅报告分歧，未继续生成其余回答。"
    )
    return "\n\n".join(blocks)
