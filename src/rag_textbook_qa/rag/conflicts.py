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


def conflict_prompt_note(conflicts: list[dict[str, Any]]) -> str:
    """Require the answer to disclose a reviewed disagreement instead of picking a side.

    Refusing to answer at all cost the reader everything else the evidence did
    support — the NULL-count dispute is one detail of a question about primary keys
    and unique indexes — so the note travels with the prompt and generation continues.
    """

    lines = ["以下资料之间存在已核实的表述冲突，回答时必须如实指出，不能只采用其中一种说法："]
    for conflict in conflicts:
        lines.append(f"关于{conflict['topic']}：")
        for side in conflict["sides"]:
            for evidence in side:
                lines.append(
                    f"- 原文：“{evidence['quote']}”【参考资料 {evidence['citation_id']}】"
                )
    lines.append(
        "请在回答中并列给出这两种说法及其资料编号，说明仅凭本次片段无法确定统一结论；"
        "问题的其余部分照常依据资料回答。"
    )
    return "\n".join(lines)
