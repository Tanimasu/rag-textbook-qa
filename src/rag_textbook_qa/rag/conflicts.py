"""Conservative guards for reviewed textbook inconsistencies, not semantic detection."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConflictPin:
    """One reviewed excerpt, identified by chunk id and a complete quote."""

    chunk_id: str
    quote: str


@dataclass(frozen=True)
class ConflictRule:
    """A reviewed disagreement to disclose when every side survives into the context."""

    id: str
    topic: str
    book_name: str
    sides: tuple[tuple[ConflictPin, ...], ...]

    def pins(self) -> Iterator[ConflictPin]:
        for side in self.sides:
            yield from side


# Match both the reviewed source identity and a complete, still-visible quote.
# Rebuilt/edited sources require review before adding new identities here.
CONFLICT_RULES: tuple[ConflictRule, ...] = (
    ConflictRule(
        id="database_unique_null_count",
        topic="UNIQUE／唯一索引允许的NULL数量",
        book_name="database",
        sides=(
            (ConflictPin("ch7_s7_3_p467", "唯一索引允许所在列包含多个NULL值。"),),
            (
                ConflictPin(
                    "ch5_p281",
                    "唯一码允许为空，但系统为保证其唯一性，最多只允许出现一个NULL值。",
                ),
                ConflictPin(
                    "ch5_p285",
                    "对于UNIQUE所约束的唯一码，则允许为NULL，但是只能有一个NULL值。",
                ),
            ),
        ),
    ),
)

# Resolve one (book, chunk id) against the live corpus; None when the id is gone.
ChunkLookup = Callable[[str, str], str | None]


def _match_side(
    sources: list[dict[str, Any]],
    rule: ConflictRule,
    alternatives: tuple[ConflictPin, ...],
) -> list[dict[str, Any]]:
    """Collect every packed source that carries one of this side's reviewed quotes."""

    matches = []
    for source in sources:
        if source.get("book_name") != rule.book_name:
            continue
        identifier = source.get("citation_id")
        if type(identifier) is not int or identifier < 1:
            continue
        for pin in alternatives:
            if source.get("chunk_id") != pin.chunk_id:
                continue
            if pin.quote not in source.get("content", ""):
                continue
            matches.append(
                {
                    "citation_id": identifier,
                    "chunk_id": pin.chunk_id,
                    "quote": pin.quote,
                }
            )
    return matches


def find_source_conflicts(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a known conflict only when both sides survive context selection."""

    conflicts = []
    for rule in CONFLICT_RULES:
        sides = []
        for alternatives in rule.sides:
            matches = _match_side(sources, rule, alternatives)
            if not matches:
                break
            sides.append(matches)
        else:
            conflicts.append({"id": rule.id, "topic": rule.topic, "sides": sides})
    return conflicts


def validate_conflict_rules(lookup: ChunkLookup) -> list[dict[str, str]]:
    """Report registered pins that no longer resolve against the indexed corpus.

    Chunk ids are chunking-dependent: re-chunking rewrites them, so a rule that
    matched yesterday can stop matching with no signal at all — the guard simply
    goes quiet and every answer looks normal. The pinned corpus is not committed,
    so no offline test can prove the rules are still live; callers resolve each
    pin against whatever corpus is actually indexed and get back the dead ones.
    """

    problems = []
    for rule in CONFLICT_RULES:
        for pin in rule.pins():
            content = lookup(rule.book_name, pin.chunk_id)
            if content is None:
                status = "missing"
            elif pin.quote not in content:
                status = "quote_changed"
            else:
                continue
            problems.append(
                {
                    "rule": rule.id,
                    "book_name": rule.book_name,
                    "chunk_id": pin.chunk_id,
                    "status": status,
                }
            )
    return problems


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
