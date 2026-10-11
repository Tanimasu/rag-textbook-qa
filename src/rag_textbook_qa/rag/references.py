"""Render answer chapter lists from the actual citation metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from rag_textbook_qa.catalog import BOOK_LABELS

_REFERENCE_TITLES = {"参考章节", "本次来源章节"}
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_CITATION = re.compile(r"【参考资料\s*(\d+)】")


def _split_reference_sections(answer: str) -> tuple[str, bool]:
    kept = []
    skipping = False
    found = False
    fence: str | None = None
    for line in answer.splitlines(keepends=True):
        if match := _FENCE.match(line):
            marker, rest = match.groups()
            if fence is None:
                if marker[0] == "~" or "`" not in rest:
                    fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not rest.strip():
                fence = None
            if not skipping:
                kept.append(line)
            continue
        if fence is None and (heading := _HEADING.match(line)):
            level, title = heading.groups()
            title = re.sub(r"\s+#+\s*$", "", title).rstrip(":：").strip()
            if len(level) <= 2:
                skipping = len(level) == 2 and title in _REFERENCE_TITLES
                found = found or skipping
        if not skipping:
            kept.append(line)
    return ("".join(kept).rstrip(), True) if found else (answer, False)


def answer_body(answer: str) -> str:
    """Keep answer prose and code, excluding generated chapter-list sections."""
    return _split_reference_sections(answer)[0]


def render_source_sections(answer: str, sources: Sequence[Mapping[str, Any]]) -> str:
    """Replace an existing chapter list, keeping the engine's raw answer untouched.

    Only sources referenced in the answer body are listed. This maps metadata;
    it does not establish that those sources support the generated claims.
    """
    body, found = _split_reference_sections(answer)
    if not found or not sources:
        return answer
    cited = {int(identifier) for identifier in _CITATION.findall(body)}
    by_id: dict[int, Mapping[str, Any]] = {}
    duplicate = set()
    for source in sources:
        identifier = source.get("citation_id")
        if type(identifier) is not int or identifier <= 0:
            continue
        if identifier in by_id:
            duplicate.add(identifier)
        else:
            by_id[identifier] = source
    lines = []
    for identifier in sorted(cited & by_id.keys() - duplicate):
        source = by_id[identifier]
        book_id = str(source.get("book_name") or "未知教材")
        book = BOOK_LABELS.get(book_id, book_id)
        section = " > ".join(str(source[field]).strip()
                             for field in ("chapter", "section_h2", "section_h3", "section_h4")
                             if source.get(field)) or "未标注章节"
        lines.append(f"- 【参考资料 {identifier}】{book}：{section}")
    if not lines:
        return body
    return body + "\n\n## 本次来源章节\n\n" + "\n".join(lines)
