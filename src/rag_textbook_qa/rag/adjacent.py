"""Optional same-section neighbour expansion for generation context."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_HEADING_FIELDS = ("chapter", "section_h2", "section_h3", "section_h4")
_CHUNK_POSITION = re.compile(r"_p([0-9]+)$")


def _heading_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")).strip() for field in _HEADING_FIELDS)


def _same_named_section(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    key = _heading_key(left)
    return any(key) and key == _heading_key(right)


def load_ordered_chunks(collection: Any, book_name: str) -> list[dict[str, Any]]:
    """Read the indexed corpus in source order without a separate chunks artifact."""

    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError(f"{book_name} 的向量库记录不完整，无法补充相邻片段")

    positioned: list[tuple[int, dict[str, Any]]] = []
    seen_positions: set[int] = set()
    for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=True):
        match = _CHUNK_POSITION.search(str(chunk_id))
        if match is None:
            raise ValueError(f"{book_name} 的 chunk_id 缺少顺序编号: {chunk_id}")
        position = int(match[1])
        if position in seen_positions:
            raise ValueError(f"{book_name} 的 chunk_id 顺序编号重复: {position}")
        seen_positions.add(position)
        values = metadata or {}
        positioned.append(
            (
                position,
                {
                    **values,
                    "chunk_id": str(chunk_id),
                    "book_name": values.get("book_name") or book_name,
                    "content": str(document or ""),
                },
            )
        )
    return [row for _, row in sorted(positioned, key=lambda item: item[0])]


def append_same_section_neighbours(
    results: Sequence[dict[str, Any]],
    corpora: Mapping[str, Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Keep ranked results first, then add exact-section next/previous chunks."""

    positions = {
        book_name: {
            str(row.get("chunk_id", "")): index for index, row in enumerate(corpus)
        }
        for book_name, corpus in corpora.items()
    }
    seen = {str(result.get("chunk_id", "")) for result in results}
    expanded = list(results)
    for anchor in results:
        anchor_id = str(anchor.get("chunk_id", ""))
        book_name = str(anchor.get("book_name", ""))
        corpus = corpora.get(book_name)
        index = positions.get(book_name, {}).get(anchor_id)
        if corpus is None or index is None:
            continue
        for direction, offset in (("next", 1), ("previous", -1)):
            neighbour_index = index + offset
            if not 0 <= neighbour_index < len(corpus):
                continue
            neighbour = corpus[neighbour_index]
            neighbour_id = str(neighbour.get("chunk_id", ""))
            if not neighbour_id or neighbour_id in seen:
                continue
            if not _same_named_section(anchor, neighbour):
                continue
            expanded.append(
                {
                    **neighbour,
                    "rank": anchor.get("rank"),
                    "method": "same-section-adjacent",
                    "adjacent_of": anchor_id,
                    "adjacent_direction": direction,
                    "query_ids": list(anchor.get("query_ids", [])),
                }
            )
            seen.add(neighbour_id)
    return expanded
