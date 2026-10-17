"""Read a small window of indexed textbook prose without model inference."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any


def passage_window(
    rows: Sequence[dict[str, Any]], chunk_id: str, *, radius: int = 1,
) -> dict[str, Any]:
    """Return a bounded source window; never substitute a missing anchor."""
    if type(radius) is not int or not 0 <= radius <= 2:
        raise ValueError("阅读范围必须在 0–2 之间")
    position = next((i for i, row in enumerate(rows) if row["chunk_id"] == chunk_id), None)
    if position is None:
        raise LookupError("原文片段已不存在，可能是教材索引已经更新")
    anchor = rows[position]
    return {
        "anchor_chunk_id": chunk_id,
        "source_sha256": hashlib.sha256(str(anchor.get("content") or "").strip().encode()).hexdigest(),
        "previous_chunk_id": rows[position - 1]["chunk_id"] if position else None,
        "next_chunk_id": rows[position + 1]["chunk_id"] if position + 1 < len(rows) else None,
        "passages": [
            {**row, "is_anchor": row["chunk_id"] == chunk_id}
            for row in rows[max(0, position - radius):position + radius + 1]
        ],
    }
