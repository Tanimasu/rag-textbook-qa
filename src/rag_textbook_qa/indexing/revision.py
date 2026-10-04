"""Small, read-only catalog fingerprints for detecting published index changes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

INDEX_REVISION_KEY = "rag_index_revision"


class IndexPublicationInProgress(RuntimeError):
    """The old collection has been renamed but its replacement is not visible yet."""


def index_catalog(db_path: str | Path) -> list[tuple[str, str, str, str]]:
    """Read collection identities and write positions without reading chunk bodies.

    Collection UUIDs support indexes built before explicit revisions existed.
    The small per-segment write-position table also detects external additions,
    deletions and updates made directly through Chroma. Staging collections are
    excluded, so an unfinished or failed build does not publish a new revision.
    """

    sqlite_path = Path(db_path).expanduser().resolve() / "chroma.sqlite3"
    if not sqlite_path.exists():
        return []
    with sqlite3.connect(f"{sqlite_path.as_uri()}?mode=ro", uri=True) as connection:
        connection.execute("BEGIN")
        replacing = connection.execute(
            """
            SELECT 1 FROM collections AS b
            LEFT JOIN collection_metadata AS m ON m.collection_id = b.id AND m.key = 'book_name'
            WHERE substr(b.name, 1, 10) = 'ragbackup_'
              AND NOT EXISTS (SELECT 1 FROM collections AS c
                              WHERE c.name = 'textbook_' || COALESCE(m.str_value, substr(b.name, 44)))
            LIMIT 1
            """
        ).fetchone()
        if replacing:
            raise IndexPublicationInProgress("教材索引正在发布，请稍后重试")
        rows = connection.execute(
            """
            SELECT c.name, c.id, COALESCE(m.str_value, ''), COALESCE(p.seq_id, '')
            FROM collections AS c
            LEFT JOIN collection_metadata AS m
              ON m.collection_id = c.id AND m.key = ?
            LEFT JOIN segments AS s
              ON s.collection = c.id AND s.scope = 'METADATA'
            LEFT JOIN max_seq_id AS p ON p.segment_id = s.id
            WHERE substr(c.name, 1, 9) = 'textbook_'
            ORDER BY c.name, c.id, s.id
            """,
            (INDEX_REVISION_KEY,),
        ).fetchall()
    return [
        (name, collection_id, revision, position.hex() if isinstance(position, bytes) else str(position))
        for name, collection_id, revision, position in rows
    ]


def index_revision(db_path: str | Path) -> str:
    """Return a stable cache key for the currently published textbook catalog."""

    payload = json.dumps(index_catalog(db_path), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
