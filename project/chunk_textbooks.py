"""Compatibility entry point for the migrated textbook chunker.

New code should import :mod:`rag_textbook_qa.ingestion.chunker`. Keeping this
small wrapper means historical commands still work from a source checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rag_textbook_qa.ingestion.chunker import (  # noqa: E402,F401
    BatchChunkResult,
    SmartTextbookChunker,
    TextChunk,
    batch_chunk_cleaned,
    batch_chunk_markdown,
    chunk_markdown,
    chunk_single_file,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
