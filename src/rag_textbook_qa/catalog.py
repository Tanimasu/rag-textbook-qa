"""Stable textbook identifiers shared by indexing, retrieval, evaluation, and UI."""

import hashlib
import re
from types import MappingProxyType

BOOK_LABELS = MappingProxyType(
    {
        "os": "操作系统",
        "computer_organization": "计算机组成原理",
        "computer_network": "计算机网络",
        "data_structure": "数据结构",
        "database": "数据库原理及应用",
    }
)

DATASET_VARIANTS = ("docling", "mineru")

CHUNK_STEM_TO_BOOK_ID = MappingProxyType(
    {
        "操作系统": "os",
        "操作系统_mineru": "os_mineru",
        "计算机组成原理": "computer_organization",
        "计算机组成原理_mineru": "computer_organization_mineru",
        "计算机网络": "computer_network",
        "计算机网络_mineru": "computer_network_mineru",
        "数据结构": "data_structure",
        "数据结构_mineru": "data_structure_mineru",
        "数据库原理及应用教程": "database",
        "数据库原理及应用教程_mineru": "database_mineru",
    }
)


def book_id_from_chunk_stem(stem: str) -> str:
    """Map a chunk filename stem to a deterministic Chroma-safe book id."""

    normalized = stem.removesuffix("_chunks")
    known = CHUNK_STEM_TO_BOOK_ID.get(normalized)
    if known is not None:
        return known

    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", normalized)
    safe = re.sub(r"\.{2,}", "_", safe).strip("._-")
    if safe and safe == normalized and len(safe) <= 503:
        return safe
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    if safe:
        return f"{safe[:494].rstrip('._-')}_{digest}"
    return f"book_{digest}"
