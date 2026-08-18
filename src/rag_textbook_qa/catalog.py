"""Stable textbook identifiers shared by indexing, retrieval, evaluation, and UI."""

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
