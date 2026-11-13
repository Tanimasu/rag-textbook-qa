"""Immutable, exhaustive cosine search over an aligned collection snapshot."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ExactVectorIndex:
    """Keep normalized float32 vectors and resolve exact-score ties by chunk ID."""

    __slots__ = ("_chunk_ids", "_id_ranks", "_matrix")

    def __init__(self, embeddings: object, chunk_ids: Sequence[str]) -> None:
        self._chunk_ids = tuple(chunk_ids)
        if any(not isinstance(value, str) or not value for value in self.chunk_ids):
            raise ValueError("向量缓存的 chunk_id 必须是非空字符串")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("向量缓存包含重复 chunk_id")
        if not self.chunk_ids:
            if embeddings is not None and np.asarray(embeddings).size:
                raise ValueError("向量数与文档数不符")
            matrix = np.empty((0, 0), dtype=np.float32)
        else:
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    matrix = np.array(embeddings, dtype=np.float32, order="C", copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("向量缓存必须是数值矩阵") from exc
            if matrix.ndim != 2 or matrix.shape[0] != len(self.chunk_ids):
                raise ValueError("向量数与文档数不符，向量缓存必须是二维矩阵")
            if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
                raise ValueError("向量缓存含非有限值或维度为空")
            norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64, optimize=False))
            if np.any(norms == 0):
                raise ValueError("向量缓存包含零向量，无法计算余弦相似度")
            np.divide(matrix, norms[:, None], out=matrix)
        # Numeric tie keys avoid rebuilding a string array on every query.
        id_ranks = np.empty(len(self.chunk_ids), dtype=np.int64)
        for rank, position in enumerate(sorted(range(len(self.chunk_ids)), key=self.chunk_ids.__getitem__)):
            id_ranks[position] = rank
        matrix.setflags(write=False)
        id_ranks.setflags(write=False)
        self._matrix = matrix
        self._id_ranks = id_ranks

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self._chunk_ids

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def search(self, query_embeddings: object, top_k: int) -> list[tuple[int, float]]:
        """Return original row positions and scores without calling BLAS or Chroma."""
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        try:
            query = np.asarray(query_embeddings, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("查询向量必须是数值矩阵") from exc
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("语义检索每次必须提供一个查询向量")
        if not np.isfinite(query).all():
            raise ValueError("查询向量含非有限值")
        if not self.chunk_ids:
            return []
        if query.shape[1] != self.matrix.shape[1]:
            raise ValueError("查询向量维度与缓存矩阵不符")
        norm = np.sqrt(np.einsum("ij,ij->i", query, query, optimize=False))[0]
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("查询向量的 L2 范数无效，无法计算余弦相似度")
        query = query[0] / norm
        # Fixed-order float64 accumulation is independent of BLAS thread settings.
        scores = np.einsum("ij,j->i", self.matrix, query, dtype=np.float64, optimize=False)
        np.clip(scores, -1., 1., out=scores)
        order = np.lexsort((self._id_ranks, -scores))[:top_k]
        return [(int(position), float(scores[position])) for position in order]
