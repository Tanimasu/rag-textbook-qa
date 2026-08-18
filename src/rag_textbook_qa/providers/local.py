"""Lazy local sentence-transformers providers."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from rag_textbook_qa.providers.base import (
    DEFAULT_QUERY_INSTRUCTION,
    MissingOptionalDependencyError,
    ModelIdentity,
    validate_embeddings,
    validate_scores,
)


def _model_device(device: str) -> str | None:
    normalized = device.strip().lower()
    if normalized == "auto":
        return None
    indexed_cuda = normalized.startswith("cuda:") and normalized.removeprefix("cuda:").isdigit()
    if normalized not in {"cpu", "cuda", "mps"} and not indexed_cuda:
        raise ValueError("device 必须是 auto、cpu、cuda、cuda:N 或 mps")
    return normalized


class LocalEmbeddingProvider:
    """Embedding provider that loads SentenceTransformer on first use."""

    def __init__(self, model: str, *, device: str = "auto") -> None:
        self._identity = ModelIdentity(
            task="embedding",
            model=model,
            normalized=True,
            query_instruction=DEFAULT_QUERY_INSTRUCTION,
        )
        self.device = _model_device(device)
        self._model: Any | None = None
        self._lock = threading.RLock()

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def _load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise MissingOptionalDependencyError(
                            "本地 embedding 需要安装：uv sync --extra local-models"
                        ) from exc
                    options = {} if self.device is None else {"device": self.device}
                    self._model = SentenceTransformer(self.identity.model, **options)
        return self._model

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        with self._lock:
            embeddings = (
                self._load()
                .encode(
                    values,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                .tolist()
            )
        return validate_embeddings(embeddings, len(values))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [self.identity.query_instruction + text for text in texts]
        return self._encode(prefixed)


class LocalRerankerProvider:
    """CrossEncoder provider that loads its model on first use."""

    def __init__(self, model: str, *, device: str = "auto") -> None:
        self._identity = ModelIdentity(task="reranker", model=model)
        self.device = _model_device(device)
        self._model: Any | None = None
        self._lock = threading.RLock()

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def _load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import CrossEncoder
                    except ImportError as exc:
                        raise MissingOptionalDependencyError(
                            "本地 reranker 需要安装：uv sync --extra local-models"
                        ) from exc
                    options = {} if self.device is None else {"device": self.device}
                    self._model = CrossEncoder(self.identity.model, **options)
        return self._model

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        values = list(documents)
        if not values:
            return []
        pairs = [(query, document) for document in values]
        with self._lock:
            scores = self._load().predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return validate_scores(scores, len(values))
