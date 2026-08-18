"""Provider contracts shared by local models, remote clients, and the worker."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, runtime_checkable

DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
PROTOCOL_VERSION = "1"


class ProviderError(RuntimeError):
    """Base error for model providers."""


class TransientProviderError(ProviderError):
    """A timeout, connection failure, or temporary server error."""


class AuthenticationError(ProviderError):
    """The remote worker rejected or requires authentication."""


class ModelMismatchError(ProviderError):
    """The configured model does not match the model served remotely."""


class ProviderProtocolError(ProviderError):
    """The provider returned malformed or incompatible data."""


class MissingOptionalDependencyError(ProviderError):
    """A provider cannot start until its optional dependency group is installed."""


@dataclass(frozen=True)
class ModelIdentity:
    """Stable identity for data compatibility across local and remote runtimes."""

    task: Literal["embedding", "reranker"]
    model: str
    normalized: bool = False
    query_instruction: str = ""
    protocol_version: str = PROTOCOL_VERSION

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, str | bool]:
        return {**asdict(self), "fingerprint": self.fingerprint}


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class RerankerProvider(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]: ...


def validate_embeddings(
    embeddings: object,
    expected_count: int,
) -> list[list[float]]:
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        raise ProviderProtocolError(
            f"embedding 数量不正确：期望 {expected_count}，实际 "
            f"{len(embeddings) if isinstance(embeddings, list) else '非列表'}"
        )
    if not embeddings and expected_count == 0:
        return []

    converted: list[list[float]] = []
    dimension: int | None = None
    for row in embeddings:
        if not isinstance(row, list) or not row:
            raise ProviderProtocolError("embedding 必须是非空数值数组")
        try:
            vector = [float(value) for value in row]
        except (TypeError, ValueError) as exc:
            raise ProviderProtocolError("embedding 包含非数值元素") from exc
        if not all(math.isfinite(value) for value in vector):
            raise ProviderProtocolError("embedding 包含 NaN 或无穷值")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ProviderProtocolError("同一响应中的 embedding 维度不一致")
        converted.append(vector)
    return converted


def validate_scores(scores: object, expected_count: int) -> list[float]:
    if not isinstance(scores, list) or len(scores) != expected_count:
        raise ProviderProtocolError(
            f"rerank 分数数量不正确：期望 {expected_count}，实际 "
            f"{len(scores) if isinstance(scores, list) else '非列表'}"
        )
    try:
        converted = [float(score) for score in scores]
    except (TypeError, ValueError) as exc:
        raise ProviderProtocolError("rerank 响应包含非数值分数") from exc
    if not all(math.isfinite(score) for score in converted):
        raise ProviderProtocolError("rerank 响应包含 NaN 或无穷值")
    return converted
