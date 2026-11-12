"""Provider construction from a single compute configuration."""

from __future__ import annotations

from rag_textbook_qa.providers.base import ProviderError
from rag_textbook_qa.providers.config import ComputeSettings
from rag_textbook_qa.providers.local import LocalEmbeddingProvider, LocalRerankerProvider
from rag_textbook_qa.providers.remote import (
    FallbackEmbeddingProvider,
    FallbackRerankerProvider,
    RemoteCooldown,
    RemoteEmbeddingProvider,
    RemoteRerankerProvider,
    RemoteWorkerClient,
)


def _remote_client(settings: ComputeSettings) -> RemoteWorkerClient:
    if settings.remote_url is None:
        raise ValueError("remote provider 缺少 remote_url")
    return RemoteWorkerClient(
        settings.remote_url,
        token=settings.remote_token,
        timeout=settings.remote_timeout_seconds,
        connect_timeout=settings.remote_connect_timeout_seconds,
    )


def create_remote_cooldown(settings: ComputeSettings) -> RemoteCooldown | None:
    """The Worker availability memory for query fallback, or None when it is unused.

    Embedding and reranking talk to the same Worker. Passing one instance to both
    factories lets a failure seen by either spare the other the same wait.
    """

    if settings.backend != "remote" or not settings.query_fallback_to_local:
        return None
    return RemoteCooldown(settings.remote_cooldown_seconds)


def create_embedding_provider(
    settings: ComputeSettings | None = None,
    *,
    allow_query_fallback: bool = False,
    cooldown: RemoteCooldown | None = None,
):
    settings = settings or ComputeSettings.from_env()
    local = LocalEmbeddingProvider(settings.embedding_model, device=settings.device)
    if settings.backend == "local":
        return local

    remote = RemoteEmbeddingProvider(_remote_client(settings), settings.embedding_model)
    if allow_query_fallback and settings.query_fallback_to_local:
        return FallbackEmbeddingProvider(
            remote, local, cooldown=cooldown or create_remote_cooldown(settings)
        )
    return remote


def create_reranker_provider(
    settings: ComputeSettings | None = None,
    *,
    allow_query_fallback: bool = False,
    cooldown: RemoteCooldown | None = None,
):
    settings = settings or ComputeSettings.from_env()
    if settings.backend == "remote" and (
        settings.reranker_max_length is not None or settings.reranker_dtype != "float32"
    ):
        raise ProviderError("远程 Worker 暂不支持重排长度或精度开关；请使用 local 后端")
    local = LocalRerankerProvider(settings.reranker_model, device=settings.device,
                                 batch_size=settings.reranker_batch_size,
                                 max_length=settings.reranker_max_length,
                                 dtype=settings.reranker_dtype)
    if settings.backend == "local":
        return local

    remote = RemoteRerankerProvider(_remote_client(settings), settings.reranker_model)
    if allow_query_fallback and settings.query_fallback_to_local:
        return FallbackRerankerProvider(
            remote, local, cooldown=cooldown or create_remote_cooldown(settings)
        )
    return remote
