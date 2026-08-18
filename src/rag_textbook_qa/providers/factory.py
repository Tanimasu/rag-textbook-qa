"""Provider construction from a single compute configuration."""

from __future__ import annotations

from rag_textbook_qa.providers.config import ComputeSettings
from rag_textbook_qa.providers.local import LocalEmbeddingProvider, LocalRerankerProvider
from rag_textbook_qa.providers.remote import (
    FallbackEmbeddingProvider,
    FallbackRerankerProvider,
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
    )


def create_embedding_provider(
    settings: ComputeSettings | None = None,
    *,
    allow_query_fallback: bool = False,
):
    settings = settings or ComputeSettings.from_env()
    local = LocalEmbeddingProvider(settings.embedding_model, device=settings.device)
    if settings.backend == "local":
        return local

    remote = RemoteEmbeddingProvider(_remote_client(settings), settings.embedding_model)
    if allow_query_fallback and settings.query_fallback_to_local:
        return FallbackEmbeddingProvider(remote, local)
    return remote


def create_reranker_provider(
    settings: ComputeSettings | None = None,
    *,
    allow_query_fallback: bool = False,
):
    settings = settings or ComputeSettings.from_env()
    local = LocalRerankerProvider(settings.reranker_model, device=settings.device)
    if settings.backend == "local":
        return local

    remote = RemoteRerankerProvider(_remote_client(settings), settings.reranker_model)
    if allow_query_fallback and settings.query_fallback_to_local:
        return FallbackRerankerProvider(remote, local)
    return remote
