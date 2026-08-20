"""Local and remote model providers for embedding and reranking."""

from rag_textbook_qa.providers.base import (
    AuthenticationError,
    EmbeddingProvider,
    MissingOptionalDependencyError,
    ModelIdentity,
    ModelMismatchError,
    ProviderCall,
    ProviderError,
    ProviderProtocolError,
    ProviderTelemetry,
    RerankerProvider,
    TransientProviderError,
    provider_trace,
)
from rag_textbook_qa.providers.config import ComputeSettings
from rag_textbook_qa.providers.factory import (
    create_embedding_provider,
    create_reranker_provider,
)

__all__ = [
    "AuthenticationError",
    "ComputeSettings",
    "EmbeddingProvider",
    "MissingOptionalDependencyError",
    "ModelIdentity",
    "ModelMismatchError",
    "ProviderCall",
    "ProviderError",
    "ProviderProtocolError",
    "ProviderTelemetry",
    "RerankerProvider",
    "TransientProviderError",
    "create_embedding_provider",
    "create_reranker_provider",
    "provider_trace",
]
