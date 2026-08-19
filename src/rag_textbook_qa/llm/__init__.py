"""OpenAI-compatible LLM client without import-time environment side effects."""

from rag_textbook_qa.llm.client import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    LLMClient,
    LLMConfigurationError,
    LLMSettings,
    create_llm_client,
)

__all__ = [
    "DEFAULT_LLM_BASE_URL",
    "DEFAULT_LLM_MODEL",
    "LLMClient",
    "LLMConfigurationError",
    "LLMSettings",
    "create_llm_client",
]
