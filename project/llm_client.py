"""Compatibility exports for the packaged OpenAI-compatible LLM client."""

from rag_textbook_qa.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMSettings,
    create_llm_client,
)

__all__ = [
    "LLMClient",
    "LLMConfigurationError",
    "LLMSettings",
    "create_llm_client",
]


if __name__ == "__main__":
    print("LLM 客户端已迁入 rag_textbook_qa.llm；请使用 project/test_llm_api.py 做联网测试。")
