import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.indexing import MultiBookVectorizer
from rag_textbook_qa.providers import ModelIdentity
from rag_textbook_qa.providers.base import DEFAULT_QUERY_INSTRUCTION
from rag_textbook_qa.rag import RAGEngine


class FakeEmbeddingProvider:
    identity = ModelIdentity(
        task="embedding",
        model="fake-embedding",
        normalized=True,
        query_instruction=DEFAULT_QUERY_INSTRUCTION,
    )

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_queries(self, texts):
        return [[0.0, 1.0] for _ in texts]


class FakeRerankerProvider:
    identity = ModelIdentity(task="reranker", model="fake-reranker")

    def rerank(self, query, documents):
        return [float(index) for index, _ in enumerate(documents)]


class FakeLLMClient:
    def __init__(self):
        self.prompts = []

    def generate_answer(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return {
            "success": True,
            "answer": "测试回答",
            "model": "fake-llm",
            "tokens": {"prompt": 1, "completion": 1, "total": 2},
            "time": 0,
        }


def build_test_vector_db(root: Path) -> Path:
    chunks_path = root / "chunks.json"
    chunks_path.write_text(
        json.dumps(chunks(), ensure_ascii=False),
        encoding="utf-8",
    )
    with MultiBookVectorizer(
        db_path=root / "db",
        embedding_provider=FakeEmbeddingProvider(),
    ) as vectorizer:
        vectorizer.vectorize_book(chunks_path, "os")
    return root / "db"


def chunks():
    return [
        {
            "chunk_id": "chunk-1",
            "content": "进程是操作系统进行资源分配和管理的基本单位。" * 5,
            "chapter": "第一章",
            "section_h2": "进程",
            "section_h3": "",
            "level": 2,
            "char_count": 125,
            "has_code": False,
            "has_image": False,
        },
        {
            "chunk_id": "chunk-2",
            "content": "线程是处理器进行调度和执行的基本单位。" * 5,
            "chapter": "第一章",
            "section_h2": "线程",
            "section_h3": "",
            "level": 2,
            "char_count": 120,
            "has_code": False,
            "has_image": False,
        },
    ]


class RagProviderIntegrationTests(unittest.TestCase):
    def test_engine_accepts_injected_providers_without_model_runtime(self):
        before = {"sentence_transformers", "torch"}.intersection(sys.modules)
        with (
            patch.dict("os.environ", {"RAG_QA_COMPUTE_BACKEND": "invalid"}),
            tempfile.TemporaryDirectory() as temporary_directory,
            contextlib.redirect_stdout(io.StringIO()),
            RAGEngine(
                db_path=temporary_directory,
                enable_llm=False,
                embedding_provider=FakeEmbeddingProvider(),
                reranker_provider=FakeRerankerProvider(),
            ) as engine,
        ):
            results = engine._rerank(
                "query",
                [{"content": "first"}, {"content": "second"}],
                top_k=1,
            )

        after = {"sentence_transformers", "torch"}.intersection(sys.modules)
        self.assertEqual(after, before)
        self.assertEqual(results[0]["content"], "second")
        self.assertEqual(results[0]["rerank_score"], 1.0)

    def test_packaged_engine_retrieves_and_uses_injected_llm_without_network(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                db_path = build_test_vector_db(root)
                llm = FakeLLMClient()
                with RAGEngine(
                    db_path=db_path,
                    embedding_provider=FakeEmbeddingProvider(),
                    reranker_provider=FakeRerankerProvider(),
                    llm_client=llm,
                    enable_hyde=True,
                    verbose=False,
                ) as engine:
                    semantic = engine.search_embedding("os", "什么是线程？", top_k=1)
                    result = engine.ask("什么是线程？", book_name="os", top_k=1)

            self.assertEqual(len(semantic), 1)
            self.assertEqual(semantic[0]["method"], "embedding")
            self.assertTrue(result["success"])
            self.assertEqual(result["answer"], "测试回答")
            self.assertIn("相关教材内容", result["prompt"])
            self.assertEqual(len(llm.prompts), 3)

    def test_engine_reports_missing_llm_configuration_without_hiding_retrieval(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.dict("os.environ", {}, clear=True),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                db_path = build_test_vector_db(root)
                with RAGEngine(
                    db_path=db_path,
                    embedding_provider=FakeEmbeddingProvider(),
                    reranker_provider=FakeRerankerProvider(),
                    enable_hyde=False,
                    verbose=False,
                ) as engine:
                    result = engine.ask("什么是进程？", book_name="os", top_k=1)

            self.assertFalse(result["success"])
            self.assertIsNone(result["answer"])
            self.assertEqual(len(result["results"]), 1)
            self.assertIn("LLM 不可用", result["error"])
            self.assertIn("LLM_API_KEY", result["error"])


if __name__ == "__main__":
    unittest.main()
