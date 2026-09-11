import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_textbook_qa.indexing import MultiBookVectorizer
from rag_textbook_qa.providers import ModelIdentity, ProviderCall, ProviderTelemetry
from rag_textbook_qa.providers.base import DEFAULT_QUERY_INSTRUCTION
from rag_textbook_qa.rag import RAGEngine


class FakeEmbeddingProvider:
    identity = ModelIdentity(
        task="embedding",
        model="fake-embedding",
        normalized=True,
        query_instruction=DEFAULT_QUERY_INSTRUCTION,
    )

    def __init__(self):
        self.telemetry = ProviderTelemetry()

    def _record(self):
        self.telemetry.record(
            ProviderCall(
                task="embedding",
                backend="remote",
                model=self.identity.model,
                device="cuda",
                platform="Windows",
                elapsed_seconds=0.01,
                success=True,
            )
        )

    def embed_documents(self, texts):
        self._record()
        return [[1.0, 0.0] for _ in texts]

    def embed_queries(self, texts):
        self._record()
        return [[0.0, 1.0] for _ in texts]


class FakeRerankerProvider:
    identity = ModelIdentity(task="reranker", model="fake-reranker")

    def __init__(self):
        self.telemetry = ProviderTelemetry()

    def rerank(self, query, documents):
        self.telemetry.record(
            ProviderCall(
                task="reranker",
                backend="remote",
                model=self.identity.model,
                device="cuda",
                platform="Windows",
                elapsed_seconds=0.02,
                success=True,
            )
        )
        return [float(index) for index, _ in enumerate(documents)]


class FakeLLMClient:
    def __init__(self):
        self.prompts = []
        self.default_model = "fake-llm"

    def generate_answer(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return {
            "success": True,
            "answer": "测试回答",
            "model": "fake-llm",
            "tokens": {"prompt": 1, "completion": 1, "total": 2},
            "time": 0,
        }

    def stream_answer(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        yield "测试"
        yield "回答"


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
            execution = result["execution"]
            self.assertEqual(execution["embedding"]["backend"], "remote")
            self.assertEqual(execution["embedding"]["device"], "cuda")
            self.assertEqual(execution["embedding"]["platform"], "Windows")
            self.assertEqual(execution["reranker"]["backend"], "remote")
            self.assertNotIn("remote_url", execution)
            self.assertNotIn("token", repr(execution).lower())

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

    def test_hybrid_search_can_skip_configured_reranker(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                db_path = build_test_vector_db(root)
                reranker = FakeRerankerProvider()
                marker = reranker.telemetry.mark()
                with RAGEngine(
                    db_path=db_path,
                    embedding_provider=FakeEmbeddingProvider(),
                    reranker_provider=reranker,
                    enable_llm=False,
                    enable_hyde=False,
                    verbose=False,
                ) as engine:
                    results = engine.search_single_book(
                        "os",
                        "什么是进程？",
                        top_k=1,
                        use_hyde=False,
                        use_reranker=False,
                    )

            self.assertEqual(len(results), 1)
            self.assertEqual(reranker.telemetry.since(marker), [])

    def test_hybrid_search_uses_rank_fusion_deduplication_and_noise_filtering(self):
        engine = object.__new__(RAGEngine)
        engine.reranker = None
        engine.search_embedding = MagicMock(
            return_value=[
                {
                    "chunk_id": "semantic-correct",
                    "book_name": "os",
                    "content": "死锁是多个进程互相等待资源。",
                    "section_h2": "3.5 死锁概述",
                    "similarity": 0.9,
                },
                {
                    "chunk_id": "duplicate-correct",
                    "book_name": "os",
                    "content": "死锁是多个进程互相等待资源。",
                    "section_h2": "3.5 死锁概述",
                    "similarity": 0.89,
                },
                {
                    "chunk_id": "exercise",
                    "book_name": "os",
                    "content": "回答下面的死锁练习。",
                    "section_h2": "习题 3",
                    "similarity": 0.88,
                },
                {
                    "chunk_id": "semantic-only",
                    "book_name": "os",
                    "content": "进程同步的其他内容。",
                    "section_h2": "进程同步",
                    "similarity": 0.87,
                },
            ]
        )
        engine.search_bm25 = MagicMock(
            return_value=[
                {
                    "chunk_id": "exercise-copy",
                    "book_name": "os",
                    "content": "回答下面的死锁练习。",
                    "section_h3": "思考题",
                    "similarity": 8.0,
                },
                {
                    "chunk_id": "keyword-correct",
                    "book_name": "os",
                    "content": "死锁是多个进程互相等待资源。",
                    "section_h2": "3.5 死锁概述",
                    "similarity": 6.0,
                },
                {
                    "chunk_id": "keyword-only",
                    "book_name": "os",
                    "content": "资源分配的其他内容。",
                    "section_h2": "资源分配",
                    "similarity": 5.0,
                },
            ]
        )

        results = RAGEngine.search_single_book(
            engine,
            "os",
            "什么是死锁？",
            top_k=3,
            use_hyde=False,
            use_reranker=False,
        )

        engine.search_embedding.assert_called_once_with(
            "os",
            "什么是死锁？",
            9,
            use_hyde=False,
        )
        engine.search_bm25.assert_called_once_with("os", "什么是死锁？", 9)
        self.assertEqual(results[0]["section_h2"], "3.5 死锁概述")
        self.assertEqual(results[0]["source_methods"], ["embedding", "bm25"])
        self.assertEqual(results[0]["source_ranks"], {"embedding": 1, "bm25": 1})
        self.assertEqual(sum("练习" in result["content"] for result in results), 0)
        self.assertEqual(len({result["content"] for result in results}), len(results))

    def test_engine_streams_answer_chunks_and_preserves_execution_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                db_path = build_test_vector_db(root)
                llm = FakeLLMClient()
                chunks = []
                with RAGEngine(
                    db_path=db_path,
                    embedding_provider=FakeEmbeddingProvider(),
                    reranker_provider=FakeRerankerProvider(),
                    llm_client=llm,
                    enable_hyde=True,
                    verbose=False,
                ) as engine:
                    result = engine.ask(
                        "什么是进程？",
                        book_name="os",
                        top_k=1,
                        use_hyde=False,
                        on_answer_chunk=chunks.append,
                    )

            self.assertTrue(result["success"])
            self.assertEqual(chunks, ["测试", "回答"])
            self.assertEqual(result["answer"], "测试回答")
            self.assertTrue(result["llm_response"]["streamed"])
            self.assertEqual(len(llm.prompts), 1)
            self.assertGreaterEqual(result["execution"]["first_token_seconds"], 0)
            self.assertEqual(result["execution"]["embedding"]["backend"], "remote")


if __name__ == "__main__":
    unittest.main()
