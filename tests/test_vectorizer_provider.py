import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rag_textbook_qa.indexing import MultiBookVectorizer, list_indexed_books
from rag_textbook_qa.providers import ModelIdentity, TransientProviderError
from rag_textbook_qa.providers.base import DEFAULT_QUERY_INSTRUCTION


class FakeEmbeddingProvider:
    def __init__(self, model="fake-embedding", error=None, fail_on_call=None):
        self._identity = ModelIdentity(
            task="embedding",
            model=model,
            normalized=True,
            query_instruction=DEFAULT_QUERY_INSTRUCTION,
        )
        self.error = error
        self.fail_on_call = fail_on_call
        self.document_calls = 0
        self.query_calls = 0

    @property
    def identity(self):
        return self._identity

    def embed_documents(self, texts):
        self.document_calls += 1
        if self.error and (
            self.fail_on_call is None or self.document_calls >= self.fail_on_call
        ):
            raise self.error
        return [[1.0, float(index)] for index, _ in enumerate(texts)]

    def embed_queries(self, texts):
        self.query_calls += 1
        if self.error:
            raise self.error
        return [[1.0, 0.0] for _ in texts]


def _chunks():
    return [
        {
            "chunk_id": "chunk-1",
            "content": "进程是操作系统进行资源分配的基本单位。",
            "chapter": "第一章",
            "section_h2": "进程",
            "section_h3": "",
            "level": 2,
            "char_count": 22,
            "has_code": False,
            "has_image": False,
        },
        {
            "chunk_id": "chunk-2",
            "content": "线程是处理器调度的基本单位。",
            "chapter": "第一章",
            "section_h2": "线程",
            "section_h3": "",
            "level": 2,
            "char_count": 15,
            "has_code": False,
            "has_image": False,
        },
    ]


class VectorizerProviderTests(unittest.TestCase):
    def test_vectorization_records_fingerprint_and_uses_injected_provider(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks_path = root / "chunks.json"
            chunks_path.write_text(json.dumps(_chunks(), ensure_ascii=False), encoding="utf-8")
            provider = FakeEmbeddingProvider()

            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                vectorizer = MultiBookVectorizer(db_path=root / "db", embedding_provider=provider)
                vectorizer.vectorize_book(str(chunks_path), "os")
                vectorizer.search_book("os", "什么是进程？", top_k=1)

            collection = vectorizer.client.get_collection("textbook_os")
            self.assertEqual(collection.count(), 2)
            self.assertEqual(
                collection.metadata["embedding_fingerprint"], provider.identity.fingerprint
            )
            self.assertEqual(provider.document_calls, 1)
            self.assertEqual(provider.query_calls, 1)
            self.assertEqual(
                list_indexed_books(root / "db")[0]["collection_name"],
                "textbook_os",
            )

    def test_provider_failure_does_not_clear_existing_collection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks_path = root / "chunks.json"
            chunks_path.write_text(json.dumps(_chunks(), ensure_ascii=False), encoding="utf-8")

            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                vectorizer = MultiBookVectorizer(
                    db_path=root / "db", embedding_provider=FakeEmbeddingProvider()
                )
                vectorizer.vectorize_book(str(chunks_path), "os")
                failing = MultiBookVectorizer(
                    db_path=root / "db",
                    embedding_provider=FakeEmbeddingProvider(
                        error=TransientProviderError("worker offline"),
                        fail_on_call=2,
                    ),
                )
                with self.assertRaises(TransientProviderError):
                    failing.vectorize_book(
                        str(chunks_path), "os", batch_size=1, clear_existing=True
                    )

            self.assertEqual(
                vectorizer.client.get_collection("textbook_os").count(),
                2,
            )
            self.assertFalse(
                any(
                    collection.name.startswith("ragbuild_")
                    for collection in vectorizer.client.list_collections()
                )
            )

    def test_invalid_chunks_fail_before_embedding_or_collection_creation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks = _chunks()
            chunks[1]["chunk_id"] = chunks[0]["chunk_id"]
            chunks_path = root / "chunks.json"
            chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            provider = FakeEmbeddingProvider()

            with contextlib.redirect_stdout(io.StringIO()):
                vectorizer = MultiBookVectorizer(
                    db_path=root / "db",
                    embedding_provider=provider,
                )
                with self.assertRaisesRegex(ValueError, "重复 chunk_id"):
                    vectorizer.vectorize_book(chunks_path, "os")

            self.assertEqual(provider.document_calls, 0)
            self.assertEqual(vectorizer.client.list_collections(), [])


if __name__ == "__main__":
    unittest.main()
