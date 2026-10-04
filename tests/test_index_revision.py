import contextlib
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from chromadb.api.models.Collection import Collection
from filelock import FileLock
from test_vectorizer_provider import FakeEmbeddingProvider, _chunks

from rag_textbook_qa.indexing import MultiBookVectorizer
from rag_textbook_qa.indexing.revision import IndexPublicationInProgress, index_revision
from rag_textbook_qa.providers import TransientProviderError
from rag_textbook_qa.rag import RAGEngine


class IndexRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.db = self.root / "db"
        self.chunks_path = self.root / "chunks.json"
        self.provider = FakeEmbeddingProvider()
        self.vectorizer = MultiBookVectorizer(db_path=self.db, embedding_provider=self.provider)
        self.addCleanup(self.vectorizer.close)
        self.write([{**row, "chunk_id": f"old_p{i}"} for i, row in enumerate(_chunks())])

    def write(self, chunks, book="os", **options):
        self.chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.vectorizer.vectorize_book(self.chunks_path, book, **options)

    def engine(self):
        with contextlib.redirect_stdout(io.StringIO()):
            engine = RAGEngine(db_path=self.db, embedding_provider=self.provider,
                               enable_reranker=False, enable_llm=False, verbose=False,
                               enable_adjacent_context=True)
        self.addCleanup(engine.close)
        return engine

    def test_refresh_replaces_text_headings_ids_and_neighbours_together(self):
        engine = self.engine()
        initial = engine.search_single_book("os", "进程", top_k=1)
        engine._context_candidates(initial, use_adjacent_context=True)
        old_revision = engine.index_revision
        newer = [{**row, "content": "新版本死锁正文" + str(i), "section_h2": "死锁",
                  "chunk_id": "new_p" + str(i)} for i, row in enumerate(_chunks())]
        self.write(newer)
        rows = engine.search_single_book("os", "死锁", top_k=2)
        self.assertNotEqual(engine.index_revision, old_revision)
        self.assertEqual({row["chunk_id"] for row in rows}, {"new_p0", "new_p1"})
        self.assertTrue(all(row["section_h2"] == "死锁" for row in rows))
        context = engine._context_candidates(rows[:1], use_adjacent_context=True)
        self.assertEqual({row["chunk_id"] for row in context}, {"new_p0", "new_p1"})
        self.assertTrue(all(row["content"].startswith("新版本") for row in context))
        with patch.object(engine, "_build_bm25_indexes") as rebuild:
            engine.refresh_index_if_changed()
        rebuild.assert_not_called()

    def test_book_addition_and_deletion_refresh_the_catalog(self):
        engine = self.engine()
        self.write(_chunks(), "database")
        self.assertEqual([b["book_name"] for b in engine.list_indexed_books()], ["database", "os"])
        self.vectorizer.client.delete_collection("textbook_os")
        self.assertEqual(engine.list_indexed_books(), [{"book_name": "database", "count": 2}])
        self.assertEqual(engine.search_bm25("os", "进程"), [])

    def test_catalog_closes_its_sqlite_handle_on_success_and_publication_gap(self):
        real_connect = sqlite3.connect
        connections = []

        def tracked_connect(*args, **kwargs):
            connection = real_connect(*args, **kwargs)
            connections.append(connection)
            return connection

        with patch("rag_textbook_qa.indexing.revision.sqlite3.connect", side_effect=tracked_connect):
            index_revision(self.db)
            old = self.vectorizer.client.get_collection("textbook_os")
            old.modify(name="ragbackup_test")
            with self.assertRaises(IndexPublicationInProgress):
                index_revision(self.db)
        self.assertEqual(len(connections), 2)
        for connection in connections:
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_catalog_does_not_wait_for_the_model_execution_lock(self):
        engine = self.engine()
        held, release, listed = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def hold_retrieval():
            with engine._index_lock:
                held.set()
                release.wait(2)

        def read_catalog():
            engine.list_indexed_books()
            listed.set()

        holder = threading.Thread(target=hold_retrieval)
        reader = threading.Thread(target=read_catalog)
        holder.start()
        try:
            self.assertTrue(held.wait(1))
            reader.start()
            self.assertTrue(listed.wait(1), "Catalog waited for the model execution lock")
        finally:
            release.set()
            holder.join(2)
            reader.join(2)

    def test_external_updates_to_legacy_indexes_invalidate_snapshot(self):
        collection = self.vectorizer.client.get_collection("textbook_os")
        metadata = {k: v for k, v in collection.metadata.items()
                    if k not in {"rag_index_revision", "hnsw:space"}}
        collection.modify(metadata=metadata)
        engine = self.engine()
        old_revision = engine.index_revision
        collection.update(ids=["old_p0"], documents=["外部更新后的正文"], embeddings=[[1., 0.]])
        rows = engine.search_bm25("os", "外部更新", top_k=2)
        self.assertNotEqual(engine.index_revision, old_revision)
        self.assertIn("外部更新后的正文", [row["content"] for row in rows])

    def test_failed_replace_and_append_keep_the_published_revision(self):
        for clear_existing in (True, False):
            with self.subTest(clear_existing=clear_existing):
                old_revision = index_revision(self.db)
                old_id = self.vectorizer.client.get_collection("textbook_os").id
                provider = FakeEmbeddingProvider(error=TransientProviderError("offline"),
                                                 fail_on_call=2)
                with (
                    MultiBookVectorizer(db_path=self.db, embedding_provider=provider) as failing,
                    self.assertRaises(TransientProviderError),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    failing.vectorize_book(self.chunks_path, "os", batch_size=1,
                                           clear_existing=clear_existing)
                self.assertEqual(index_revision(self.db), old_revision)
                self.assertEqual(self.vectorizer.client.get_collection("textbook_os").id, old_id)
                self.assertEqual([c.name for c in self.vectorizer.client.list_collections()],
                                 ["textbook_os"])

    def test_successful_append_preserves_old_vectors_and_publishes_new_revision(self):
        before = index_revision(self.db)
        old_vectors = self.vectorizer.client.get_collection("textbook_os").get(
            ids=["old_p0", "old_p1"], include=["embeddings"]
        )["embeddings"].tolist()
        new = [{**_chunks()[0], "chunk_id": "chunk-3", "content": "新增的死锁正文"}]
        calls_before = self.provider.document_calls
        self.write(new, clear_existing=False, batch_size=1)
        collection = self.vectorizer.client.get_collection("textbook_os")
        self.assertEqual(collection.count(), 3)
        self.assertEqual(self.provider.document_calls, calls_before + 1)
        self.assertNotEqual(index_revision(self.db), before)
        self.assertEqual(set(collection.get()["ids"]), {"old_p0", "old_p1", "chunk-3"})
        copied_vectors = collection.get(ids=["old_p0", "old_p1"], include=["embeddings"])
        for old, copied in zip(old_vectors, copied_vectors["embeddings"].tolist(), strict=True):
            for old_value, copied_value in zip(old, copied, strict=True):
                # Chroma re-normalizes copied cosine vectors in float32.
                self.assertAlmostEqual(old_value, copied_value, places=6)

    def test_staging_is_invisible_and_a_publication_gap_is_retried(self):
        before = index_revision(self.db)
        self.vectorizer.client.create_collection("ragbuild_unfinished")
        self.assertEqual(index_revision(self.db), before)
        engine = self.engine()
        old = self.vectorizer.client.get_collection("textbook_os")
        old.modify(name="ragbackup_test")
        with self.assertRaises(IndexPublicationInProgress):
            index_revision(self.db)
        with patch("rag_textbook_qa.rag.engine.time.sleep", side_effect=lambda _: old.modify(name="textbook_os")):
            self.assertEqual(engine.refresh_index_if_changed(), before)

    def test_mid_search_rebuild_retries_instead_of_combining_generations(self):
        engine = self.engine()
        original = engine.search_embedding
        calls = []

        def replace_after_embedding(*args, **kwargs):
            rows = original(*args, **kwargs)
            calls.append(None)
            if len(calls) == 1:
                self.write([{**row, "content": "替换正文" + str(i), "chunk_id": "new-" + str(i)}
                            for i, row in enumerate(_chunks())])
            return rows

        with patch.object(engine, "search_embedding", side_effect=replace_after_embedding):
            rows = engine.search_single_book("os", "替换正文", top_k=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual({row["chunk_id"] for row in rows}, {"new-0", "new-1"})

    def test_a_second_writer_is_refused_before_embedding_and_releases_after_failure(self):
        calls_before = self.provider.document_calls
        before = index_revision(self.db)
        with (
            FileLock(self.vectorizer._build_lock_path("os"), timeout=0),
            self.assertRaisesRegex(RuntimeError, "正在构建"),
        ):
            self.vectorizer.vectorize_book(self.chunks_path, "os", clear_existing=False)
        self.assertEqual(self.provider.document_calls, calls_before)
        self.assertEqual(index_revision(self.db), before)
        self.write(_chunks(), clear_existing=False)
        self.assertNotEqual(index_revision(self.db), before)

    def test_long_book_id_can_be_created_replaced_and_read(self):
        book = "a" * 503
        self.write(_chunks(), book)
        before = index_revision(self.db)
        old_id = self.vectorizer.client.get_collection("textbook_" + book).id
        self.write(_chunks(), book)
        self.assertNotEqual(index_revision(self.db), before)
        self.assertNotEqual(self.vectorizer.client.get_collection("textbook_" + book).id, old_id)
        self.assertEqual(self.vectorizer.client.get_collection("textbook_" + book).count(), 2)
        self.assertEqual(len(self.vectorizer.client.list_collections()), 2)

    def test_publication_failures_before_and_after_rename_restore_the_old_collection(self):
        for failure in ("old_rename", "new_rename", "new_rename_after_commit"):
            with self.subTest(failure=failure):
                before = index_revision(self.db)
                original_id = self.vectorizer.client.get_collection("textbook_os").id
                original_modify = Collection.modify

                def fail_rename(collection, *args, failure=failure, original_modify=original_modify, **kwargs):
                    target = kwargs.get("name", "")
                    if failure == "old_rename" and target.startswith("ragbackup_"):
                        raise RuntimeError("rename unavailable")
                    if target == "textbook_os" and collection.name.startswith("ragbuild_"):
                        if failure == "new_rename_after_commit":
                            original_modify(collection, *args, **kwargs)
                        if failure != "old_rename":
                            raise RuntimeError("rename unavailable")
                    return original_modify(collection, *args, **kwargs)

                with (
                    patch.object(Collection, "modify", fail_rename),
                    self.assertRaisesRegex(RuntimeError, "rename unavailable"),
                ):
                    self.write(_chunks())
                self.assertEqual(index_revision(self.db), before)
                self.assertEqual(self.vectorizer.client.get_collection("textbook_os").id, original_id)
                self.assertEqual([c.name for c in self.vectorizer.client.list_collections()], ["textbook_os"])


if __name__ == "__main__":
    unittest.main()
