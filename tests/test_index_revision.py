import contextlib
import io
import json
import sqlite3
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from chromadb.api.models.Collection import Collection
from chromadb.api.rust import RustBindingsAPI
from filelock import FileLock
from test_vectorizer_provider import FakeEmbeddingProvider, _chunks

from rag_textbook_qa.indexing import MultiBookVectorizer
from rag_textbook_qa.indexing.revision import IndexPublicationInProgress, index_revision
from rag_textbook_qa.indexing.snapshot import TemporaryIndexDirectory
from rag_textbook_qa.providers import TransientProviderError
from rag_textbook_qa.rag import RAGEngine


class IndexRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryIndexDirectory()
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
        # Append inputs must have new identities; failed inference still exercises
        # copying the old vectors into staging before the second batch fails.
        self.chunks_path.write_text(json.dumps([
            {**chunk, "chunk_id": f"new_p{index}"} for index, chunk in enumerate(_chunks())
        ]), encoding="utf-8")
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

    def test_append_rejects_model_mismatch_before_embedding(self):
        provider = FakeEmbeddingProvider(model="other-embedding")
        before = index_revision(self.db)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            MultiBookVectorizer(db_path=self.db, embedding_provider=provider) as writer,
            self.assertRaisesRegex(ValueError, "模型与当前 Provider 不一致"),
        ):
            writer.vectorize_book(self.chunks_path, "os", clear_existing=False)
        self.assertEqual(provider.document_calls, 0)
        self.assertEqual(index_revision(self.db), before)

    def test_duplicate_append_is_refused_before_embedding_and_keeps_old_text(self):
        before = index_revision(self.db)
        calls_before = self.provider.document_calls
        self.chunks_path.write_text(json.dumps([
            {**_chunks()[0], "chunk_id": "old_p0", "content": "修改过的正文"}
        ]), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "已存在的 chunk_id"):
            self.vectorizer.vectorize_book(self.chunks_path, "os", clear_existing=False)
        self.assertEqual(self.provider.document_calls, calls_before)
        self.assertEqual(index_revision(self.db), before)
        stored = self.vectorizer.client.get_collection("textbook_os").get(ids=["old_p0"], include=["documents"])
        self.assertEqual(stored["documents"], [_chunks()[0]["content"]])

    def test_interrupted_publication_is_restored_even_if_new_inference_fails(self):
        before = index_revision(self.db)
        published = self.vectorizer.client.get_collection("textbook_os")
        published_id = published.id
        published.modify(name="ragbackup_interrupted")
        self.vectorizer.client.create_collection("ragbuild_abandoned", metadata={"book_name": "os"})
        provider = FakeEmbeddingProvider(error=TransientProviderError("offline"))
        with (
            contextlib.redirect_stdout(io.StringIO()),
            MultiBookVectorizer(db_path=self.db, embedding_provider=provider) as writer,
            self.assertRaises(TransientProviderError),
        ):
            writer.vectorize_book(self.chunks_path, "os")
        self.assertEqual(index_revision(self.db), before)
        self.assertEqual(self.vectorizer.client.get_collection("textbook_os").id, published_id)
        self.assertEqual(self.vectorizer.client.get_collection("ragbuild_abandoned").count(), 0)

    def test_append_after_interrupted_publication_preserves_previous_chunks(self):
        self.vectorizer.client.get_collection("textbook_os").modify(name="ragbackup_interrupted")
        calls_before = self.provider.document_calls
        self.write([{**_chunks()[0], "chunk_id": "new_p0"}], clear_existing=False)
        collection = self.vectorizer.client.get_collection("textbook_os")
        self.assertEqual(set(collection.get()["ids"]), {"old_p0", "old_p1", "new_p0"})
        self.assertEqual(self.provider.document_calls, calls_before + 1)

    def test_legacy_backup_without_book_metadata_is_restored_before_append(self):
        published = self.vectorizer.client.get_collection("textbook_os")
        metadata = {key: value for key, value in published.metadata.items()
                    if key not in {"book_name", "hnsw:space"}}
        published.modify(metadata=metadata)
        published.modify(name=f"ragbackup_{'0' * 32}_os")
        self.write([{**_chunks()[0], "chunk_id": "new_p0"}], clear_existing=False)
        collection = self.vectorizer.client.get_collection("textbook_os")
        self.assertEqual(set(collection.get()["ids"]), {"old_p0", "old_p1", "new_p0"})

    def test_clipped_legacy_backup_is_preserved_without_embedding(self):
        book = "b" * 500
        self.write(_chunks(), book)
        published = self.vectorizer.client.get_collection(f"textbook_{book}")
        published.modify(metadata={"description": "legacy"})
        backup_name = f"ragbackup_{'0' * 32}_{book[:469]}"
        published.modify(name=backup_name)
        calls_before = self.provider.document_calls
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "完整教材标识"):
            self.vectorizer.vectorize_book(self.chunks_path, book, clear_existing=False)
        self.assertEqual(self.provider.document_calls, calls_before)
        self.assertFalse(self.vectorizer._collection_exists(f"textbook_{book}"))
        self.assertEqual(self.vectorizer.client.get_collection(backup_name).count(), 2)

    def test_backup_of_legacy_collection_saves_full_book_identity_and_configuration(self):
        book = "b" * 500
        self.write(_chunks(), book)
        published = self.vectorizer.client.get_collection(f"textbook_{book}")
        data = published.get(include=["embeddings", "documents", "metadatas"])
        legacy_metadata = {key: value for key, value in published.metadata.items()
                           if key != "book_name"}
        self.vectorizer.client.delete_collection(f"textbook_{book}")
        # Build a real legacy collection with hnsw:space still in its metadata.
        legacy = self.vectorizer.client.create_collection(
            f"textbook_{book}", metadata=legacy_metadata,
            configuration={"hnsw": {"space": "cosine", "sync_threshold": 1}},
        )
        legacy.add(**{key: data[key] for key in ("ids", "documents", "metadatas", "embeddings")})
        original_get = self.vectorizer.client.get_collection
        original_modify = Collection.modify
        backups = []

        def record_backup(collection, *args, **kwargs):
            result = original_modify(collection, *args, **kwargs)
            if kwargs.get("name", "").startswith("ragbackup_"):
                backup = original_get(kwargs["name"])
                backups.append((backup.metadata, backup.configuration["hnsw"]["space"]))
            return result

        with patch.object(Collection, "modify", record_backup):
            self.write(_chunks(), book)
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0][0]["book_name"], book)
        self.assertEqual(backups[0][0]["embedding_model"], legacy_metadata["embedding_model"])
        self.assertEqual(backups[0][1], "cosine")

    def test_ambiguous_backups_are_preserved_without_embedding(self):
        self.vectorizer.client.get_collection("textbook_os").modify(name="ragbackup_first")
        self.vectorizer.client.create_collection("ragbackup_second", metadata={"book_name": "os"})
        calls_before = self.provider.document_calls
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "多个发布备份"):
            self.vectorizer.vectorize_book(self.chunks_path, "os")
        self.assertEqual(self.provider.document_calls, calls_before)
        self.assertEqual({collection.name for collection in self.vectorizer.client.list_collections()},
                         {"ragbackup_first", "ragbackup_second"})

    def test_backup_cleanup_failure_does_not_hide_a_successful_publication(self):
        for deleted_before_error in (False, True):
            with self.subTest(deleted_before_error=deleted_before_error):
                book = f"cleanup_{deleted_before_error}"
                self.write(_chunks(), book)
                original_id = self.vectorizer.client.get_collection("textbook_" + book).id
                original_delete = self.vectorizer.client.delete_collection

                def failed_cleanup(name, deleted_before_error=deleted_before_error,
                                   original_delete=original_delete, **kwargs):
                    if name.startswith("ragbackup_"):
                        if deleted_before_error:
                            original_delete(name, **kwargs)
                        raise RuntimeError("cleanup unavailable")
                    return original_delete(name, **kwargs)

                with patch.object(self.vectorizer.client, "delete_collection", side_effect=failed_cleanup):
                    self.write([{**_chunks()[0], "chunk_id": "new_p0", "content": "完整的新正文"}], book)
                published = self.vectorizer.client.get_collection("textbook_" + book)
                self.assertNotEqual(published.id, original_id)
                self.assertEqual(published.get()["documents"], ["完整的新正文"])
                self.assertTrue(index_revision(self.db))
                backups = [collection for collection in self.vectorizer.client.list_collections()
                           if collection.name.startswith("ragbackup_")
                           and collection.metadata["book_name"] == book]
                self.assertEqual(len(backups), 0 if deleted_before_error else 1)

    def test_legacy_model_name_blocks_incompatible_append_without_fingerprint(self):
        collection = self.vectorizer.client.get_collection("textbook_os")
        metadata = {key: value for key, value in collection.metadata.items()
                    if key not in {"embedding_fingerprint", "hnsw:space"}}
        collection.modify(metadata=metadata)
        before = index_revision(self.db)
        provider = FakeEmbeddingProvider(model="other-embedding")
        with (
            contextlib.redirect_stdout(io.StringIO()),
            MultiBookVectorizer(db_path=self.db, embedding_provider=provider) as writer,
            self.assertRaisesRegex(ValueError, "模型与当前 Provider 不一致"),
        ):
            writer.vectorize_book(self.chunks_path, "os", clear_existing=False)
        self.assertEqual(provider.document_calls, 0)
        self.assertEqual(index_revision(self.db), before)

    def test_published_vectors_survive_a_small_native_cache(self):
        # Windows uses a much smaller native cache. Reproduce that limit on
        # every platform, without changing the process's real handle limits.
        original_init = RustBindingsAPI.__init__

        def small_cache(api, system):
            original_init(api, system)
            api.hnsw_cache_size = 64

        with (
            patch.object(RustBindingsAPI, "__init__", small_cache),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            MultiBookVectorizer(db_path=self.root / "small-cache", embedding_provider=self.provider) as writer,
        ):
            writer.vectorize_book(self.chunks_path, "seed")
            source = writer.client.get_collection("textbook_seed")
            expected = source.get(include=["embeddings"])["embeddings"].tolist()
            # This file is absent under Chroma's default sync threshold, even
            # after count/get. Check durability before applying cache pressure.
            self.assertTrue(list(writer.db_path.glob("*/index_metadata.pickle")))
            for index in range(130):
                writer.vectorize_book(self.chunks_path, f"other{index}")
            rows = source.get(include=["embeddings"])
            self.assertEqual(rows["embeddings"].tolist(), expected)
            ranked = source.query(query_embeddings=[[1., 0.]], n_results=2)
            self.assertEqual(set(ranked["ids"][0]), {"old_p0", "old_p1"})

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
