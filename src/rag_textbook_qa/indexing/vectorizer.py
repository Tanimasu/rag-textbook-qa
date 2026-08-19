"""Vectorize textbook chunks and store them in local Chroma collections."""

from __future__ import annotations

import json
import re
import time
import traceback
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import chromadb
from tqdm import tqdm

from rag_textbook_qa.catalog import book_id_from_chunk_stem
from rag_textbook_qa.config import Settings
from rag_textbook_qa.providers import ComputeSettings, EmbeddingProvider
from rag_textbook_qa.providers.factory import create_embedding_provider

_BOOK_ID_PATTERN = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?")
_REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "content",
    "chapter",
    "section_h2",
    "level",
    "char_count",
    "has_code",
    "has_image",
}


def _validate_book_id(book_name: str) -> str:
    if (
        not _BOOK_ID_PATTERN.fullmatch(book_name)
        or ".." in book_name
        or len(book_name) > 503
    ):
        raise ValueError(
            "book_name 必须是 Chroma 安全标识：只能包含字母、数字、点、下划线或"
            "连字符，首尾为字母或数字，不含连续点，且不超过 503 字符"
        )
    return book_name


def _load_chunks(chunks_path: str | Path) -> list[dict[str, Any]]:
    path = Path(chunks_path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        chunks = json.load(stream)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks 文件必须包含至少一个文本块")

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(chunks, 1):
        if not isinstance(item, dict):
            raise TypeError(f"第 {index} 个 chunk 必须是 JSON object")
        missing = sorted(_REQUIRED_CHUNK_FIELDS - item.keys())
        if missing:
            raise ValueError(f"第 {index} 个 chunk 缺少字段: {', '.join(missing)}")
        chunk_id = item["chunk_id"]
        content = item["content"]
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"第 {index} 个 chunk_id 必须是非空字符串")
        if chunk_id in seen_ids:
            raise ValueError(f"chunks 文件包含重复 chunk_id: {chunk_id}")
        if not isinstance(content, str) or not content:
            raise ValueError(f"第 {index} 个 content 必须是非空字符串")
        for field in ("chapter", "section_h2", "section_h3", "section_h4"):
            value = item.get(field, "")
            if not isinstance(value, str):
                raise TypeError(f"第 {index} 个 {field} 必须是字符串")
        for field in ("level", "char_count"):
            value = item[field]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"第 {index} 个 {field} 必须是整数")
        for field in ("has_code", "has_image"):
            if not isinstance(item[field], bool):
                raise TypeError(f"第 {index} 个 {field} 必须是布尔值")
        seen_ids.add(chunk_id)
        validated.append(item)
    return validated


def list_indexed_books(db_path: str | Path) -> list[dict[str, Any]]:
    """Return safe collection summaries without creating a model provider."""

    client = chromadb.PersistentClient(path=str(Path(db_path).expanduser().resolve()))
    summaries = []
    for collection in sorted(client.list_collections(), key=lambda item: item.name):
        if not collection.name.startswith("textbook_"):
            continue
        metadata = collection.metadata or {}
        summaries.append(
            {
                "book_name": collection.name.removeprefix("textbook_"),
                "collection_name": collection.name,
                "count": collection.count(),
                "embedding_model": metadata.get("embedding_model"),
                "embedding_fingerprint": metadata.get("embedding_fingerprint"),
            }
        )
    return summaries


class MultiBookVectorizer:
    """Build and query one Chroma collection per textbook."""

    def __init__(
        self,
        model_name: str | None = None,
        db_path: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        compute_settings: ComputeSettings | None = None,
        allow_query_fallback: bool = False,
    ) -> None:
        print("初始化向量化器...")

        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            provider_settings = compute_settings or ComputeSettings.from_env()
            if model_name is not None:
                provider_settings = replace(
                    provider_settings,
                    embedding_model=model_name,
                )
            self.embedding_provider = create_embedding_provider(
                provider_settings,
                allow_query_fallback=allow_query_fallback,
            )
        print(f"Embedding 模型: {self.embedding_provider.identity.model}")

        resolved_db_path = (
            Path(db_path).expanduser().resolve()
            if db_path is not None
            else Settings.load().paths.vector_db
        )
        print(f"初始化向量数据库: {resolved_db_path}")
        self.db_path = resolved_db_path
        self.client = chromadb.PersistentClient(path=str(resolved_db_path))
        print("数据库初始化完成\n")

    def _collection_exists(self, collection_name: str) -> bool:
        return any(
            collection.name == collection_name
            for collection in self.client.list_collections()
        )

    def vectorize_book(
        self,
        chunks_path: str | Path,
        book_name: str,
        batch_size: int = 32,
        clear_existing: bool = True,
    ) -> str:
        """Vectorize one chunk JSON file into ``textbook_{book_name}``."""

        book_name = _validate_book_id(book_name)
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        print("=" * 70)
        print(f"开始向量化教材: {book_name}")
        print("=" * 70)
        print(f"加载 chunks: {chunks_path}")
        chunks = _load_chunks(chunks_path)
        total = len(chunks)
        print(f"加载了 {total} 个 chunks\n")

        first_documents = [chunk["content"] for chunk in chunks[:batch_size]]
        first_embeddings = self.embedding_provider.embed_documents(first_documents)
        collection_name = f"textbook_{book_name}"
        collection_metadata = {
            "description": f"{book_name} 教材分块",
            "hnsw:space": "cosine",
            "embedding_model": self.embedding_provider.identity.model,
            "embedding_fingerprint": self.embedding_provider.identity.fingerprint,
        }

        if clear_existing:
            write_collection_name = f"ragbuild_{uuid.uuid4().hex}"
            collection = self.client.create_collection(
                name=write_collection_name,
                metadata=collection_metadata,
            )
        else:
            write_collection_name = collection_name
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=collection_metadata,
            )
            self._validate_collection_embedding(collection)
        print(f"集合写入目标: {write_collection_name}\n")

        print(f"开始向量化（批大小={batch_size}）...")
        start_time = time.time()
        try:
            for offset in tqdm(range(0, total, batch_size), desc="向量化进度"):
                batch_chunks = chunks[offset : offset + batch_size]
                ids = [chunk["chunk_id"] for chunk in batch_chunks]
                documents = [chunk["content"] for chunk in batch_chunks]
                metadatas = [
                    {
                        "book_name": book_name,
                        "chapter": chunk["chapter"],
                        "section_h2": chunk["section_h2"],
                        "section_h3": chunk.get("section_h3", ""),
                        "section_h4": chunk.get("section_h4", ""),
                        "level": chunk["level"],
                        "char_count": chunk["char_count"],
                        "has_code": chunk["has_code"],
                        "has_image": chunk["has_image"],
                    }
                    for chunk in batch_chunks
                ]
                embeddings = (
                    first_embeddings
                    if offset == 0
                    else self.embedding_provider.embed_documents(documents)
                )
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
        except BaseException:
            if clear_existing and self._collection_exists(write_collection_name):
                self.client.delete_collection(write_collection_name)
            raise

        if clear_existing:
            collection = self._promote_collection(collection, collection_name)

        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"《{book_name}》向量化完成！")
        print("=" * 70)
        print(f"集合名称: {collection_name}")
        print(f"总块数: {total}")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"平均: {elapsed_time / total:.3f} 秒/块")
        print(f"数据库大小: {collection.count()} 条")
        print("=" * 70)
        return collection_name

    def _promote_collection(self, staging_collection: Any, collection_name: str):
        """Replace the visible collection only after staging is complete."""

        staging_name = staging_collection.name
        backup_name = None
        if self._collection_exists(collection_name):
            backup_name = f"ragbackup_{uuid.uuid4().hex}"
            self.client.get_collection(collection_name).modify(name=backup_name)

        try:
            staging_collection.modify(name=collection_name)
        except BaseException:
            if backup_name and self._collection_exists(backup_name):
                self.client.get_collection(backup_name).modify(name=collection_name)
            if self._collection_exists(staging_name):
                self.client.delete_collection(staging_name)
            raise

        if backup_name and self._collection_exists(backup_name):
            self.client.delete_collection(backup_name)
            print(f"已替换旧数据: {collection_name}")
        return self.client.get_collection(collection_name)

    def search_book(self, book_name: str, query: str, top_k: int = 5) -> None:
        collection_name = f"textbook_{_validate_book_id(book_name)}"
        if not self._collection_exists(collection_name):
            print(f"找不到集合: {collection_name}")
            print("请先向量化该教材")
            return

        collection = self.client.get_collection(collection_name)
        self._validate_collection_embedding(collection)
        print(f"\n搜索教材: 《{book_name}》")
        print(f"查询内容: {query!r}")
        print("-" * 70)
        query_embedding = self.embedding_provider.embed_queries([query])
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

        for index, (document, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            1,
        ):
            print(f"\n【结果 {index}】")
            print(f"相似度: {1 - distance:.4f}")
            print(f"教材: {metadata['book_name']}")
            print(f"章节: {metadata['chapter']}")
            print(f"小节: {metadata['section_h2']}")
            if metadata.get("section_h3"):
                print(metadata["section_h3"])
            tags = []
            if metadata["has_code"]:
                tags.append("代码")
            if metadata["has_image"]:
                tags.append("图片")
            if tags:
                print(f"标签: {' '.join(tags)}")
            print(f"内容预览: {document[:200]}...")
            print("-" * 70)

    def list_books(self) -> None:
        print("\n已向量化的教材列表：")
        print("-" * 70)
        books = list_indexed_books(self.db_path)
        if not books:
            print("（暂无数据）")
        for index, book in enumerate(books, 1):
            print(f"{index}. 《{book['book_name']}》 - {book['count']} 个 chunks")
        print("-" * 70)

    def _validate_collection_embedding(self, collection: Any) -> None:
        metadata = collection.metadata or {}
        fingerprint = metadata.get("embedding_fingerprint")
        if fingerprint and fingerprint != self.embedding_provider.identity.fingerprint:
            raise ValueError(
                "向量库 embedding 模型与当前 Provider 不一致；"
                "请使用同一模型或重新向量化该教材"
            )


def parse_selection(raw: str, total: int) -> list[int]:
    """Parse ``all / 1 / 1,3 / 1-3`` into zero-based indices."""

    normalized = raw.strip().lower()
    if normalized == "all":
        return list(range(total))

    indices: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            if start.isdigit() and end.isdigit():
                indices.update(range(int(start) - 1, int(end)))
        elif part.isdigit():
            indices.add(int(part) - 1)
    return sorted(index for index in indices if 0 <= index < total)


def interactive_main() -> None:
    """Compatibility UI for selecting multiple files interactively."""

    from dotenv import load_dotenv

    settings = Settings.load()
    load_dotenv(settings.paths.root / "project" / ".env")
    chunk_files = sorted(settings.paths.chunks.glob("*_chunks.json"))
    if not chunk_files:
        print(f"在 {settings.paths.chunks} 下没有找到 *_chunks.json 文件")
        return

    print("\n" + "=" * 70)
    print("可用的 chunks 文件")
    print("=" * 70)
    for index, path in enumerate(chunk_files, 1):
        print(f"{index:2d}. {path.name:<50s} ({path.stat().st_size // 1024} KB)")
    print("=" * 70)
    raw = input("输入编号（all / 1 / 1,3 / 1-3）: ").strip()
    selected_indices = parse_selection(raw, len(chunk_files))
    if not selected_indices:
        print("未选中任何文件，退出。")
        return

    vectorizer = MultiBookVectorizer(db_path=settings.paths.vector_db)
    success: list[str] = []
    failed: list[str] = []
    for index in selected_indices:
        path = chunk_files[index]
        book_name = book_id_from_chunk_stem(path.stem)
        try:
            vectorizer.vectorize_book(path, book_name)
            success.append(book_name)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"\n错误：{path.name} 向量化失败 — {exc}")
            traceback.print_exc()
            failed.append(book_name)

    print("\n" + "=" * 70)
    print("向量化汇总")
    print(f"成功: {len(success)} 本 {success}")
    if failed:
        print(f"失败: {len(failed)} 本 {failed}")
    print("=" * 70)
    vectorizer.list_books()
