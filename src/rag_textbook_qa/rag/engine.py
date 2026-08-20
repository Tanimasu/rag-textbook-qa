"""Hybrid BM25, embedding, reranking, and LLM question-answering engine."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, Self

import jieba
from rank_bm25 import BM25Okapi

from rag_textbook_qa.indexing import MultiBookVectorizer
from rag_textbook_qa.llm import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    create_llm_client,
)
from rag_textbook_qa.providers import (
    ComputeSettings,
    EmbeddingProvider,
    RerankerProvider,
)
from rag_textbook_qa.providers.factory import create_reranker_provider


class AnswerGenerator(Protocol):
    def generate_answer(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        retry: int = 2,
    ) -> dict[str, Any]: ...


def _environment_value(primary: str, fallback: str, default: str) -> str:
    return os.getenv(primary) or os.getenv(fallback) or default


class RAGEngine:
    """Hybrid textbook RAG engine with injectable compute and LLM providers."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        model_name: str | None = None,
        enable_llm: bool = True,
        api_key: str | None = None,
        api_base: str | None = None,
        llm_model: str | None = None,
        verbose: bool = True,
        enable_reranker: bool = True,
        reranker_model: str | None = None,
        enable_hyde: bool = True,
        embedding_provider: EmbeddingProvider | None = None,
        reranker_provider: RerankerProvider | None = None,
        compute_settings: ComputeSettings | None = None,
        llm_client: AnswerGenerator | None = None,
    ) -> None:
        print("初始化 RAG 引擎...")
        self.verbose = verbose
        self.enable_hyde = enable_hyde

        if compute_settings is None:
            providers_fully_injected = embedding_provider is not None and (
                reranker_provider is not None or not enable_reranker
            )
            compute_settings = (
                ComputeSettings() if providers_fully_injected else ComputeSettings.from_env()
            )

        self.vectorizer = MultiBookVectorizer(
            model_name=model_name,
            db_path=db_path,
            embedding_provider=embedding_provider,
            compute_settings=compute_settings,
            allow_query_fallback=True,
        )

        self.bm25_indexes: dict[str, BM25Okapi] = {}
        self.bm25_corpus: dict[str, list[str]] = {}
        self.bm25_doc_ids: dict[str, list[str]] = {}
        self._build_bm25_indexes()

        self.reranker = reranker_provider
        if enable_reranker and self.reranker is None:
            reranker_settings = replace(
                compute_settings,
                reranker_model=reranker_model or compute_settings.reranker_model,
            )
            self.reranker = create_reranker_provider(
                reranker_settings,
                allow_query_fallback=True,
            )
        if self.reranker is not None and self.verbose:
            print(f"Reranker 已配置: {self.reranker.identity.model}")

        self.llm = llm_client
        self.enable_llm = enable_llm
        self.llm_initialization_error: str | None = None
        if enable_llm and self.llm is None:
            resolved_api_key = (
                api_key
                if api_key is not None
                else _environment_value("RAG_API_KEY", "LLM_API_KEY", "")
            )
            resolved_api_base = (
                api_base
                if api_base is not None
                else _environment_value(
                    "RAG_API_BASE",
                    "LLM_API_BASE",
                    DEFAULT_LLM_BASE_URL,
                )
            )
            resolved_model = (
                llm_model
                if llm_model is not None
                else _environment_value("RAG_MODEL", "LLM_MODEL", DEFAULT_LLM_MODEL)
            )
            try:
                self.llm = create_llm_client(
                    api_key=resolved_api_key,
                    base_url=resolved_api_base,
                    model=resolved_model,
                    verbose=verbose,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                self.llm_initialization_error = str(exc)
                if self.verbose:
                    print(f"LLM 初始化失败: {exc}")
                    print("将只提供检索功能，不生成答案")
                self.enable_llm = False

        if self.verbose:
            print("RAG 引擎初始化完成\n")

    def close(self) -> None:
        """Release the underlying Chroma client and its file handles."""

        self.vectorizer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _build_bm25_indexes(self) -> None:
        if self.verbose:
            print("构建 BM25 关键词索引...")
        collections = self.vectorizer.client.list_collections()
        book_collections = [
            collection
            for collection in collections
            if collection.name.startswith("textbook_")
        ]
        indexed_count = 0
        for collection in book_collections:
            book_name = collection.name.removeprefix("textbook_")
            data = collection.get(include=["documents", "metadatas"])
            documents = data["documents"] or []
            if not documents:
                continue
            document_ids = data["ids"]
            self.bm25_indexes[book_name] = BM25Okapi(
                [list(jieba.cut(document)) for document in documents]
            )
            self.bm25_corpus[book_name] = documents
            self.bm25_doc_ids[book_name] = document_ids
            indexed_count += 1
        if self.verbose:
            print(f"BM25 索引构建完成（{indexed_count} 本教材）")

    def _rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not self.reranker or not results:
            return results[:top_k]
        scores = self.reranker.rerank(
            query,
            [result["content"] for result in results],
        )
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
        results.sort(key=lambda item: item["rerank_score"], reverse=True)
        return results[:top_k]

    def search_bm25(
        self,
        book_name: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if book_name not in self.bm25_indexes:
            return []

        bm25 = self.bm25_indexes[book_name]
        documents = self.bm25_corpus[book_name]
        document_ids = self.bm25_doc_ids[book_name]
        scores = bm25.get_scores(list(jieba.cut(query)))
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]
        collection = self.vectorizer.client.get_collection(f"textbook_{book_name}")

        results = []
        for rank, index in enumerate(top_indices, 1):
            metadata = collection.get(ids=[document_ids[index]])["metadatas"][0]
            results.append(
                {
                    "rank": rank,
                    "similarity": float(scores[index]) * 0.05,
                    "method": "bm25",
                    "book_name": metadata["book_name"],
                    "chapter": metadata["chapter"],
                    "section_h2": metadata["section_h2"],
                    "section_h3": metadata.get("section_h3", ""),
                    "content": documents[index],
                    "has_code": metadata["has_code"],
                    "has_image": metadata["has_image"],
                    "char_count": metadata["char_count"],
                }
            )
        return results

    def _generate_hypothetical_doc(self, query: str) -> str:
        prompt = f"""请根据以下问题，生成一段约100字的假设性教材原文，
就像这个问题的答案出现在计算机教材正文中的样子。
只输出正文内容，不要包含问题本身或任何前缀。

问题：{query}

教材原文："""
        if self.llm is None:
            return query
        try:
            response = self.llm.generate_answer(
                prompt,
                temperature=0.3,
                max_tokens=200,
            )
            if response["success"]:
                if self.verbose:
                    print(f"HyDE 假设文档: {response['answer'][:80]}...")
                return str(response["answer"])
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if self.verbose:
                print(f"HyDE 生成失败，回退原始查询: {exc}")
        return query

    def search_embedding(
        self,
        book_name: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        collection_name = f"textbook_{book_name}"
        collection_names = {
            collection.name for collection in self.vectorizer.client.list_collections()
        }
        if collection_name not in collection_names:
            if self.verbose:
                print(f"未找到教材集合: {collection_name}")
            return []

        collection = self.vectorizer.client.get_collection(collection_name)
        self.vectorizer.validate_collection_embedding(collection)
        if self.enable_hyde and self.enable_llm and self.llm:
            hypothetical_document = self._generate_hypothetical_doc(query)
            query_embedding = self.vectorizer.embedding_provider.embed_documents(
                [hypothetical_document]
            )
        else:
            query_embedding = self.vectorizer.embedding_provider.embed_queries([query])

        response = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )
        return [
            {
                "rank": rank,
                "similarity": float(1 - distance),
                "method": "embedding",
                "book_name": metadata["book_name"],
                "chapter": metadata["chapter"],
                "section_h2": metadata["section_h2"],
                "section_h3": metadata.get("section_h3", ""),
                "content": document,
                "has_code": metadata["has_code"],
                "has_image": metadata["has_image"],
                "char_count": metadata["char_count"],
            }
            for rank, (document, metadata, distance) in enumerate(
                zip(
                    response["documents"][0],
                    response["metadatas"][0],
                    response["distances"][0],
                ),
                1,
            )
        ]

    def search_single_book(
        self,
        book_name: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        candidate_count = top_k * 3 if self.reranker else top_k
        semantic = self.search_embedding(book_name, query, candidate_count * 3)
        keyword = self.search_bm25(book_name, query, candidate_count)
        semantic = [
            result
            for result in semantic
            if not any(
                marker in result["section_h2"]
                for marker in ("小结", "习题", "思考题")
            )
            and result["char_count"] > 100
        ]
        for result in semantic:
            result["final_score"] = result["similarity"]
        for result in keyword:
            result["final_score"] = result["similarity"] * 0.3
        combined = sorted(
            semantic + keyword,
            key=lambda item: item["final_score"],
            reverse=True,
        )[:candidate_count]
        return self._rerank(query, combined, top_k)

    def search_all_books(
        self,
        query: str,
        top_k_per_book: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        if top_k_per_book <= 0:
            raise ValueError("top_k_per_book 必须大于 0")
        all_results = {}
        collections = sorted(
            self.vectorizer.client.list_collections(),
            key=lambda collection: collection.name,
        )
        for collection in collections:
            if not collection.name.startswith("textbook_"):
                continue
            book_name = collection.name.removeprefix("textbook_")
            results = self.search_single_book(book_name, query, top_k_per_book)
            if results:
                all_results[book_name] = results
        return all_results

    @staticmethod
    def build_context(
        results: list[dict[str, Any]],
        max_length: int = 2000,
    ) -> str:
        context = ""
        length = 0
        for index, result in enumerate(results, 1):
            block = f"""
【参考资料 {index}】（相似度: {result['similarity']:.3f} | 方法: {result['method']}）
 教材: {result['book_name']}
 章节: {result['chapter']} - {result['section_h2']}
 内容:
{result['content']}
---
"""
            if length + len(block) > max_length:
                context += "\n（部分内容省略）\n"
                break
            context += block
            length += len(block)
        return context

    @staticmethod
    def build_prompt(query: str, context: str) -> str:
        system_prompt = """你是一个计算机课程的专业 AI 助教，请严格依据教材内容回答问题。

要求：
1. 不要编造教材没有的内容
2. 先给出简明答案（2-3句话），再给出详细解释
3. 如有多个要点，使用编号列表
4. 最后标注引用的章节

回答格式示例：
## 简明答案
[2-3句话的核心答案]

## 详细解释
1. ...
2. ...

## 参考章节
📚 [章节信息]
"""
        return f"""{system_prompt}

## 学生问题
{query}

## 相关教材内容
{context}

请开始你的回答：
"""

    def ask(
        self,
        query: str,
        book_name: str | None = None,
        top_k: int = 5,
        use_llm: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if self.verbose:
            print(f"\n{'=' * 70}\n查询: {query}\n{'=' * 70}\n")

        if book_name:
            results = self.search_single_book(book_name, query, top_k)
        else:
            grouped = self.search_all_books(
                query,
                top_k_per_book=max(1, top_k // 2),
            )
            results = [result for group in grouped.values() for result in group]
            results.sort(
                key=lambda item: item.get("final_score", item["similarity"]),
                reverse=True,
            )
            results = self._rerank(query, results, top_k)

        if not results:
            return {
                "query": query,
                "results": [],
                "context": "",
                "prompt": "",
                "answer": "❌ 没有找到相关内容",
                "llm_response": None,
                "error": "没有找到相关内容",
                "success": False,
            }

        context = self.build_context(results)
        prompt = self.build_prompt(query, context)
        llm_response = None
        answer = None
        generation_error = None
        if use_llm and self.enable_llm and self.llm:
            llm_response = self.llm.generate_answer(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = llm_response["answer"]
            if not llm_response["success"]:
                generation_error = llm_response.get("error") or "LLM 生成失败"
            if self.verbose and llm_response["success"]:
                print(f"\n{answer}\n")
                print(
                    f"模型: {llm_response['model']} | "
                    f"tokens: {llm_response['tokens']['total']} | "
                    f"耗时: {llm_response['time']} 秒 | 引用: {len(results)} 条"
                )
            elif self.verbose:
                print(f"生成失败: {llm_response.get('error', '未知错误')}")
                self.display_results({"results": results})
        else:
            if use_llm:
                detail = self.llm_initialization_error or "LLM 未启用"
                generation_error = f"LLM 不可用：{detail}"
            if self.verbose:
                self.display_results({"results": results})
                print(prompt[:800] + ("..." if len(prompt) > 800 else ""))

        return {
            "query": query,
            "results": results,
            "context": context,
            "prompt": prompt,
            "answer": answer,
            "llm_response": llm_response,
            "error": generation_error,
            "success": llm_response["success"] if llm_response else False,
        }

    def answer(
        self,
        query: str,
        book_name: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        if self.verbose:
            print("警告：answer() 已废弃，建议使用 ask(use_llm=False)")
        return self.ask(
            query=query,
            book_name=book_name,
            top_k=top_k,
            use_llm=False,
        )

    @staticmethod
    def display_results(result_dict: dict[str, Any]) -> None:
        results = result_dict.get("results", [])
        if not results:
            print("没有找到相关内容")
            return
        print(f"找到 {len(results)} 条相关内容：\n")
        for index, result in enumerate(results, 1):
            print("─" * 70)
            print(f"【结果 {index}】")
            print(f"相似度: {result['similarity']:.4f} | 方法: {result['method']}")
            print(f"教材: {result['book_name']}")
            print(f"章节: {result['chapter']} | {result['section_h2']}")
            content = result["content"]
            print(f"内容: {content[:150]}{'...' if len(content) > 150 else ''}")
            extra = []
            if result.get("has_code"):
                extra.append("含代码")
            if result.get("has_image"):
                extra.append("含图片")
            if extra:
                print(f"标签: {' | '.join(extra)}")
        print("─" * 70 + "\n")
