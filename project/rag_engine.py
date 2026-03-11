# rag_engine.py
"""
RAG 问答系统核心引擎（已集成 BM25 + 向量语义混合检索 + LLM）
"""

import json
import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from vectorize_chunks import MultiBookVectorizer

# 关键词检索（BM25）
from rank_bm25 import BM25Okapi
import jieba

_API_KEY   = os.getenv("RAG_API_KEY")  or os.getenv("LLM_API_KEY", "")
_API_BASE  = os.getenv("RAG_API_BASE") or os.getenv("LLM_API_BASE", "https://api.ohmygpt.com/v1")
_LLM_MODEL = os.getenv("RAG_MODEL")   or os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview")


class RAGEngine:
    """RAG问答引擎（Hybrid：Embedding + BM25 + LLM）"""

    def __init__(
            self,
            db_path: str = "./vector_db",
            model_name: str = "BAAI/bge-large-zh-v1.5",
            enable_llm: bool = True,
            api_key: str = _API_KEY,
            api_base: str = _API_BASE,
            llm_model: str = _LLM_MODEL,
            verbose: bool = True,
            enable_reranker: bool = True,
            reranker_model: str = "BAAI/bge-reranker-base",
            enable_hyde: bool = True
    ):
        """
        初始化 RAG 引擎

        Args:
            db_path: 向量数据库路径
            model_name: 嵌入模型名称
            enable_llm: 是否启用 LLM
            api_key: LLM API 密钥
            api_base: LLM API 地址
            llm_model: LLM 模型名称
            verbose: 是否显示详细日志
            enable_reranker: 是否启用 Cross-Encoder 重排序
            reranker_model: 重排序模型名称
        """
        print("🚀 初始化 RAG 引擎...")

        self.verbose = verbose
        self.enable_hyde = enable_hyde

        # 向量数据库
        self.vectorizer = MultiBookVectorizer(
            model_name=model_name,
            db_path=db_path
        )

        # BM25 索引
        self.bm25_indexes = {}
        self.bm25_corpus = {}
        self.bm25_doc_ids = {}
        self._build_bm25_indexes()

        # Cross-Encoder 重排序
        self.reranker = None
        if enable_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(reranker_model)
                print("✅ Reranker 初始化完成")
            except Exception as e:
                print(f"⚠️  Reranker 初始化失败: {e}")

        # LLM 客户端
        self.llm = None
        self.enable_llm = enable_llm

        if enable_llm:
            try:
                from llm_client import create_llm_client
                self.llm = create_llm_client(
                    api_key=api_key,
                    base_url=api_base,
                    model=llm_model,
                    verbose=verbose
                )
                print("✅ LLM 客户端初始化完成")
            except Exception as e:
                print(f"⚠️  LLM 初始化失败: {e}")
                print("   将只提供检索功能，不生成答案")
                self.enable_llm = False
        else:
            print("ℹ️  LLM 功能已禁用")

        print("✅ RAG 引擎初始化完成\n")

    # ======================================================================
    #                 构建 BM25 关键词检索索引
    # ======================================================================
    def _build_bm25_indexes(self):
        """构建 BM25 索引"""
        if self.verbose:
            print("📚 构建 BM25 关键词索引...")

        collections = self.vectorizer.client.list_collections()
        book_collections = [c for c in collections if c.name.startswith("textbook_")]

        for col in book_collections:
            book_name = col.name.replace("textbook_", "")
            data = col.get(include=["documents", "metadatas"])
            docs = data["documents"]
            ids = data["ids"]
            tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
            bm25 = BM25Okapi(tokenized_docs)

            self.bm25_indexes[book_name] = bm25
            self.bm25_corpus[book_name] = docs
            self.bm25_doc_ids[book_name] = ids

        if self.verbose:
            print(f"✅ BM25 索引构建完成（{len(book_collections)} 本教材）")

    # ======================================================================
    #                 Cross-Encoder 重排序
    # ======================================================================
    def _rerank(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """使用 Cross-Encoder 对候选结果重排序"""
        if not self.reranker or not results:
            return results[:top_k]
        pairs = [(query, r["content"]) for r in results]
        scores = self.reranker.predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]

    # ======================================================================
    #                 BM25 关键词检索
    # ======================================================================
    def search_bm25(self, book_name: str, query: str, top_k: int = 3):
        """BM25 关键词检索"""
        if book_name not in self.bm25_indexes:
            return []

        bm25 = self.bm25_indexes[book_name]
        docs = self.bm25_corpus[book_name]
        ids = self.bm25_doc_ids[book_name]

        query_tokens = list(jieba.cut(query))
        scores = bm25.get_scores(query_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        collection = self.vectorizer.client.get_collection(f"textbook_{book_name}")

        for rank, idx in enumerate(top_idx, 1):
            doc = docs[idx]
            doc_id = ids[idx]
            metadata = collection.get(ids=[doc_id])["metadatas"][0]
            scaled_score = float(scores[idx]) * 0.05

            results.append({
                "rank": rank,
                "similarity": scaled_score,
                "method": "bm25",
                "book_name": metadata["book_name"],
                "chapter": metadata["chapter"],
                "section_h2": metadata["section_h2"],
                "section_h3": metadata.get("section_h3", ""),
                "content": doc,
                "has_code": metadata["has_code"],
                "has_image": metadata["has_image"],
                "char_count": metadata["char_count"],
            })

        return results

    # ======================================================================
    #                 HyDE：生成假设性文档
    # ======================================================================
    def _generate_hypothetical_doc(self, query: str) -> str:
        """用 LLM 生成一段假设性教材原文，用于 HyDE 检索"""
        prompt = """请根据以下问题，生成一段约100字的假设性教材原文，
就像这个问题的答案出现在计算机教材正文中的样子。
只输出正文内容，不要包含问题本身或任何前缀。

问题：{query}

教材原文：""".format(query=query)
        try:
            response = self.llm.generate_answer(prompt, temperature=0.3, max_tokens=200)
            if response["success"]:
                if self.verbose:
                    print(f"📝 HyDE 假设文档: {response['answer'][:80]}...")
                return response["answer"]
        except Exception as e:
            if self.verbose:
                print(f"⚠️  HyDE 生成失败，回退原始查询: {e}")
        return query

    # ======================================================================
    #                原本的语义向量检索
    # ======================================================================
    def search_embedding(self, book_name: str, query: str, top_k: int = 3):
        """语义向量检索"""
        collection_name = f"textbook_{book_name}"

        try:
            collection = self.vectorizer.client.get_collection(collection_name)
        except:
            if self.verbose:
                print(f"❌ 未找到教材集合: {collection_name}")
            return []

        # HyDE：用假设性文档 embedding；否则用 BGE instruction 前缀
        if self.enable_hyde and self.enable_llm and self.llm:
            encode_text = self._generate_hypothetical_doc(query)
        else:
            encode_text = "为这个句子生成表示以用于检索相关文章：" + query

        query_emb = self.vectorizer.model.encode(
            [encode_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=query_emb,
            n_results=top_k
        )

        formatted = []
        for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
        ), 1):
            formatted.append({
                "rank": i,
                "similarity": float(1 - distance),
                "method": "embedding",
                "book_name": metadata["book_name"],
                "chapter": metadata["chapter"],
                "section_h2": metadata["section_h2"],
                "section_h3": metadata.get("section_h3", ""),
                "content": doc,
                "has_code": metadata["has_code"],
                "has_image": metadata["has_image"],
                "char_count": metadata["char_count"],
            })

        return formatted

    # ======================================================================
    #                 混合（Hybrid）检索：BM25 + Embedding
    # ======================================================================
    def search_single_book(self, book_name: str, query: str, top_k: int = 5):
        """混合检索（单本教材）"""
        candidate_k = top_k * 3 if self.reranker else top_k
        semantic = self.search_embedding(book_name, query, candidate_k * 3)
        keyword = self.search_bm25(book_name, query, candidate_k)

        # 过滤掉干扰内容
        semantic = [r for r in semantic if not any(x in r["section_h2"] for x in ["小结", "习题", "思考题"])]
        semantic = [r for r in semantic if r["char_count"] > 100]

        # embedding 权重 1.0，BM25 权重 0.3（提升术语密集型问题的检索精度）
        for r in semantic:
            r["final_score"] = r["similarity"] * 1.0
        for r in keyword:
            r["final_score"] = r["similarity"] * 0.3

        combined = semantic + keyword
        combined.sort(key=lambda x: x["final_score"], reverse=True)
        combined = combined[:candidate_k]

        return self._rerank(query, combined, top_k)

    # ======================================================================
    #          跨教材搜索：自动对每本教材做 Hybrid 搜索
    # ======================================================================
    def search_all_books(self, query: str, top_k_per_book: int = 3):
        """跨教材检索"""
        collections = self.vectorizer.client.list_collections()
        book_collections = [c for c in collections if c.name.startswith("textbook_")]

        all_results = {}
        for col in book_collections:
            book_name = col.name.replace("textbook_", "")
            results = self.search_single_book(book_name, query, top_k_per_book)
            if results:
                all_results[book_name] = results

        return all_results

    # ======================================================================
    #                     构建上下文（控制长度）
    # ======================================================================
    def build_context(self, results: List[Dict], max_length: int = 2000):
        """构建上下文"""
        context = ""
        length = 0

        for i, r in enumerate(results, 1):
            block = f"""
【参考资料 {i}】（相似度: {r['similarity']:.3f} | 方法: {r['method']}）
 教材: {r['book_name']}
 章节: {r['chapter']} - {r['section_h2']}
 内容:
{r['content']}
---
"""

            if length + len(block) > max_length:
                context += "\n（部分内容省略）\n"
                break

            context += block
            length += len(block)

        return context

    # ======================================================================
    #                     构建 Prompt
    # ======================================================================
    def build_prompt(self, query: str, context: str):
        """构建 Prompt"""
        sys_prompt = """你是一个计算机课程的专业 AI 助教，请严格依据教材内容回答问题。

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

        prompt = f"""{sys_prompt}

## 学生问题
{query}

## 相关教材内容
{context}

请开始你的回答：
"""
        return prompt

    # ======================================================================
    #                  ⭐ 完整问答流程（检索 + LLM 生成）
    # ======================================================================
    def ask(
            self,
            query: str,
            book_name: Optional[str] = None,
            top_k: int = 5,
            use_llm: bool = True,
            temperature: float = 0.7,
            max_tokens: int = 2000
    ) -> Dict:
        """
        完整的 RAG 问答流程

        Args:
            query: 用户问题
            book_name: 指定教材（None 则搜索所有）
            top_k: 返回结果数量
            use_llm: 是否使用 LLM 生成答案
            temperature: LLM 温度参数
            max_tokens: LLM 最大生成长度

        Returns:
            包含检索结果和生成答案的字典
        """
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"🔍 查询: {query}")
            print(f"{'=' * 70}\n")

        # 1. 检索相关内容
        if book_name:
            results = self.search_single_book(book_name, query, top_k)
        else:
            all_results = self.search_all_books(query, top_k_per_book=top_k // 2)
            results = []
            for rs in all_results.values():
                results.extend(rs)
            results.sort(key=lambda x: x.get("final_score", x["similarity"]), reverse=True)
            results = self._rerank(query, results, top_k)

        # 检查是否找到结果
        if not results:
            return {
                "query": query,
                "results": [],
                "context": "",
                "prompt": "",
                "answer": "❌ 没有找到相关内容",
                "llm_response": None,
                "success": False
            }

        if self.verbose:
            print(f"📊 找到 {len(results)} 条相关内容\n")

        # 2. 构建上下文和 Prompt
        context = self.build_context(results)
        prompt = self.build_prompt(query, context)

        # 3. 是否使用 LLM 生成答案
        llm_response = None
        answer = None

        if use_llm and self.enable_llm and self.llm:
            if self.verbose:
                print("🤖 正在生成答案...\n")

            llm_response = self.llm.generate_answer(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if llm_response["success"]:
                answer = llm_response["answer"]

                if self.verbose:
                    print("\n✅ 答案生成完成！\n")
                    print("=" * 70)
                    print(answer)
                    print("=" * 70)
                    print(f"\n📊 统计信息：")
                    print(f"   - 使用模型: {llm_response['model']}")
                    print(f"   - Token 消耗: {llm_response['tokens']['total']} "
                          f"(提示:{llm_response['tokens']['prompt']} + "
                          f"生成:{llm_response['tokens']['completion']})")
                    print(f"   - 生成时间: {llm_response['time']} 秒")
                    print(f"   - 引用资料: {len(results)} 条\n")
            else:
                if self.verbose:
                    print(f"❌ 生成失败: {llm_response['error']}\n")

                answer = llm_response["answer"]

                # 生成失败时显示检索结果
                if self.verbose:
                    print("📋 检索到的相关内容：\n")
                    self.display_results({"results": results})
        else:
            # 不使用 LLM 时显示检索结果
            if self.verbose:
                if not self.enable_llm:
                    print("ℹ️  LLM 未启用，显示检索结果：\n")
                else:
                    print("ℹ️  未使用 LLM，显示检索结果：\n")

                self.display_results({"results": results})

                # 显示 prompt 供参考
                print("\n" + "=" * 70)
                print("📝 生成的 Prompt（可用于其他 LLM）：")
                print("=" * 70)
                print(prompt[:800])
                if len(prompt) > 800:
                    print("...\n（省略部分内容）")
                print("=" * 70 + "\n")

        return {
            "query": query,
            "results": results,
            "context": context,
            "prompt": prompt,
            "answer": answer,
            "llm_response": llm_response,
            "success": llm_response["success"] if llm_response else False
        }

    # ======================================================================
    #                  保留原有的 answer 方法（向后兼容）
    # ======================================================================
    def answer(self, query: str, book_name: Optional[str] = None, top_k: int = 5):
        """
        原有的方法，返回检索结果和 prompt（不调用 LLM）

        已废弃：建议使用 ask() 方法
        """
        if self.verbose:
            print("⚠️  警告：answer() 方法已废弃，建议使用 ask(use_llm=False)")

        return self.ask(query=query, book_name=book_name, top_k=top_k, use_llm=False)

    # ======================================================================
    #                   输出美化
    # ======================================================================
    def display_results(self, result_dict):
        """显示检索结果"""
        results = result_dict.get("results", [])

        if not results:
            print("❌ 没有找到相关内容")
            return

        print(f"📊 找到 {len(results)} 条相关内容：\n")

        for i, r in enumerate(results, 1):
            print("─" * 70)
            print(f"【结果 {i}】")
            print(f"  📈 相似度: {r['similarity']:.4f}  🔍 方法: {r['method']}")
            print(f"  📚 教材: {r['book_name']}")
            print(f"  📖 章节: {r['chapter']} | {r['section_h2']}")

            # 内容预览（更智能的截断）
            content = r['content']
            if len(content) > 150:
                print(f"  📝 内容: {content[:150]}...")
            else:
                print(f"  📝 内容: {content}")

            # 额外信息
            extra_info = []
            if r.get('has_code'):
                extra_info.append("💻 含代码")
            if r.get('has_image'):
                extra_info.append("🖼️ 含图片")

            if extra_info:
                print(f"  ℹ️  {' | '.join(extra_info)}")

        print("─" * 70 + "\n")


# ======================================================================
#                            测试入口
# ======================================================================
def main():
    """主函数：测试 RAG + LLM 完整流程"""

    db_path = r"./vector_db"

    # 初始化 RAG 引擎（可选择是否启用 LLM）
    # enable_llm=True：启用 LLM 生成答案
    # enable_llm=False：只进行检索，不生成答案
    rag = RAGEngine(
        db_path=db_path,
        enable_llm=True,  # ⭐ 改为 True 启用 LLM
        verbose=True
    )

    # 测试问题
    # book 标识符对应 ChromaDB collection 名（去掉 textbook_ 前缀）：
    #   os / os_mineru                              → 操作系统
    #   computer_organization / *_mineru            → 计算机组成原理
    #   computer_network / *_mineru                 → 计算机网络
    #   data_structure / *_mineru                   → 数据结构
    #   database / database_mineru                  → 数据库原理及应用教程
    test_queries = [
        {"query": "什么是进程？", "book": "os"},
        {"query": "CPU 的功能是什么？", "book": "computer_organization"},
        {"query": "什么是死锁？如何预防？", "book": "os"},
    ]

    print("\n" + "=" * 70)
    print("🎓 计算机课程 AI 助教系统")
    print("=" * 70)
    rag.vectorizer.list_books()
    print("💡 使用说明：")
    print("   - 输入 'test' 运行测试用例")
    print("   - 输入问题直接提问")
    print("   - 输入 'quit' 或 'exit' 退出")
    print("=" * 70 + "\n")

    # 交互式问答循环
    while True:
        user_input = input("\n💬 你的问题 > ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 感谢使用，再见！\n")
            break

        if user_input.lower() == 'test':
            for i, case in enumerate(test_queries, 1):
                print(f"\n{'=' * 70}")
                print(f"测试用例 {i}/{len(test_queries)}")
                print(f"{'=' * 70}")

                rag.ask(
                    query=case["query"],
                    book_name=case.get("book"),
                    top_k=5
                )

                if i < len(test_queries):
                    input("\n按回车继续下一个问题...")
            continue

        # 正常问答
        rag.ask(query=user_input, top_k=5)


if __name__ == "__main__":
    main()
