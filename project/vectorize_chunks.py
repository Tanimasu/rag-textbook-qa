# vectorize_multi_books.py
"""
文件名: vectorize_chunks.py
主要功能:
    - 读取 JSON 格式的文本分块，使用 sentence-transformers（text2vec-base-chinese）生成向量嵌入
    - 将向量和元数据（章节、级别、是否含代码/图片等）批量存入 ChromaDB，每本教材独立一个集合
    - 支持多本教材管理：可添加、列出、在单本或全部教材中进行相似度检索
    - 提供 MultiBookVectorizer 类，封装完整的向量化与检索流程
所属模块: 向量化 / 检索
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time


class MultiBookVectorizer:
    def __init__(
            self,
            model_name: str = "BAAI/bge-large-zh-v1.5",
            db_path: str = "./chroma_db"
    ):
        """初始化向量化器"""
        print("初始化向量化器...")

        # 加载 embedding 模型y
        print(f"加载模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("模型加载完成")

        # 初始化 Chroma 客户端
        print(f"初始化向量数据库: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        print("数据库初始化完成\n")

    def vectorize_book(
            self,
            chunks_path: str,
            book_name: str,  # 书籍标识
            batch_size: int = 32,
            clear_existing: bool = True
    ):
        """
        向量化单本教材

        Args:
            chunks_path: chunks JSON 文件路径
            book_name: 书籍标识（用作集合名称），如 "os", "computer_organization"
            batch_size: 批处理大小
            clear_existing: 是否清空该书的旧数据
        """
        print("=" * 70)
        print(f"开始向量化教材: {book_name}")
        print("=" * 70)

        # 为每本书创建独立的集合
        collection_name = f"textbook_{book_name}"

        # 清空旧数据
        if clear_existing:
            try:
                self.client.delete_collection(collection_name)
                print(f"🗑已清空旧数据: {collection_name}")
            except:
                pass

        # 创建新集合
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": f"{book_name} 教材分块",
                "hnsw:space": "cosine"
            }
        )
        print(f"集合创建: {collection_name}\n")

        # 加载 chunks
        print(f"加载 chunks: {chunks_path}")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        total = len(chunks)
        print(f"加载了 {total} 个 chunks\n")

        # 批量处理
        print(f"开始向量化（批大小={batch_size}）...")
        start_time = time.time()

        for i in tqdm(range(0, total, batch_size), desc="向量化进度"):
            batch_chunks = chunks[i:i + batch_size]

            # 准备数据
            ids = [chunk['chunk_id'] for chunk in batch_chunks]
            documents = [chunk['content'] for chunk in batch_chunks]

            # 准备元数据（添加书籍标识）
            metadatas = [
                {
                    'book_name': book_name,  # 添加书籍标识
                    'chapter': chunk['chapter'],
                    'section_h2': chunk['section_h2'],
                    'section_h3': chunk.get('section_h3', ''),
                    'section_h4': chunk.get('section_h4', ''),
                    'level': chunk['level'],
                    'char_count': chunk['char_count'],
                    'has_code': chunk['has_code'],
                    'has_image': chunk['has_image'],
                }
                for chunk in batch_chunks
            ]

            # 生成 embeddings
            embeddings = self.model.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()

            # 存入数据库
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

        elapsed_time = time.time() - start_time

        # 完成
        print("\n" + "=" * 70)
        print(f"  《{book_name}》向量化完成！")
        print("=" * 70)
        print(f"   统计信息:")
        print(f"   集合名称: {collection_name}")
        print(f"   总块数: {total}")
        print(f"   耗时: {elapsed_time:.2f} 秒")
        print(f"   平均: {elapsed_time / total:.3f} 秒/块")
        print(f"   数据库大小: {collection.count()} 条")
        print("=" * 70)

        return collection_name

    def search_book(
            self,
            book_name: str,  # 指定搜索哪本书
            query: str,
            top_k: int = 5
    ):
        """
        在指定教材中搜索

        Args:
            book_name: 书籍标识
            query: 查询文本
            top_k: 返回top-k结果
        """
        collection_name = f"textbook_{book_name}"

        try:
            collection = self.client.get_collection(collection_name)
        except:
            print(f"   找不到集合: {collection_name}")
            print(f"   请先运行 vectorize_book() 向量化该教材")
            return

        print(f"\n 搜索教材: 《{book_name}》")
        print(f" 查询内容: '{query}'")
        print("-" * 70)

        # 生成查询向量
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        # 搜索
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        # 显示结果
        for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
        ), 1):
            similarity = 1 - distance

            print(f"\n【结果 {i}】")
            print(f"   相似度: {similarity:.4f}")
            print(f"   教材: {metadata['book_name']}")
            print(f"   章节: {metadata['chapter']}")
            print(f"   小节: {metadata['section_h2']}")
            if metadata.get('section_h3'):
                print(f"         {metadata['section_h3']}")

            # 标签
            tags = []
            if metadata['has_code']:
                tags.append("代码")
            if metadata['has_image']:
                tags.append("图片")
            if tags:
                print(f"    标签: {' '.join(tags)}")

            print(f"   内容预览:")
            print(f"     {doc[:200]}...")
            print("-" * 70)

    def list_books(self):
        """列出所有已向量化的教材"""
        print("\n 已向量化的教材列表：")
        print("-" * 70)

        collections = self.client.list_collections()
        textbook_collections = [c for c in collections if c.name.startswith("textbook_")]

        if not textbook_collections:
            print("  （暂无数据）")
        else:
            for i, collection in enumerate(textbook_collections, 1):
                book_name = collection.name.replace("textbook_", "")
                count = collection.count()
                print(f"  {i}. 《{book_name}》 - {count} 个chunks")

        print("-" * 70)


def _parse_selection(raw: str, total: int) -> list[int]:
    """将用户输入解析为 0-based 索引列表。
    支持格式：all / 1 / 1,3 / 1-3 / 1,3-5,7
    """
    raw = raw.strip().lower()
    if raw == "all":
        return list(range(total))

    indices = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            indices.update(range(int(a) - 1, int(b)))  # 转 0-based
        elif part.isdigit():
            indices.add(int(part) - 1)
    return sorted(i for i in indices if 0 <= i < total)


def main():
    """主函数：自动扫描 output/ 下所有 *_chunks.json，交互选择后向量化"""
    from pathlib import Path
    import re

    output_dir = Path(r"D:\CodeField\Graduation_project\project\output")
    db_path    = r"D:\CodeField\Graduation_project\project\vector_db"

    # ── 文件 stem → ChromaDB 集合名映射表 ───────────────────────────────
    # ChromaDB 只允许 [a-zA-Z0-9._-]，且首尾必须是字母或数字
    BOOK_NAME_MAP = {
        "操作系统":                   "os",
        "操作系统_mineru":            "os_mineru",
        "计算机组成原理":             "computer_organization",
        "计算机组成原理_mineru":      "computer_organization_mineru",
        "计算机网络":                 "computer_network",
        "计算机网络_mineru":          "computer_network_mineru",
        "数据结构":                   "data_structure",
        "数据结构_mineru":            "data_structure_mineru",
        "数据库原理及应用教程":       "database",
        "数据库原理及应用教程_mineru":"database_mineru",
    }

    def to_collection_name(stem: str) -> str:
        """stem 已去掉 _chunks 后缀，转为合法集合名"""
        if stem in BOOK_NAME_MAP:
            return BOOK_NAME_MAP[stem]
        # 兜底：去掉非法字符，保留 [a-zA-Z0-9._-]
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", stem).strip("_")
        if not safe:
            safe = "book_" + str(abs(hash(stem)))[:8]
        return safe

    # ── 1. 自动发现所有 chunks 文件 ─────────────────────────────────────
    chunk_files = sorted(output_dir.glob("*_chunks.json"))
    if not chunk_files:
        print(f"在 {output_dir} 下没有找到 *_chunks.json 文件")
        return

    print("\n" + "=" * 70)
    print(" 可用的 chunks 文件")
    print("=" * 70)
    for i, f in enumerate(chunk_files, 1):
        size_kb = f.stat().st_size // 1024
        print(f"  {i:2d}. {f.name:<50s}  ({size_kb} KB)")
    print("=" * 70)

    # ── 2. 选择要处理的文件 ─────────────────────────────────────────────
    print("\n输入要向量化的编号（支持格式：all / 1 / 1,3 / 1-3 / 1,3-5）")
    raw = input("选择: ").strip()
    selected_indices = _parse_selection(raw, len(chunk_files))

    if not selected_indices:
        print("未选中任何文件，退出。")
        return

    selected_files = [chunk_files[i] for i in selected_indices]
    print(f"\n已选择 {len(selected_files)} 个文件：")
    for f in selected_files:
        print(f"  - {f.name}")

    # ── 3. 初始化向量化器（只加载一次模型）──────────────────────────────
    vectorizer = MultiBookVectorizer(db_path=db_path)

    # ── 4. 逐一向量化 ────────────────────────────────────────────────────
    success, failed = [], []
    for f in selected_files:
        # book_name 取文件 stem，去掉末尾 _chunks，再映射为合法集合名
        stem = f.stem[:-7] if f.stem.endswith("_chunks") else f.stem
        book_name = to_collection_name(stem)
        try:
            vectorizer.vectorize_book(
                chunks_path=str(f),
                book_name=book_name,
                batch_size=32,
                clear_existing=True
            )
            success.append(book_name)
        except Exception as e:
            print(f"\n错误：{f.name} 向量化失败 — {e}")
            import traceback; traceback.print_exc()
            failed.append(book_name)

    # ── 5. 汇总 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("向量化汇总")
    print(f"  成功: {len(success)} 本  {success}")
    if failed:
        print(f"  失败: {len(failed)} 本  {failed}")
    print("=" * 70)

    vectorizer.list_books()


if __name__ == "__main__":
    main()
