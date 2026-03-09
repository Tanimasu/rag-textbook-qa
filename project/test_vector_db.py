"""
test_vector_db.py
检查 ChromaDB 向量数据库的状态：列出所有已向量化的教材集合，并显示样本数据。
运行方式：python test_vector_db.py
"""
import chromadb

DB_PATH = "./vector_db"


def check_vector_db(db_path: str = DB_PATH):
    """列出数据库中所有教材集合，并打印每个集合的基本信息和一条样本记录。"""
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()

    if not collections:
        print("数据库为空，请先运行 vectorize_chunks.py 进行向量化。")
        return

    print("=" * 60)
    print(f"向量数据库路径: {db_path}")
    print(f"共 {len(collections)} 个集合")
    print("=" * 60)

    for col_info in collections:
        col = client.get_collection(col_info.name)
        count = col.count()
        print(f"\n集合: {col_info.name}  ({count} 条向量)")

        # 取一条样本，展示元数据和内容片段
        if count > 0:
            sample = col.get(limit=1, include=["documents", "metadatas"])
            metadata = sample["metadatas"][0]
            content = sample["documents"][0]
            print(f"  样本元数据: {metadata}")
            print(f"  样本内容:   {content[:100]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    check_vector_db()
