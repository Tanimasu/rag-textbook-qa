"""
clean_db.py
交互式清理 ChromaDB 向量数据库：列出所有集合，按名称删除指定集合或清空全部。
运行方式：python clean_db.py
"""
import chromadb

DB_PATH = "./vector_db"


def clean_collections(db_path: str = DB_PATH):
    """列出所有集合并提示用户选择删除。"""
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()

    if not collections:
        print("数据库中没有任何集合。")
        return

    print("=" * 50)
    print(f"数据库: {db_path}")
    print("=" * 50)
    for i, col in enumerate(collections, 1):
        print(f"  {i}. {col.name}  ({col.count()} 条向量)")

    print("\n输入集合名称删除该集合，输入 all 删除所有，输入 q 退出。")
    name = input("> ").strip()

    if name == "q":
        print("已取消。")
        return

    if name == "all":
        for col in collections:
            client.delete_collection(col.name)
            print(f"已删除: {col.name}")
    else:
        try:
            client.delete_collection(name)
            print(f"已删除: {name}")
        except Exception:
            print(f"集合不存在: {name}")


if __name__ == "__main__":
    clean_collections()
