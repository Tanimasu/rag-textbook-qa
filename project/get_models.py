"""
get_models.py
查询当前 API 端点支持的模型列表，用于确认可用模型名称。
运行方式：python get_models.py
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY  = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_API_BASE", "https://api.ohmygpt.com/v1")


def list_models(api_key: str = API_KEY, base_url: str = BASE_URL):
    """打印指定 API 端点下所有可用模型的 ID 和所有者。"""
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"API: {base_url}")
    print("=" * 50)

    models = client.models.list()
    print(f"共 {len(models.data)} 个可用模型:\n")

    for i, model in enumerate(models.data, 1):
        owner = getattr(model, "owned_by", "unknown")
        print(f"  {i:2d}. {model.id}  (owned_by: {owner})")

    print("=" * 50)


if __name__ == "__main__":
    list_models()
