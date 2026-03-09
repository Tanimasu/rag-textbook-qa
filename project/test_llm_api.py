"""
test_llm_api.py
测试 LLM API 连通性，分两种模式：
  - quick_test(): 用已知可用的配置快速验证 API Key 和模型是否正常
  - discover_endpoint(): 遍历多个候选配置，找出第一个可用的端点（调试用）
运行方式：python test_llm_api.py
"""
import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 当前使用的配置（quick_test 用这里的值）
API_KEY  = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_API_BASE", "https://api.ohmygpt.com/v1")
MODEL    = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview")

# 候选端点列表（discover_endpoint 用这里的值）
CANDIDATE_CONFIGS = [
    {"name": "ohmygpt + grok",          "base_url": "https://api.ohmygpt.com",    "model": "grok-4-1-fast-non-reasoning"},
    {"name": "nloli + claude-sonnet-4",  "base_url": "https://chatapi.nloli.xyz/v1","model": "anthropic/claude-sonnet-4"},
    {"name": "nloli + gpt-4",            "base_url": "https://chatapi.nloli.xyz/v1","model": "gpt-4"},
    {"name": "nloli + gpt-5.1",          "base_url": "https://chatapi.nloli.xyz/v1","model": "gpt-5.1-chat-latest"},
]
TEST_PROMPT = "请只回复"测试成功"这四个字，不要加其他内容。"


# -----------------------------------------------------------------------
# 模式1：快速验证（已知可用配置）
# -----------------------------------------------------------------------
def quick_test(api_key: str = API_KEY, base_url: str = BASE_URL, model: str = MODEL):
    """用已知配置发一条简单请求，验证 API Key 和模型是否正常响应。"""
    client = OpenAI(api_key=api_key, base_url=base_url)

    print("=" * 60)
    print("快速连通性测试")
    print(f"  Base URL: {base_url}")
    print(f"  Model:    {model}")
    print("=" * 60)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一位专业的计算机科学导师。"},
                {"role": "user",   "content": "请用一句话解释什么是进程？"},
            ],
            temperature=0.7,
            max_tokens=200,
        )

        answer = response.choices[0].message.content
        usage  = response.usage

        print(f"\nAI 回答: {answer}")
        print(f"\nToken 用量:")
        print(f"  输入: {usage.prompt_tokens}")
        print(f"  输出: {usage.completion_tokens}")
        print(f"  合计: {usage.total_tokens}")
        print(f"\n模型: {response.model}  |  完成原因: {response.choices[0].finish_reason}")
        print("\n测试通过，配置可用。")

    except Exception as e:
        print(f"\n测试失败: {type(e).__name__}: {e}")


# -----------------------------------------------------------------------
# 模式2：端点探测（用于调试/更换 API 服务时）
# -----------------------------------------------------------------------
def discover_endpoint(api_key: str = API_KEY):
    """
    遍历 CANDIDATE_CONFIGS，找到第一个返回合法 JSON 响应的端点。
    找到后打印可用配置并停止，全部失败则输出建议。
    """
    print("=" * 60)
    print("端点探测模式")
    print(f"测试 {len(CANDIDATE_CONFIGS)} 个候选配置")
    print("=" * 60)

    for cfg in CANDIDATE_CONFIGS:
        url = cfg["base_url"].rstrip("/") + "/chat/completions"
        print(f"\n尝试: {cfg['name']}")
        print(f"  URL:   {url}")
        print(f"  Model: {cfg['model']}")

        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg["model"],
                    "messages": [{"role": "user", "content": TEST_PROMPT}],
                    "max_tokens": 50,
                },
                timeout=30,
            )

            print(f"  HTTP: {response.status_code}")

            if response.status_code != 200:
                try:
                    err = response.json()
                    print(f"  错误: {err.get('error', {}).get('message', response.text[:200])}")
                except Exception:
                    print(f"  错误: {response.text[:200]}")
                continue

            # 排除返回 HTML 的错误页面
            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type or response.text.strip().startswith("<!"):
                print("  返回 HTML 页面（端点路径可能不对）")
                continue

            data = response.json()
            answer = None
            if "choices" in data and data["choices"]:
                answer = data["choices"][0]["message"]["content"]

            if answer:
                print(f"  回复: {answer}")
                print(f"\n找到可用配置:")
                print(f"  BASE_URL = \"{cfg['base_url']}\"")
                print(f"  MODEL    = \"{cfg['model']}\"")
                return

            print(f"  响应无内容: {json.dumps(data, ensure_ascii=False)[:200]}")

        except Exception as e:
            print(f"  请求异常: {e}")

    print("\n所有配置均失败。建议检查：")
    print("  1. API Key 是否有效")
    print("  2. 服务商文档中的端点路径是否有变化")
    print("  3. 账户余额是否充足")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    # 默认运行快速测试；如果需要探测端点，改为调用 discover_endpoint()
    quick_test()
    # discover_endpoint()
