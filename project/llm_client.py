# llm_client.py
"""
大语言模型调用客户端
支持 OpenAI 兼容的 API
"""

import os
import time
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_API_KEY   = os.getenv("LLM_API_KEY", "")
_API_BASE  = os.getenv("LLM_API_BASE", "https://api.ohmygpt.com/v1")
_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview")


class LLMClient:
    """大语言模型客户端"""

    def __init__(
            self,
            api_key: str,
            base_url: str,
            model: str = "gpt-5.1-chat-latest",
            verbose: bool = True
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥
            base_url: API 地址（应包含 /v1 或 /v1/）
            model: 默认使用的模型
            verbose: 是否显示详细日志
        """
        # 确保 base_url 格式正确
        if not base_url.endswith('/'):
            base_url = base_url + '/'

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.default_model = model
        self.verbose = verbose

        if self.verbose:
            print(f"🤖 LLM 客户端初始化:")
            print(f"   Base URL: {base_url}")
            print(f"   默认模型: {model}")

    def generate_answer(
            self,
            prompt: str,
            model: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 2000,
            retry: int = 2
    ) -> Dict:
        """
        生成答案

        Args:
            prompt: 完整的提示词
            model: 使用的模型（None 则用默认）
            temperature: 温度参数（0-1，越低越确定）
            max_tokens: 最大生成长度
            retry: 失败重试次数

        Returns:
            包含答案和元信息的字典
        """

        if model is None:
            model = self.default_model

        last_error = None

        for attempt in range(retry + 1):
            try:
                if self.verbose:
                    if attempt > 0:
                        print(f"   ⚠️  重试 ({attempt}/{retry})...")
                    else:
                        print(f"\n🔄 调用 LLM API...")
                        print(f"   模型: {model}")
                        print(f"   温度: {temperature}")
                        print(f"   最大 Tokens: {max_tokens}")

                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )

                elapsed_time = time.time() - start_time

                # 提取结果
                answer = response.choices[0].message.content

                if self.verbose:
                    print(f"   ✅ 成功！（{round(elapsed_time, 2)}秒）")
                    print(f"   Token 使用: {response.usage.total_tokens} "
                          f"(输入:{response.usage.prompt_tokens} + "
                          f"输出:{response.usage.completion_tokens})")

                return {
                    "success": True,
                    "answer": answer,
                    "model": model,
                    "tokens": {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    },
                    "time": round(elapsed_time, 2),
                    "finish_reason": response.choices[0].finish_reason
                }

            except Exception as e:
                last_error = e
                error_msg = str(e)

                if self.verbose:
                    print(f"   ❌ 失败: {error_msg[:100]}")

                if attempt < retry:
                    time.sleep(1)
                    continue
                else:
                    break

        # 所有尝试都失败了
        if self.verbose:
            print(f"   ❌ 全部尝试失败")

        return {
            "success": False,
            "error": str(last_error),
            "answer": f"❌ 生成答案时出错：{str(last_error)}",
            "model": model,
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "time": 0
        }

    def stream_answer(
            self,
            prompt: str,
            model: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 2000
    ):
        """
        流式生成答案（适合实时显示）

        Args:
            prompt: 完整的提示词
            model: 使用的模型
            temperature: 温度参数
            max_tokens: 最大生成长度

        Yields:
            生成的文本片段
        """
        if model is None:
            model = self.default_model

        try:
            if self.verbose:
                print(f"\n🔄 流式调用 LLM API...")
                print(f"   模型: {model}\n")

            stream = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            for chunk in stream:
                # ⭐ 关键修复：检查 choices 是否存在且非空
                if hasattr(chunk, 'choices') and chunk.choices:
                    if len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield delta.content

        except Exception as e:
            error_msg = f"\n\n❌ 流式生成错误：{str(e)}"
            if self.verbose:
                print(error_msg)
            yield error_msg

    def chat(
            self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 2000
    ) -> Dict:
        """
        多轮对话接口

        Args:
            messages: 对话历史 [{"role": "user/assistant/system", "content": "..."}]
            model: 模型名称
            temperature: 温度
            max_tokens: 最大 tokens

        Returns:
            包含答案的字典
        """
        if model is None:
            model = self.default_model

        try:
            if self.verbose:
                print(f"\n🔄 多轮对话...")
                print(f"   模型: {model}")
                print(f"   消息数: {len(messages)}")

            start_time = time.time()

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            elapsed_time = time.time() - start_time
            answer = response.choices[0].message.content

            if self.verbose:
                print(f"   ✅ 成功！（{round(elapsed_time, 2)}秒）")

            return {
                "success": True,
                "answer": answer,
                "model": model,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "time": round(elapsed_time, 2)
            }

        except Exception as e:
            error_msg = str(e)
            if self.verbose:
                print(f"   ❌ 失败: {error_msg[:100]}")

            return {
                "success": False,
                "error": error_msg,
                "answer": f"❌ 对话失败：{error_msg}",
                "model": model,
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "time": 0
            }


def create_llm_client(
        api_key: str = _API_KEY,
        base_url: str = _API_BASE,
        model: str = _LLM_MODEL,
        verbose: bool = True
) -> LLMClient:
    """
    创建 LLM 客户端的便捷函数

    Args:
        api_key: API 密钥
        base_url: API 地址
        model: 默认模型
        verbose: 是否显示详细日志

    Returns:
        LLM 客户端实例

    ⚠️ 注意：请不要在公开场合分享你的 API 密钥！
    """
    return LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        verbose=verbose
    )


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 测试 LLM 客户端")
    print("=" * 70)

    # 创建客户端
    client = create_llm_client()

    # 测试 1：普通生成
    print("\n【测试 1】普通生成")
    print("-" * 70)
    result = client.generate_answer(
        "请用一句话解释什么是进程？",
        max_tokens=100
    )

    if result["success"]:
        print(f"\n✅ 成功！")
        print(f"\n💬 回答：\n{result['answer']}")
        print(f"\n📊 统计：")
        print(f"   - Tokens: {result['tokens']['total']}")
        print(f"   - 耗时: {result['time']}秒")
    else:
        print(f"\n❌ 失败：{result['error']}")

    # 测试 2：流式生成
    print("\n\n【测试 2】流式生成")
    print("-" * 70)
    print("\n💬 AI: ", end="", flush=True)

    full_text = ""
    try:
        for chunk in client.stream_answer(
                "用三句话介绍什么是死锁？",
                max_tokens=150
        ):
            print(chunk, end="", flush=True)
            full_text += chunk

        print()  # 换行

        if full_text:
            print(f"\n✅ 流式生成成功！共 {len(full_text)} 字符")
        else:
            print(f"\n⚠️  流式生成完成，但没有内容")

    except Exception as e:
        print(f"\n❌ 流式生成异常: {e}")

    # 测试 3：多轮对话
    print("\n\n【测试 3】多轮对话")
    print("-" * 70)
    messages = [
        {"role": "system", "content": "你是一位操作系统专家，请用简洁的语言回答。"},
        {"role": "user", "content": "进程和线程的主要区别是什么？"}  # ⭐ 改为更明确的问题
    ]
    result = client.chat(messages, max_tokens=150)  # ⭐ 增加 max_tokens
    if result["success"]:
        print(f"\n✅ 成功！")
        answer = result['answer']

        # ⭐ 添加内容检查
        if answer and answer.strip():
            print(f"\n💬 AI: {answer}")
        else:
            print(f"\n⚠️  API 返回成功但内容为空")
            print(f"   原始回答: '{answer}'")

        print(f"\n📊 统计：")
        print(f"   - Tokens: {result['tokens']['total']}")
        print(f"   - 耗时: {result['time']}秒")
    else:
        print(f"\n❌ 失败：{result['error']}")
