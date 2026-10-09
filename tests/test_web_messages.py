import unittest

from rag_textbook_qa.web.messages import answer_export_markdown, answer_message, compute_trace_items


class WebMessageTests(unittest.TestCase):
    def test_export_preserves_actual_source_numbers_full_text_and_fence_boundaries(self):
        excerpt = '正文前半\n```python\nprint("值")\n```\n末尾证据 <原样>'
        exported = answer_export_markdown('为什么？', '答案【参考资料 7】', [
            {"citation_id": 7, "book_name": "computer_organization", "content": excerpt,
             "section_h3": "4.5.8 写入策略", "section_h4": "1.写回法",
             "truncated": True, "table_compacted": True,
             "private_metadata": "do-not-export", "rank": 11}
        ])
        self.assertIn('答案【参考资料 7】', exported)
        self.assertIn('参考资料 7 · 计算机组成原理', exported)
        self.assertIn('4.5.8 写入策略 > 1.写回法', exported)
        self.assertIn('````text\n'+excerpt+'\n````', exported)
        self.assertIn('片段已截断。', exported)
        self.assertIn('表格按行整理。', exported)
        self.assertNotIn('do-not-export', exported)

    def test_answer_is_preferred_when_generation_succeeds(self):
        self.assertEqual(
            answer_message({"answer": "教材答案", "error": None}),
            "教材答案",
        )

    def test_generation_error_is_shown_instead_of_generic_fallback(self):
        message = answer_message(
            {
                "answer": None,
                "error": "LLM 不可用：必须设置 LLM_API_KEY",
                "success": False,
            }
        )

        self.assertIn("未能生成答案", message)
        self.assertIn("LLM_API_KEY", message)
        self.assertNotEqual(message, "抱歉，未能生成答案。")

    def test_very_long_provider_errors_are_bounded(self):
        message = answer_message(
            {
                "answer": "第三方客户端生成的冗长错误",
                "error": "x" * 1000,
                "success": False,
            }
        )

        self.assertLessEqual(len(message), 510)
        self.assertTrue(message.endswith("..."))

    def test_compute_trace_labels_remote_cuda_and_local_mps_fallback(self):
        remote = compute_trace_items(
            {
                "embedding": {
                    "backend": "remote",
                    "device": "cuda",
                    "platform": "Windows",
                    "elapsed_seconds": 0.14,
                    "calls": 1,
                    "fallback_used": False,
                },
                "reranker": {
                    "backend": "local",
                    "device": "mps",
                    "platform": "Darwin",
                    "elapsed_seconds": 1.23,
                    "calls": 1,
                    "fallback_used": True,
                },
                "retrieval_seconds": 1.5,
                "generation_seconds": 2,
                "first_token_seconds": 0.4,
                "total_seconds": 3.5,
            }
        )

        self.assertIn("远程 Worker（Windows） · CUDA", remote[0]["text"])
        self.assertIn("已回退到本地（macOS） · MPS", remote[1]["text"])
        self.assertEqual(remote[1]["kind"], "fallback")
        self.assertIn("首字 0.400 秒", remote[2]["text"])
        self.assertIn("总计 3.500 秒", remote[2]["text"])

    def test_compute_trace_is_empty_for_legacy_messages(self):
        self.assertEqual(compute_trace_items(None), [])


if __name__ == "__main__":
    unittest.main()
