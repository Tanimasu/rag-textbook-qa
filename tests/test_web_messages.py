import unittest

from rag_textbook_qa.web.messages import answer_export_markdown, answer_message, compute_trace_items


class WebMessageTests(unittest.TestCase):
    def test_display_and_export_reference_chapters_match_recorded_sources(self):
        sources = [{"citation_id": 7, "book_name": "computer_network", "content": "原文。",
                    "section_h3": "1.7.3 五层协议"}]
        raw = "正文【参考资料 7】\n\n## 参考章节\n1.7.5 错误章节【参考资料 7】"
        result = {"answer": raw, "success": True, "context_sources": sources}
        displayed = answer_message(result)
        exported = answer_export_markdown("问题", raw, sources)
        self.assertIn("1.7.3 五层协议", displayed)
        self.assertNotIn("错误章节", displayed)
        self.assertIn(displayed, exported)
        self.assertEqual(result["answer"], raw)

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

    def test_invalid_measurements_are_unknown_and_cannot_crash_trace(self):
        for value in (float("inf"), float("-inf"), float("nan"), True, -1, None, "bad", 10**400):
            with self.subTest(value=value):
                execution = {"embedding": {"elapsed_seconds": value, "calls": value},
                             "retrieval_seconds": value, "generation_seconds": value,
                             "first_token_seconds": value, "total_seconds": value}
                items = compute_trace_items(execution)
                self.assertIn("耗时未知", items[0]["text"])
                self.assertNotIn(" 次", items[0]["text"])
                self.assertIn("总计 耗时未知", items[-1]["text"])
                self.assertNotIn("0.000 秒", items[-1]["text"])
                self.assertIs(execution["total_seconds"], value)
        valid = compute_trace_items({"embedding": {"elapsed_seconds": 0, "calls": 3},
                                     "total_seconds": "1.25"})
        self.assertIn("0.000 秒 · 3 次", valid[0]["text"])
        self.assertIn("总计 1.250 秒", valid[-1]["text"])
        fractional = compute_trace_items({"embedding": {"elapsed_seconds": 0, "calls": 2.5}})
        self.assertNotIn("2 次", fractional[0]["text"])


if __name__ == "__main__":
    unittest.main()
