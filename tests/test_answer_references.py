import copy
import unittest

from rag_textbook_qa.rag.references import answer_body, render_source_sections

SOURCES = [
    {"citation_id": 7, "book_name": "computer_network", "chapter": "第1章",
     "section_h3": "1.7.3 五层协议", "section_h4": "实际小节"},
    {"citation_id": 2, "book_name": "computer_network", "section_h3": "1.7.5 TCP/IP"},
]


class AnswerReferenceTests(unittest.TestCase):
    def test_maps_only_body_citations_to_actual_chapters_without_mutating_inputs(self):
        answer = "正文【参考资料 7】\n\n## 参考章节\n1.7.5 TCP/IP【参考资料 7】\n"
        sources = copy.deepcopy(SOURCES)
        rendered = render_source_sections(answer, sources)
        self.assertIn("正文【参考资料 7】", rendered)
        self.assertIn("计算机网络：第1章 > 1.7.3 五层协议 > 实际小节", rendered)
        self.assertNotIn("TCP/IP", rendered)
        self.assertNotIn("【参考资料 2】", rendered)
        self.assertEqual(sources, SOURCES)
        self.assertEqual(render_source_sections(rendered, sources), rendered)

    def test_keeps_fenced_examples_and_following_answer_sections(self):
        for marker in ("```", "~~~~"):
            with self.subTest(marker=marker):
                code = f"{marker}text\n## 参考章节\n示例行\n{marker}\n"
                body = "正文【参考资料 7】\n\n" + code
                answer = body + "\n## 参考章节\n错误章节\n### 子项\n错误子项\n\n## 补充说明\n正文补充。"
                rendered = render_source_sections(answer, SOURCES)
                self.assertIn(code, rendered)
                self.assertIn("## 补充说明\n正文补充。", rendered)
                self.assertNotIn("错误章节", rendered)
                self.assertNotIn("错误子项", rendered)
                self.assertEqual(answer_body(rendered), answer_body(answer))

    def test_chapter_only_citations_do_not_become_body_evidence(self):
        answer = "正文没有引用。\n\n## 参考章节\n错误章节【参考资料 7】"
        self.assertEqual(render_source_sections(answer, SOURCES), "正文没有引用。")
        self.assertEqual(answer_body(answer), "正文没有引用。")

    def test_ambiguous_or_unknown_source_numbers_never_get_invented_chapters(self):
        for sources in ([*SOURCES, SOURCES[0]], [{**SOURCES[0], "citation_id": True}]):
            with self.subTest(sources=sources):
                answer = "正文【参考资料 7】【参考资料 99】\n\n## 参考章节\n模型章节"
                self.assertEqual(render_source_sections(answer, sources), "正文【参考资料 7】【参考资料 99】")

    def test_answers_without_a_reference_section_or_recorded_sources_remain_exact(self):
        for answer in ("简短回答【参考资料 7】\n", "```text\n## 参考章节\n示例\n```\n"):
            with self.subTest(answer=answer):
                self.assertEqual(render_source_sections(answer, SOURCES), answer)
        legacy = "答案\n\n## 参考章节\n历史章节"
        self.assertEqual(render_source_sections(legacy, []), legacy)


if __name__ == "__main__":
    unittest.main()
