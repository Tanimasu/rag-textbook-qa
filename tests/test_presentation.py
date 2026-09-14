import unittest

from rag_textbook_qa.rag.presentation import render_search_results


class PresentationTests(unittest.TestCase):
    def test_no_results_says_so(self):
        self.assertEqual(render_search_results([]), "没有找到相关内容")

    def test_rendered_chunk_shows_score_headings_and_shortened_content(self):
        rendered = render_search_results(
            [
                {
                    "similarity": 0.8231,
                    "method": "hybrid",
                    "book_name": "os",
                    "chapter": "第三章",
                    "section_h2": "3.5 死锁",
                    "content": "内容" * 200,
                    "has_code": True,
                }
            ]
        )

        self.assertIn("找到 1 条相关内容", rendered)
        self.assertIn("相似度: 0.8231 | 方法: hybrid", rendered)
        self.assertIn("章节: 第三章 | 3.5 死锁", rendered)
        self.assertIn("标签: 含代码", rendered)
        self.assertIn("...", rendered)
        self.assertNotIn("内容" * 200, rendered)


if __name__ == "__main__":
    unittest.main()
