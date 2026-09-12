import unittest
from unittest.mock import MagicMock

from rag_textbook_qa.rag.context import evidence_excerpt
from rag_textbook_qa.rag.engine import RAGEngine


class TableContextTests(unittest.TestCase):
    def test_spanning_headers_and_cells_preserve_column_positions(self):
        table = ('<table><tr><th rowspan="2">算法</th><th colspan="2">复杂度</th></tr>'
                 '<tr><th>时间</th><th>空间</th></tr>'
                 '<tr><td>A&amp;B</td><td>O(n)</td><td>O(1)</td></tr></table>')
        text, omitted, compacted = evidence_excerpt(table, 1000)
        self.assertIn('行1: 算法 | 复杂度 | 复杂度', text)
        self.assertIn('行2: 算法 | 时间 | 空间', text)
        self.assertIn('行3: A&B | O(n) | O(1)', text)
        self.assertFalse(omitted)
        self.assertTrue(compacted)

    def test_budget_never_cuts_table_cells(self):
        table = '<table><tr><th>名称</th></tr><tr><td>' + '长值' * 100 + '</td></tr></table>'
        text, omitted, _ = evidence_excerpt(table, 70)
        self.assertLessEqual(len(text), 70)
        self.assertIn('行1: 名称', text)
        self.assertNotIn('长值', text)
        self.assertTrue(omitted)

    def test_unsupported_table_is_skipped_instead_of_losing_caption(self):
        table = '<table><caption>单位：万元</caption><tr><td>10</td></tr></table>'
        text, omitted, compacted = evidence_excerpt(table, 200)
        self.assertEqual(text, '')
        self.assertTrue(omitted)
        self.assertFalse(compacted)

    def test_next_source_used_when_first_table_cannot_fit(self):
        table = '<table><tr><td>' + '甲' * 1000 + '</td></tr></table>'
        results = [{'book_name': 'os', 'content': table},
                   {'book_name': 'os', 'content': '后续完整证据'}]
        text, sources = RAGEngine.select_context(results, 100)
        self.assertEqual(len(sources), 1)
        self.assertIn('后续完整证据', text)
        self.assertEqual(sources[0]['citation_id'], 1)

    def test_packed_evidence_matches_source_and_does_not_mutate_retrieval(self):
        table = '<table><tr><th>术语</th><th>值</th></tr><tr><td>缓存</td><td>快</td></tr></table>'
        result = {'book_name': 'os', 'content': table}
        text, sources = RAGEngine.select_context([result], 300)
        self.assertEqual(result['content'], table)
        self.assertEqual(text, ''.join(source['context_text'] for source in sources))
        self.assertIn('缓存 | 快', sources[0]['content'])
        self.assertTrue(sources[0]['table_compacted'])
        self.assertFalse(sources[0]['truncated'])

    def test_no_evidence_does_not_call_llm(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.verbose = False
        engine.compute_settings = MagicMock(backend="local")
        engine.vectorizer = MagicMock()
        engine.reranker = None
        engine.llm = MagicMock()
        engine.search_single_book = MagicMock(return_value=[{
            "book_name": "os",
            "content": "<table><caption>不可省略的单位</caption><tr><td>10</td></tr></table>",
        }])
        result = engine.ask("数值是多少", book_name="os")
        self.assertFalse(result["success"])
        self.assertEqual(result["context_sources"], [])
        engine.llm.generate_answer.assert_not_called()

    def test_prose_still_respects_existing_budget(self):
        self.assertEqual(evidence_excerpt('abcdef', 3), ('abc', True, False))


if __name__ == '__main__':
    unittest.main()
