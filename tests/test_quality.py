import json
import tempfile
import unittest
from pathlib import Path

from rag_textbook_qa.ingestion.quality import analyze_chunks, analyze_markdown


class QualityAnalysisTests(unittest.TestCase):
    def test_markdown_analysis_returns_legacy_statistics_and_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "教材.md"
            path.write_text("# 第一章\n\n## 1.1 标题\n<!-- image -->\n正文", encoding="utf-8")
            report = analyze_markdown(path)

        self.assertEqual(report["stats"]["二级标题（##）"], 1)
        self.assertEqual(report["stats"]["图片占位符"], 1)
        self.assertEqual(len(report["issues"]), 2)

    def test_chunk_analysis_preserves_legacy_grading_rules(self):
        chunks = [
            {
                "chunk_id": "small",
                "content": "短内容",
                "char_count": 3,
                "has_code": False,
            },
            {
                "chunk_id": "large",
                "content": "长" * 2001,
                "char_count": 2001,
                "has_code": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.json"
            path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            report = analyze_chunks(path)

        self.assertEqual(report["total"], 2)
        self.assertEqual(report["issues"]["too_small"], ["small"])
        self.assertEqual(report["issues"]["too_large"], ["large"])
        self.assertEqual(report["grade"], "poor")

    def test_chunk_analysis_detects_hierarchy_mismatch_and_duplicate_content(self):
        chunks = [
            {
                "chunk_id": "wrong-parent",
                "chapter": "第3章",
                "section_h2": "3.4 栈与递归",
                "section_h3": "3.5.2 循环队列",
                "content": "循环队列用于解决假溢出。" * 5,
                "char_count": 65,
                "has_code": False,
            },
            {
                "chunk_id": "duplicate",
                "chapter": "第3章",
                "section_h2": "3.5 队列",
                "section_h3": "3.5.2 循环队列",
                "content": "循环队列用于解决假溢出。" * 5,
                "char_count": 65,
                "has_code": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.json"
            path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            report = analyze_chunks(path)

        self.assertEqual(report["issues"]["hierarchy_mismatch"], ["wrong-parent"])
        self.assertEqual(report["issues"]["duplicate_content"], ["duplicate"])


if __name__ == "__main__":
    unittest.main()
