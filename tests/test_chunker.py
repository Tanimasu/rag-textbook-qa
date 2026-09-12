import contextlib
import io
import json
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

from rag_textbook_qa.ingestion.chunker import (
    SmartTextbookChunker,
    batch_chunk_markdown,
    chunk_markdown,
)

SAMPLE_MARKDOWN = """# 第1章 导论
这是第一章的导论内容，它用来验证章节上下文。
## 1.1 基本概念
第一句介绍基本概念。第二句继续补充概念。第三句用来触发长文分割。
### 1.1.1 示例
这里包含代码 ```python``` 与[图片]占位符，用于验证特殊字段。
"""


class ChunkerTests(unittest.TestCase):
    def test_chunk_markdown_schema_and_heading_context(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample_cleaned.md"
            output = root / "sample_chunks.json"
            source.write_text(SAMPLE_MARKDOWN, encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                chunks = chunk_markdown(
                    source,
                    output,
                    max_chunk_size=45,
                    min_chunk_size=10,
                    overlap_size=5,
                )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), len(chunks))
            self.assertGreater(len(payload), 1)
            self.assertEqual(
                set(payload[0]),
                {
                    "chunk_id",
                    "chapter",
                    "section_h2",
                    "section_h3",
                    "section_h4",
                    "content",
                    "level",
                    "char_count",
                    "has_code",
                    "has_image",
                },
            )
            self.assertEqual(payload[0]["chapter"], "第1章 导论")
            self.assertTrue(any(chunk["section_h2"] == "1.1 基本概念" for chunk in payload))
            self.assertTrue(any(chunk["has_code"] for chunk in payload))
            self.assertTrue(any(chunk["has_image"] for chunk in payload))
            self.assertTrue(SmartTextbookChunker.preview_path(output).is_file())

    def test_output_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample_cleaned.md"
            first = root / "first.json"
            second = root / "second.json"
            source.write_text(SAMPLE_MARKDOWN, encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                chunk_markdown(source, first, max_chunk_size=45, min_chunk_size=10)
                chunk_markdown(source, second, max_chunk_size=45, min_chunk_size=10)

            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_heading_without_direct_body_still_updates_descendant_context(self):
        markdown = """# 第7章 视图和索引
## 7.1 视图
### 7.1.1 视图概述
视图是在基本表之上定义的虚拟表，这段内容足够长，可以单独形成一个测试文本块。
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "headings.md"
            source.write_text(markdown, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                chunks = SmartTextbookChunker(min_chunk_size=10).chunk_document(source)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chapter, "第7章 视图和索引")
        self.assertEqual(chunks[0].section_h2, "7.1 视图")
        self.assertEqual(chunks[0].section_h3, "7.1.1 视图概述")
        self.assertTrue(chunks[0].chunk_id.startswith("ch7_s7_1_"))

    def test_numbered_section_repairs_missing_chapter_heading(self):
        markdown = """# 第1章 导论
第一章正文足够长，用来建立初始上下文并生成一个独立的文本块。
第2章线性表
## 2.1 线性表定义
第二章标题未被解析成 Markdown，但二级标题编号仍应纠正章节上下文。
### 2.2.1 错位的子标题
三级标题编号与二级标题冲突时，也应纠正它的二级父级编号。
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "missing-chapter.md"
            source.write_text(markdown, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                chunks = SmartTextbookChunker(min_chunk_size=10).chunk_document(source)

        second_chapter = [chunk for chunk in chunks if chunk.chapter == "第2章"]
        self.assertEqual(len(second_chapter), 2)
        self.assertEqual(second_chapter[0].section_h2, "2.1 线性表定义")
        self.assertEqual(second_chapter[1].section_h2, "2.2")
        self.assertEqual(second_chapter[1].section_h3, "2.2.1 错位的子标题")

    def test_non_numbered_h3_does_not_replace_numbered_parent_section(self):
        markdown = """# 第3章 栈和队列
## 3.5 队列
### 3.5.2 循环队列
循环队列正文用于建立编号三级标题，它包含足够字符以形成文本块。
### define MAXQSIZE 100
这里原本是代码预处理指令，被解析器误识别成了同级标题。
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "false-heading.md"
            source.write_text(markdown, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                chunks = SmartTextbookChunker(min_chunk_size=10).chunk_document(source)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[1].section_h2, "3.5 队列")
        self.assertEqual(chunks[1].section_h3, "3.5.2 循环队列")
        self.assertEqual(chunks[1].section_h4, "define MAXQSIZE 100")

    def test_prose_overlap_preserves_long_unpunctuated_text(self):
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        chunker = SmartTextbookChunker(max_chunk_size=12, min_chunk_size=2, overlap_size=3)
        chunks = chunker.split_long_content(text, 1)
        restored = chunks[0].content
        for previous, current in pairwise(chunks):
            self.assertEqual(previous.content[-3:], current.content[:3])
            restored += current.content[3:]
        self.assertEqual(restored, text)
        self.assertTrue(all(chunk.char_count <= 12 for chunk in chunks))

    def test_long_section_with_a_formula_splits_around_it(self):
        prose = "正文内容。" * 200
        markdown = f"# 第1章\n## 1.1 推导\n{prose}\n$$\nE = mc^2\n$$\n{prose}"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "formula.md"
            source.write_text(markdown, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                chunks = SmartTextbookChunker().chunk_document(source)
        self.assertGreater(len(chunks), 1)
        holder = [chunk for chunk in chunks if "E = mc^2" in chunk.content]
        self.assertEqual(len(holder), 1)
        self.assertEqual(holder[0].content.count("$$"), 2)
        self.assertTrue(all(chunk.char_count <= 800 for chunk in chunks))

    def test_oversized_table_ships_whole_while_its_prose_splits(self):
        table = "<table>" + "<tr><td>单元格</td></tr>" * 200 + "</table>"
        prose = "正文内容。" * 200
        markdown = f"# 第1章\n## 1.1 表格\n{prose}\n{table}\n{prose}"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "table.md"
            source.write_text(markdown, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                chunks = SmartTextbookChunker().chunk_document(source)
        tables = [chunk for chunk in chunks if "<table>" in chunk.content]
        self.assertEqual(len(tables), 1)
        self.assertTrue(tables[0].content.endswith("</table>"))
        prose_chunks = [chunk for chunk in chunks if "<table>" not in chunk.content]
        self.assertGreater(len(prose_chunks), 1)
        self.assertTrue(all(chunk.char_count <= 800 for chunk in prose_chunks))

    def test_char_count_matches_stored_content_around_blank_lines(self):
        markdown = "# 第1章\n\n## 1.1 概念\n\n\n概念正文。\n\n\n## 1.2 细节\n\n" + "细节" * 500
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "blank.md"
            source.write_text(markdown, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                chunks = SmartTextbookChunker(min_chunk_size=4).chunk_document(source)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.char_count, len(chunk.content))
            self.assertEqual(chunk.content, chunk.content.strip())

    def test_short_sections_are_preserved_without_cross_heading_merge(self):
        markdown = "# 第1章\n## 1.1 简介\n短定义\n## 1.2 正文\n" + "正文" * 60
        markdown += "\n# 第2章\n## 2.1 简介\n第二章短定义"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "short.md"
            source.write_text(markdown, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                chunks = SmartTextbookChunker().chunk_document(source)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].content, "短定义")
        self.assertEqual(chunks[2].chapter, "第2章")
        self.assertEqual(chunks[2].content, "第二章短定义")

    def test_fenced_code_preserves_comments_blank_lines_and_boundaries(self):
        code = "```python\n# comment\n\nx = 1\n" + "print(x)\n" * 30 + "```"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "code.md"
            source.write_text("# 第1章\n" + code, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                chunks = SmartTextbookChunker(max_chunk_size=80, min_chunk_size=10).chunk_document(
                    source
                )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chapter, "第1章")
        self.assertEqual(chunks[0].content, code)

    def test_inline_tables_remain_whole_including_cell_formulas(self):
        for opening, closing in (("<table>", "</table>"), ("<TABLE class='x'>", "</TABLE>")):
            with self.subTest(opening=opening):
                table = opening + "<tr><td>$$x$$" + "甲" * 100 + "</td></tr>\n" + closing
                text = "说明文字 " + table + " 完成\n" + "后续正文。" * 40
                chunker = SmartTextbookChunker(
                    max_chunk_size=40, min_chunk_size=1, overlap_size=0
                )
                chunks = chunker.split_section(text, 2)
                containing = [chunk for chunk in chunks if opening in chunk.content]
                self.assertEqual(len(containing), 1)
                self.assertIn(table, containing[0].content)
                self.assertTrue(all(chunk.char_count <= 40 for chunk in chunks[1:]))
                self.assertEqual(
                    "".join("".join(chunk.content.split()) for chunk in chunks),
                    "".join(text.split()),
                )

    def test_existing_output_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample_cleaned.md"
            output = root / "sample_chunks.json"
            source.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
            output.write_text("preserved", encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                chunk_markdown(source, output)

            self.assertEqual(output.read_text(encoding="utf-8"), "preserved")

    def test_force_cannot_overwrite_the_source_markdown(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "sample.md"
            source.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
            original = source.read_bytes()

            with self.assertRaises(ValueError):
                chunk_markdown(source, source, overwrite=True)

            self.assertEqual(source.read_bytes(), original)

    def test_invalid_sizes_are_rejected(self):
        invalid_options = (
            {"max_chunk_size": 0},
            {"min_chunk_size": 0},
            {"max_chunk_size": 10, "min_chunk_size": 11},
            {"overlap_size": -1},
        )
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                SmartTextbookChunker(**options)

    def test_batch_api_rejects_an_empty_input_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            with self.assertRaises(FileNotFoundError):
                batch_chunk_markdown(input_dir, output_dir)

            self.assertFalse(output_dir.exists())

    def test_batch_api_sorts_inputs_and_skips_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "b_cleaned.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")
            (input_dir / "a_cleaned.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                first = batch_chunk_markdown(
                    input_dir,
                    output_dir,
                    max_chunk_size=45,
                    min_chunk_size=10,
                    write_preview=False,
                )
                second = batch_chunk_markdown(
                    input_dir,
                    output_dir,
                    max_chunk_size=45,
                    min_chunk_size=10,
                    write_preview=False,
                )

            self.assertEqual(
                [path.name for path in first.created],
                ["a_chunks.json", "b_chunks.json"],
            )
            self.assertEqual(second.created, ())
            self.assertEqual(
                [path.name for path in second.skipped_existing],
                ["a_chunks.json", "b_chunks.json"],
            )


if __name__ == "__main__":
    unittest.main()
