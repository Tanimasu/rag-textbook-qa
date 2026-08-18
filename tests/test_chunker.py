import contextlib
import io
import json
import tempfile
import unittest
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

    def test_existing_output_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample_cleaned.md"
            output = root / "sample_chunks.json"
            source.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
            output.write_text("preserved", encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(
                FileExistsError
            ):
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
