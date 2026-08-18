import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from rag_textbook_qa.cli import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MARKDOWN = """# 第1章 导论
这是用于命令行回归测试的章节导论，内容长度足以形成一个文本块。
## 1.1 基本概念
第一句介绍概念。第二句补充概念。第三句用于验证确定性分块。
"""


class IngestCliTests(unittest.TestCase):
    def run_cli(self, arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["--workspace", str(REPOSITORY_ROOT), *arguments])
        self.assertEqual(exit_code, 0)
        return output.getvalue()

    def test_clean_chunk_and_check_use_explicit_temp_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "教材.md"
            cleaned = root / "教材_cleaned.md"
            chunks = root / "教材_chunks.json"
            source.write_text(SAMPLE_MARKDOWN, encoding="utf-8")

            self.run_cli(["ingest", "clean", str(source), "--output", str(cleaned)])
            self.run_cli(
                [
                    "ingest",
                    "chunk",
                    str(cleaned),
                    "--output",
                    str(chunks),
                    "--min-chunk-size",
                    "10",
                    "--no-preview",
                ]
            )
            report_text = self.run_cli(
                ["ingest", "check", str(chunks), "--kind", "chunks", "--json"]
            )

            self.assertTrue(cleaned.is_file())
            self.assertTrue(chunks.is_file())
            self.assertGreater(len(json.loads(chunks.read_text(encoding="utf-8"))), 0)
            self.assertEqual(json.loads(report_text)["path"], str(chunks.resolve()))

    def test_batch_chunk_command_sorts_and_writes_to_separate_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir()
            (inputs / "b_cleaned.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")
            (inputs / "a_cleaned.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")

            message = self.run_cli(
                [
                    "ingest",
                    "chunk",
                    str(inputs),
                    "--output",
                    str(outputs),
                    "--batch",
                    "--min-chunk-size",
                    "10",
                    "--no-preview",
                ]
            )

            self.assertIn("新建 2", message)
            self.assertEqual(
                sorted(path.name for path in outputs.glob("*.json")),
                ["a_chunks.json", "b_chunks.json"],
            )

    def test_force_chunk_cannot_overwrite_its_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "教材.md"
            source.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
            original = source.read_bytes()
            error = io.StringIO()

            with contextlib.redirect_stderr(error):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "ingest",
                            "chunk",
                            str(source),
                            "--output",
                            str(source),
                            "--force",
                        ]
                    )

            self.assertEqual(raised.exception.code, 1)
            self.assertIn("不能是同一个文件", error.getvalue())
            self.assertEqual(source.read_bytes(), original)

    def test_ingest_does_not_require_a_discoverable_workspace(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "教材.md"
            source.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
            previous_directory = Path.cwd()
            output = io.StringIO()
            try:
                os.chdir(root)
                with contextlib.redirect_stdout(output):
                    exit_code = main(["ingest", "check", str(source), "--json"])
            finally:
                os.chdir(previous_directory)

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output.getvalue())["path"], str(source.resolve()))


if __name__ == "__main__":
    unittest.main()
