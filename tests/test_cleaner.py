import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rag_textbook_qa.ingestion.cleaner import SmartMarkdownCleaner, clean_markdown

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPATIBILITY_SCRIPT = REPOSITORY_ROOT / "project" / "clean_markdown.py"


class SmartMarkdownCleanerTests(unittest.TestCase):
    def test_title_normalization_preserves_fenced_directives(self):
        cleaner = SmartMarkdownCleaner()
        blocks = [
            "```c\n#include <stdio.h>\n#define LIMIT 10\n#if LIMIT\n#endif\n```",
            "   ~~~~c\n#define LIMIT 10\n~~~\n# retained\n    ~~~~\n~~~~ trailing\n~~~~~",
            "````c\n```\n# retained\n~~~~\n````",
            "```c\n#define LIMIT 10\n# unclosed",
        ]
        for block in blocks:
            with self.subTest(block=block):
                self.assertEqual(cleaner.normalize_titles(block), block)

        content = "```c\n#define LIMIT 10\n```\n# 1.2 正文标题\n"
        self.assertEqual(
            cleaner.normalize_titles(content),
            "```c\n#define LIMIT 10\n```\n## 1.2 正文标题\n",
        )

    def test_clean_markdown_preserves_fenced_preprocessor_directives(self):
        source = "# 第1章 测试章节\n\n```c\n#include <stdio.h>\n#define LIMIT 10\n#if LIMIT\n#endif\n```\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.md"
            output_path = root / "output.md"
            input_path.write_text(source, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                result = clean_markdown(input_path, output_path)
            for directive in ("#include <stdio.h>", "#define LIMIT 10", "#if LIMIT", "#endif"):
                self.assertIn(directive, result.splitlines())
            self.assertEqual(input_path.read_text(encoding="utf-8"), source)

    def test_title_levels_preserve_legacy_rules(self):
        cleaner = SmartMarkdownCleaner()

        cases = {
            "第1章 操作系统引论": (1, "第1章 操作系统引论"),
            "1.2 进程": (2, "1.2 进程"),
            "1.2.3 调度": (3, "1.2.3 调度"),
            "1.方便性": (4, "1.方便性"),
            "int main()": (0, "int main()"),
        }
        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(cleaner.detect_title_level(title), expected)

    def test_clean_markdown_uses_explicit_temp_paths(self):
        source = """## 第1章
## 操作系统引论

<!-- image -->

#### 1．方便性

① 支持 A&amp;B 和 value\\_name。

```c
if（ready），return；
```
"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.md"
            output_path = root / "output.md"
            input_path.write_text(source, encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                result = clean_markdown(input_path, output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), result)
            self.assertTrue(result.endswith("\n"))
            self.assertIn("# 第1章 操作系统引论", result)
            self.assertIn("> 📷 **[图片]**", result)
            self.assertIn("#### 1.方便性", result)
            self.assertIn("(1) 支持 A&B 和 value_name。", result)
            self.assertIn("if(ready),return;", result)

    def test_compatibility_script_without_paths_only_prints_usage(self):
        completed = subprocess.run(
            [sys.executable, str(COMPATIBILITY_SCRIPT)],
            cwd=tempfile.gettempdir(),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("usage:", completed.stderr)

    def test_compatibility_script_rejects_in_place_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "book.md"
            original = "# 第1章 测试\n"
            input_path.write_text(original, encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(COMPATIBILITY_SCRIPT), str(input_path), str(input_path)],
                cwd=tempfile.gettempdir(),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("输入和输出不能是同一个文件", completed.stderr)
            self.assertEqual(input_path.read_text(encoding="utf-8"), original)

    def test_existing_output_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.md"
            output_path = root / "output.md"
            input_path.write_text("# 第1章 测试\n", encoding="utf-8")
            output_path.write_text("preserved", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                clean_markdown(input_path, output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "preserved")


if __name__ == "__main__":
    unittest.main()
