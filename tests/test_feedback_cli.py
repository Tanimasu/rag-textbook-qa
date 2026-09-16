import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rag_textbook_qa.api.feedback import AnswerRegistry, FeedbackStore
from rag_textbook_qa.cli import main


class FeedbackCliTests(unittest.TestCase):
    def make_workspace(self, root: Path) -> None:
        (root / "src" / "rag_textbook_qa").mkdir(parents=True)
        (root / "project").mkdir()
        (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")

    def test_export_uses_the_workspace_feedback_database(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(root)
            store = FeedbackStore(root / "artifacts" / "product" / "feedback.sqlite3")
            registry = AnswerRegistry()
            answer_id = registry.remember(
                query="什么是进程？",
                book_id="os",
                result={
                    "status": "answered",
                    "answer": "教材回答",
                    "sources": [],
                    "conflicts": [],
                    "timing": {"total_seconds": 1.0},
                },
            )
            store.save(registry.resolve(answer_id), rating="helpful", reason=None, comment="")
            output = root / "exports" / "feedback.jsonl"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--workspace",
                        str(root),
                        "feedback",
                        "export",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["answer_id"], answer_id)
            self.assertEqual(record["rating"], "helpful")
            self.assertIn("已导出 1 条反馈", stdout.getvalue())

    def test_export_reports_when_no_feedback_database_exists(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(root)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--workspace",
                        str(root),
                        "feedback",
                        "export",
                        "--output",
                        str(root / "feedback.jsonl"),
                    ]
                )

            self.assertEqual(raised.exception.code, 1)
            self.assertIn("还没有反馈数据库", stderr.getvalue())

    def test_summary_prints_aggregate_json_without_question_text(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_workspace(root)
            store = FeedbackStore(root / "artifacts" / "product" / "feedback.sqlite3")
            registry = AnswerRegistry()
            answer_id = registry.remember(
                query="不会出现在摘要中的问题",
                book_id="os",
                result={
                    "status": "answered",
                    "answer": "教材回答",
                    "sources": [],
                    "conflicts": [],
                    "timing": {"total_seconds": 2.0},
                },
            )
            store.save(registry.resolve(answer_id), rating="helpful", reason=None, comment="")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    ["--workspace", str(root), "feedback", "summary", "--json"]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["helpful_rate"], 1.0)
            self.assertNotIn("不会出现在摘要", stdout.getvalue())

            human = io.StringIO()
            with contextlib.redirect_stdout(human):
                self.assertEqual(
                    main(["--workspace", str(root), "feedback", "summary"]),
                    0,
                )
            self.assertIn("反馈总数: 1", human.getvalue())
            self.assertIn("好评率: 100.0%", human.getvalue())
            self.assertIn("总耗时: 平均 2.000 秒", human.getvalue())


if __name__ == "__main__":
    unittest.main()
