import json
import tempfile
import unittest
from pathlib import Path

from rag_textbook_qa.evaluation.generation import (
    Arm,
    ContextVariant,
    GenerationCase,
    context_blocks,
    quote_found,
    score_answer,
)
from rag_textbook_qa.evaluation.generation_runner import (
    JUDGE_VERSION,
    judge_answer,
    run_generation_experiment,
)

SPOOLING = (
    "而是会由假脱机管理进程完成两项工作： $\\textcircled{1}$ 在磁盘缓冲区中为之申请一个空闲盘块；"
    " $\\textcircled{2}$ 为用户进程申请一张空白的请求打印表。"
)
SOURCE = {"citation_id": 1, "content": "进程是系统进行资源分配和调度的一个独立单位。"}
CASE = GenerationCase(
    "01",
    "什么是进程？",
    ("进程的含义",),
    {"baseline": ContextVariant("【参考资料 1】\n内容:\n" + SOURCE["content"], (SOURCE,), ("答",))},
)
EXTRACTION = json.dumps(
    {
        "claims": [{"id": 1, "type": "fact", "text": "进程是独立单位"}],
        "coverage": [{"id": 1, "addressed": True}],
    },
    ensure_ascii=False,
)
VERDICTS = json.dumps(
    {
        "verdicts": [
            {
                "id": 1,
                "label": "supported",
                "evidence": [{"source_id": 1, "quote": "资源分配和调度的一个独立单位"}],
            }
        ]
    },
    ensure_ascii=False,
)


class FormattingTolerantQuoteTests(unittest.TestCase):
    def test_latex_and_enumeration_markers_do_not_break_a_faithful_quote(self):
        for quote in (
            "完成两项工作：①在磁盘缓冲区中为之申请一个空闲盘块",
            "完成两项工作：在磁盘缓冲区中为之申请一个空闲盘块",
            "完成两项工作：(1)在磁盘缓冲区中为之申请一个空闲盘块",
        ):
            with self.subTest(quote=quote):
                self.assertTrue(quote_found(quote, SPOOLING))

    def test_one_dropped_character_in_a_long_quote_still_counts(self):
        self.assertTrue(quote_found("为用户进程申请一张空的请求打印表", SPOOLING))

    def test_words_stitched_from_different_places_do_not_count(self):
        self.assertFalse(quote_found("假脱机管理进程申请空闲的请求打印表", SPOOLING))


class JudgeInputTests(unittest.TestCase):
    def test_blocks_keep_each_citations_own_heading(self):
        context = (
            "【参考资料 1】\n章节: 5.9.1 TCP 的连接建立\n内容:\n三次握手。\n---\n"
            "【参考资料 2】\n章节: 6.2 万维网\n内容:\n超文本。\n---\n"
        )
        sources = [
            {"citation_id": 2, "content": "超文本。"},
            {"citation_id": 1, "content": "三次握手。"},
        ]
        blocks = context_blocks(context, sources)

        self.assertEqual([block["citation_id"] for block in blocks], [2, 1])
        self.assertIn("5.9.1 TCP 的连接建立", blocks[1]["content"])
        self.assertNotIn("超文本", blocks[1]["content"])

    def test_scores_keep_the_judges_quotes_for_audit(self):
        claims = [{"id": 1, "type": "fact", "text": "进程是独立单位"}]
        verdicts = {1: json.loads(VERDICTS)["verdicts"][0] | {"reason": ""}}
        score = score_answer(claims, [True], verdicts, [SOURCE])

        self.assertEqual(score["claims"][0]["evidence"][0]["quote"], "资源分配和调度的一个独立单位")

    def test_malformed_json_is_asked_again_with_the_error_named(self):
        prompts = []

        def judge(prompt: str) -> str:
            prompts.append(prompt)
            if "任务一" in prompt:
                return EXTRACTION
            verification_calls = [p for p in prompts if "任务一" not in p]
            return '{"verdicts": [{"id": 1 "label": "supported"}]}' if len(verification_calls) == 1 else VERDICTS

        result = judge_answer(CASE, Arm("kept", "baseline", None), "答", judge)

        self.assertEqual(result["score"]["supported"], 1)
        self.assertNotIn("合法 JSON", prompts[1])
        self.assertIn("合法 JSON", prompts[2])

    def test_protocol_records_the_judge_version(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_generation_experiment(
                [CASE],
                [Arm("kept", "baseline", None)],
                output_dir=output,
                generator=lambda prompt, temperature: {},
                judge=lambda prompt: EXTRACTION if "任务一" in prompt else VERDICTS,
                samples=1,
                seed=0,
                concurrency=1,
                protocol={},
                prompt_builder=lambda question, context: question,
                log=lambda line: None,
            )
            frozen = json.loads((output / "protocol.json").read_text(encoding="utf-8"))

        self.assertEqual(frozen["settings"]["judge_version"], JUDGE_VERSION)


if __name__ == "__main__":
    unittest.main()
