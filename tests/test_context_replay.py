import contextlib
import hashlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from test_vectorizer_provider import FakeEmbeddingProvider

from rag_textbook_qa.indexing import MultiBookVectorizer
from rag_textbook_qa.indexing.revision import index_revision
from rag_textbook_qa.indexing.snapshot import TemporaryIndexDirectory
from rag_textbook_qa.rag.context import select_context
from scripts.replay_context_expansion import (
    bridge_parent_context,
    compact_parent_context,
    load_index_corpora,
    parent_neighbours,
    question_parent_context,
    replay,
)


def chunk(identifier, subsection, *, chapter="第一章", section="1.1 主题", book="os"):
    return {
        "chunk_id": identifier, "book_name": book,
        "chapter": chapter, "section_h2": section,
        "section_h3": subsection, "section_h4": "",
        "content": f"{subsection}的完整教材正文。", "char_count": 50,
        "has_code": False, "has_image": False, "level": 3,
    }


class ParentNeighbourTests(unittest.TestCase):
    def test_crosses_subsections_but_preserves_ranked_order_and_book_identity(self):
        corpus = [chunk("x_p1", "定义"), chunk("x_p2", "原理"), chunk("x_p3", "流程")]
        other = [{**row, "book_name": "database"} for row in corpus]
        ranked = [corpus[0], other[0]]
        result = parent_neighbours(ranked, {"os": corpus, "database": other}, radius=2)
        self.assertEqual(result[:2], ranked)
        self.assertEqual(
            [(r["book_name"], r["chunk_id"]) for r in result[2:]],
            [("os", "x_p2"), ("os", "x_p3"),
             ("database", "x_p2"), ("database", "x_p3")],
        )
        self.assertEqual([r["adjacent_distance"] for r in result[2:]], [1, 2, 1, 2])

    def test_never_jumps_over_a_boundary_even_when_the_heading_reappears(self):
        anchor = chunk("x_p1", "定义")
        for boundary in (
            chunk("x_p2", "别节", section="1.2 其他"),
            chunk("x_p2", "别章", chapter="第二章"),
        ):
            with self.subTest(boundary=boundary):
                corpus = [anchor, boundary, chunk("x_p3", "同名节")]
                self.assertEqual(parent_neighbours([anchor], {"os": corpus}, radius=2),
                                 [anchor])

    def test_missing_named_parent_or_anchor_cannot_add_neighbours(self):
        for anchor in (chunk("unknown", "定义"), chunk("x_p1", "定义", section="")):
            with self.subTest(anchor=anchor):
                self.assertEqual(parent_neighbours(
                    [anchor], {"os": [chunk("x_p1", "定义"), chunk("x_p2", "原理")]},
                    radius=1,
                ), [anchor])

    def test_rejects_ambiguous_or_invalid_radius(self):
        for radius in (True, 0, -1, 1.5):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                parent_neighbours([], {}, radius=radius)

    def test_compact_replay_skips_exercises_and_preserves_whole_prose_paragraphs(self):
        anchor = chunk("x_p1", "定义")
        exercise = chunk("x_p2", "习题")
        detail = chunk("x_p3", "机制")
        detail["content"] = "这是一段需要完整加入的机制解释，不能被截成半句话。\n\n" * 2
        corpus = [anchor, exercise, detail]
        context, sources = compact_parent_context([anchor], {"os": corpus})
        self.assertEqual([s["chunk_id"] for s in sources], ["x_p1", "x_p3"])
        self.assertEqual(sources[1]["content"], detail["content"].rstrip())
        self.assertFalse(sources[1]["truncated"])
        self.assertNotIn("习题", context)

    def test_compact_replay_does_not_clip_a_large_supplement(self):
        anchor = chunk("x_p1", "定义")
        huge = {**chunk("x_p2", "机制"), "content": "很长的说明。" * 300}
        self.assertEqual(compact_parent_context([anchor], {"os": [anchor, huge]}),
                         select_context([anchor]))

    def test_compact_replay_validates_budget_even_without_supplements(self):
        for budget in (True, 1.5, 0, -1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                compact_parent_context([], {}, context_budget=budget)

    def test_question_selection_can_find_mechanisms_after_an_image_or_table(self):
        anchor = chunk("x_p1", "栈用途")
        explanation = {**chunk("x_p2", "应用"), "content": (
            "这个开头描述了教材案例的背景条件，但没有说明使用的数据结构。\n\n"
            "![](images/example.png)\n\n"
            "栈用于保存调用参数和返回地址，调用返回时栈按后进先出的顺序恢复现场。\n\n"
            "<table><tr><td>占位的教材表格示例，表格不能被当成正文</td></tr></table>\n\n"
            "这段无关说明足够长，但与本次数据结构的用途没有关系。"
        )}
        baseline, baseline_sources = select_context([anchor])
        context, sources = question_parent_context(
            "栈的后进先出用途？", [anchor], {"os": [anchor, explanation]},
        )
        self.assertTrue(context.startswith(baseline))
        self.assertEqual(sources[0], baseline_sources[0])
        self.assertEqual(sources[1]["content"], explanation["content"].split("\n\n")[2])
        self.assertFalse(sources[1]["truncated"])

    def test_question_selection_cannot_add_unmatched_text_or_exercises(self):
        anchor = chunk("x_p1", "栈用途")
        exercise = {**chunk("x_p2", "习题"), "content": "栈的用途需要在练习中总结，这段习题有关键词也不能作为正文补充。"}
        irrelevant = {**chunk("x_p3", "机制"), "content": "线程调度需要操作系统来管理，这是与问题不相干的完整教材说明。"}
        for query in ("栈？", "", "的如何"):
            with self.subTest(query=query):
                self.assertEqual(question_parent_context(
                    query, [anchor], {"os": [anchor, exercise, irrelevant]},
                ), select_context([anchor]))


class BridgeSelectionTests(unittest.TestCase):
    def test_named_dependency_uses_the_query_relevant_parent_before_other_anchors(self):
        broad = {**chunk("a_p1", "概述", section="1.1 内存简介"),
                 "content": "计算机内存可以通过接口来接收数据，这是一段概述中的完整解释。"}
        distractor = {**chunk("a_p2", "接口", section="1.1 内存简介"),
                      "content": "接口负责不同设备之间的数据传递，这是本题没有询问的完整说明。"}
        anchor = {**chunk("b_p3", "写时复制", section="1.2 页面共享"),
                  "content": "写时复制先共享原有页面，并通过引用计数记录共享状态，以减少复制开销。"}
        definition = {**chunk("b_p4", "引用计数", section="1.2 页面共享"),
                      "content": "引用计数记录一个页面有多少个使用者，多个使用者可以共同持有同一页面。\n\n"
                                 "当使用者释放页面时计数会减少，全部使用者退出后才可以回收这份页面。"}
        ranked = [broad, anchor]
        context, sources = bridge_parent_context(
            "写时复制具体如何实现？", ranked,
            {"os": [broad, distractor, anchor, definition]},
        )
        baseline, baseline_sources = select_context(ranked)
        self.assertTrue(context.startswith(baseline))
        self.assertEqual(sources[:2], baseline_sources)
        self.assertEqual([s["chunk_id"] for s in sources[2:]], ["b_p4"])
        self.assertEqual(sources[2]["content"], definition["content"])

    def test_bridge_requires_the_concept_in_delivered_anchor_prose(self):
        anchor = {**chunk("x_p1", "写时复制"),
                  "content": "写时复制在发生修改时才建立独立页面，避免预先复制全部页面造成浪费。"}
        neighbour = {**chunk("x_p2", "引用计数"),
                     "content": "引用计数记录对象的使用者数量，数量减少到零之后才可以回收对象。"}
        self.assertEqual(bridge_parent_context(
            "写时复制？", [anchor], {"os": [anchor, neighbour]},
        ), select_context([anchor]))

    def test_named_dependency_cannot_jump_an_image_to_claim_introductory_prose(self):
        anchor = {**chunk("x_p1", "写时复制"),
                  "content": "写时复制使用引用计数记录页面共享状态，使页面在发生修改前不必复制。"}
        neighbour = {**chunk("x_p2", "引用计数"), "content": (
            "![这是引用计数的教材示意图](images/example.png)\n\n"
            "引用计数记录对象的使用者数量，数量减少到零之后才可以回收对象。"
        )}
        self.assertEqual(bridge_parent_context(
            "写时复制？", [anchor], {"os": [anchor, neighbour]},
        ), select_context([anchor]))

    def test_complete_three_paragraph_flow_in_a_bridged_parent_fits_without_clipping(self):
        anchor = {**chunk("x_p1", "调度器"),
                  "content": "调度器通过优先级来分配处理器时间，以下正文提供概要而没有展开处理步骤。"}
        dependency = {**chunk("x_p2", "优先级"),
                      "content": "优先级表示一个任务获得处理器的先后顺序，是调度决策的一个依据。"}
        parts = [
            "调度器先接收就绪任务，并记录任务到达的时间，作为后续处理的数据依据。",
            "调度器再比较任务优先级，从全部就绪任务中选择下一个可以运行的任务。",
            "调度器最后把处理器分配给选中的任务，并保存原任务的现场以便恢复运行。",
        ]
        neighbour = {**chunk("x_p3", "调度器"), "content": "\n\n".join(parts)}
        _, sources = bridge_parent_context(
            "调度器工作流程是什么？", [anchor], {"os": [anchor, dependency, neighbour]},
        )
        self.assertEqual(sources[-1]["content"], neighbour["content"])
        self.assertFalse(sources[-1]["truncated"])

    def test_a_concept_already_in_the_question_is_not_a_missing_dependency(self):
        anchor = {**chunk("x_p1", "调度器"),
                  "content": "调度器负责分配处理器时间，这是一段没有展开工作流程的完整概述。"}
        parts = ["调度器在这一阶段处理就绪任务，并记录任务信息用于下一阶段。" * 2] * 3
        neighbour = {**chunk("x_p2", "调度器"), "content": "\n\n".join(parts)}
        self.assertEqual(bridge_parent_context(
            "调度器工作流程是什么？", [anchor], {"os": [anchor, neighbour]},
        ), question_parent_context(
            "调度器工作流程是什么？", [anchor], {"os": [anchor, neighbour]},
        ))

    def test_exercises_empty_queries_and_invalid_budgets_keep_existing_guards(self):
        anchor = {**chunk("x_p1", "写时复制"),
                  "content": "写时复制使用引用计数记录页面共享状态，使页面在发生修改前不必复制。"}
        exercise = {**chunk("x_p2", "引用计数练习题"),
                    "content": "引用计数与写时复制的关系可以在习题中考查，这里有关键词也不能选取。"}
        for query in ("写时复制？", "", "的如何"):
            with self.subTest(query=query):
                self.assertEqual(bridge_parent_context(
                    query, [anchor], {"os": [anchor, exercise]},
                ), select_context([anchor]))
        for budget in (True, 0, -1, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                bridge_parent_context("写时复制？", [], {}, context_budget=budget)


class ContextReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryIndexDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db = self.root / "db"
        self.capture = self.root / "capture"
        self.capture.mkdir()
        self.output = self.root / "output"
        chunks_path = self.root / "chunks.json"
        chunks_path.write_text(json.dumps([
            chunk("x_p1", "定义"), chunk("x_p2", "原理"), chunk("x_p3", "流程"),
        ], ensure_ascii=False), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            vectorizer = MultiBookVectorizer(
                db_path=self.db, embedding_provider=FakeEmbeddingProvider(),
            )
            try:
                vectorizer.vectorize_book(chunks_path, "os")
            finally:
                vectorizer.close()
        self.questions = self.root / "questions.json"
        self.questions.write_text(json.dumps([
            {"book_name": "os", "question": "定义？", "relevant_sections": ["1.1 主题"],
             "split": "dev"},
            {"book_name": "os", "question": "验收题？", "relevant_sections": ["1.1 主题"],
             "split": "holdout"},
        ], ensure_ascii=False), encoding="utf-8")
        self.corpus = load_index_corpora(self.db)["os"]
        ranked = [self.corpus[0]]
        context, sources = select_context(ranked, 4000)
        self.case = {
            "id": "dev-01", "book_name": "os", "question": "定义？",
            "relevant_sections": ["1.1 主题"],
            "variants": {"current": {"ranked": ranked, "context": context,
                                      "sources": sources}},
        }
        self.save_case()
        (self.capture / "protocol.json").write_text(json.dumps({
            "complete": True, "split": "dev", "index_revision": index_revision(self.db),
            "dataset_sha256": hashlib.sha256(self.questions.read_bytes()).hexdigest(),
            "top_k": 5, "context_budget": 4000,
        }), encoding="utf-8")

    def save_case(self):
        (self.capture / "dev-01.json").write_text(
            json.dumps(self.case, ensure_ascii=False), encoding="utf-8",
        )

    def run_replay(self):
        return replay(capture_dir=self.capture, db_path=self.db, questions_path=self.questions,
                      output_dir=self.output)

    def test_real_chroma_replay_is_read_only_and_never_creates_a_model_or_socket(self):
        database_before = (self.db / "chroma.sqlite3").read_bytes()
        capture_before = (self.capture / "dev-01.json").read_bytes()
        with (
            patch("rag_textbook_qa.rag.engine.RAGEngine.__init__",
                  side_effect=AssertionError("Engine construction forbidden")),
            patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")),
        ):
            report = self.run_replay()
        self.assertEqual(report["summary"]["baseline"]["questions"], 1)
        self.assertEqual(report["summary"]["same_section"]["mean_context_chars"],
                         report["summary"]["baseline"]["mean_context_chars"])
        self.assertEqual(report["cases"][0]["arms"]["parent_radius_2"]["added_sources"], 2)
        self.assertEqual(report["cases"][0]["arms"]["parent_radius_2"]["sources"][0],
                         self.case["variants"]["current"]["sources"][0])
        self.assertEqual((self.db / "chroma.sqlite3").read_bytes(), database_before)
        self.assertEqual((self.capture / "dev-01.json").read_bytes(), capture_before)
        self.assertTrue((self.output / "report.json").is_file())

    def test_rejects_stale_bodies_or_headings_before_writing_a_report(self):
        for field in ("content", "section_h3"):
            with self.subTest(field=field):
                old = self.case["variants"]["current"]["ranked"][0][field]
                self.case["variants"]["current"]["ranked"][0][field] = "已改变"
                self.save_case()
                with self.assertRaisesRegex(ValueError, "索引原文不一致"):
                    self.run_replay()
                self.assertFalse(self.output.exists())
                self.case["variants"]["current"]["ranked"][0][field] = old

    def test_rejects_holdout_duplicate_and_missing_development_captures(self):
        self.case["question"] = "验收题？"
        self.save_case()
        with self.assertRaisesRegex(ValueError, "非开发题"):
            self.run_replay()
        self.case["question"] = "定义？"
        self.save_case()
        duplicate = self.capture / "dev-02.json"
        duplicate.write_bytes((self.capture / "dev-01.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "重复问题"):
            self.run_replay()
        duplicate.unlink()
        (self.capture / "dev-01.json").unlink()
        with self.assertRaisesRegex(ValueError, "完整覆盖"):
            self.run_replay()
        self.assertFalse(self.output.exists())

    def test_index_read_connection_enforces_read_only_mode(self):
        import sqlite3

        real_connect = sqlite3.connect
        calls = []

        def tracked_connect(*args, **kwargs):
            calls.append((args, kwargs))
            connection = real_connect(*args, **kwargs)
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM embeddings")
            connection.rollback()
            return connection

        with patch("scripts.replay_context_expansion.sqlite3.connect", tracked_connect):
            self.assertEqual(len(load_index_corpora(self.db)["os"]), 3)
        self.assertIn("?mode=ro", calls[0][0][0])
        self.assertTrue(calls[0][1]["uri"])

    def test_source_span_distinguishes_corpus_availability_from_delivered_evidence(self):
        source = self.root / "source.md"
        source.write_text(self.corpus[1]["content"], encoding="utf-8")
        spans = self.root / "spans.json"
        spans.write_text(json.dumps([{
            "book_name": "os", "question": "定义？", "split": "dev",
            "relevant_sections": ["1.1 主题"],
            "evidence": {"path": str(source), "start_line": 1, "end_line": 1,
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        }], ensure_ascii=False), encoding="utf-8")
        report = replay(capture_dir=self.capture, db_path=self.db, questions_path=self.questions,
                        evidence_path=spans, output_dir=self.output)
        arms = report["cases"][0]["arms"]
        for name, expected in (("baseline", 0), ("same_section", 0), ("parent_radius_2", 1)):
            with self.subTest(arm=name):
                score = arms[name]["evidence_spans"][0]
                self.assertEqual(score["indexed_corpus_coverage"], 1)
                self.assertEqual(score["source_evidence_context_coverage"], expected)


if __name__ == "__main__":
    unittest.main()
