"""Offline checks for optional planning, bounded retrieval and fallbacks."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_textbook_qa.llm.client import LLMClient
from rag_textbook_qa.rag.context import DEFAULT_CONTEXT_BUDGET, select_context
from rag_textbook_qa.rag.decomposition import plan_queries
from rag_textbook_qa.rag.engine import RAGEngine


class DecompositionTests(unittest.TestCase):
    def planner(self, queries):
        return MagicMock(plan_queries=MagicMock(return_value=json.dumps(
            {"decompose": True, "independent": True, "queries": queries})))

    def engine(self):
        engine = object.__new__(RAGEngine)
        engine.verbose = False
        engine.enable_llm = True
        engine.llm = self.planner(["进程定义", "线程定义"])
        engine.llm.generate_answer.return_value = {"success": True, "answer": "答案"}
        engine.vectorizer = SimpleNamespace(embedding_provider=None)
        engine.context_budget = DEFAULT_CONTEXT_BUDGET
        engine._execution_summary = MagicMock(return_value={})
        engine.search_single_book = MagicMock(return_value=[
            {"book_name": "os", "content": "原文证据", "similarity": 1.0}])
        engine._rerank = MagicMock(side_effect=lambda q, rows, k: rows[:k])
        return engine

    def test_valid_plan_and_single_request(self):
        llm = self.planner(["进程定义", "线程定义"])
        self.assertEqual(plan_queries("比较进程和线程", llm, 5)["status"], "active")
        llm.plan_queries.assert_called_once()

    def test_invalid_plans_fall_back(self):
        for queries in (["相同", " 相同 "], ["原问题", "其他"], ["a"],
                        ["a", "b", "c", "d"], ["a", "x" * 201], ["a", 42]):
            with self.subTest(queries=queries):
                self.assertEqual(plan_queries("原问题", self.planner(queries), 5)["status"], "fallback")
        for raw in ("not json", '[]', '{"decompose": "true"}',
                    '{"decompose":true,"independent":false,"queries":["a","b"]}'):
            self.assertEqual(plan_queries("q", MagicMock(plan_queries=MagicMock(return_value=raw)), 5)
                             ["status"], "fallback")

    def test_simple_and_small_budget(self):
        llm = MagicMock(plan_queries=MagicMock(return_value='{"decompose":false}'))
        self.assertEqual(plan_queries("定义", llm, 5)["status"], "not_needed")
        llm.reset_mock()
        self.assertEqual(plan_queries("定义", llm, 1)["status"], "fallback")
        llm.plan_queries.assert_not_called()

    def test_planner_exception_does_not_leak_error(self):
        llm = MagicMock(plan_queries=MagicMock(side_effect=TimeoutError("secret endpoint")))
        plan = plan_queries("q", llm, 5)
        self.assertEqual(plan["status"], "fallback")
        self.assertNotIn("secret", str(plan))

    def test_default_off_and_failed_plan_use_original_route(self):
        engine = self.engine()
        engine.ask("问题", book_name="os")
        engine.llm.plan_queries.assert_not_called()
        engine.llm.plan_queries.side_effect = TimeoutError()
        result = engine.ask("问题", book_name="os", use_decomposition=True, use_hyde=True)
        self.assertEqual(result["decomposition"]["status"], "fallback")
        engine.search_single_book.assert_called_with("os", "问题", 5, use_hyde=True)

    def test_no_llm_does_not_plan(self):
        engine = self.engine()
        result = engine.ask("问题", book_name="os", use_llm=False, use_decomposition=True)
        engine.llm.plan_queries.assert_not_called()
        self.assertEqual(result["decomposition"]["status"], "fallback")

    def test_original_route_scope_dedup_and_one_rerank(self):
        engine = self.engine()
        rows = engine.search_decomposed("原问题", ["a", "b"], "os", 5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query_ids"], [0, 1, 2])
        self.assertEqual([call.args[:2] for call in engine.search_single_book.call_args_list],
                         [("os", "原问题"), ("os", "a"), ("os", "b")])
        for call in engine.search_single_book.call_args_list:
            self.assertFalse(call.kwargs["use_hyde"])
            self.assertFalse(call.kwargs["use_reranker"])
        engine._rerank.assert_called_once()
        self.assertEqual(engine._rerank.call_args.args[0], "原问题")

    def test_all_books_candidate_cap_and_child_coverage(self):
        engine = self.engine()
        engine.vectorizer.client = MagicMock()
        engine.vectorizer.client.list_collections.return_value = [
            SimpleNamespace(name="textbook_os"), SimpleNamespace(name="textbook_net"),
            SimpleNamespace(name="ragbuild_staging")]
        engine.search_single_book.side_effect = lambda book, q, **kw: [
            {"book_name": book, "content": f"{q}-{i}"} for i in range(30)]
        rows = engine.search_decomposed("原问题", ["a", "b", "c"], None, 10)
        candidates = engine._rerank.call_args.args[1]
        self.assertEqual(len(candidates), 60)
        self.assertEqual({r["book_name"] for r in candidates}, {"os", "net"})
        self.assertTrue({1, 2, 3}.issubset({i for r in rows for i in r["query_ids"]}))

    def test_context_reserves_space_and_tracks_missing_route(self):
        engine = self.engine()
        engine.search_decomposed = MagicMock(return_value=[
            {"book_name": "os", "content": "资料完整句。" * 1000, "query_ids": [1]},
            {"book_name": "os", "content": "线程证据", "query_ids": [2]}])
        result = engine.ask("比较", book_name="os", use_decomposition=True)
        self.assertEqual(len(result["context_sources"]), 2)
        self.assertLessEqual(len(result["context"]), DEFAULT_CONTEXT_BUDGET)
        self.assertEqual(result["decomposition"]["uncovered_query_ids"], [])
        engine.search_decomposed.return_value = engine.search_decomposed.return_value[:1]
        result = engine.ask("比较", book_name="os", use_decomposition=True)
        self.assertEqual(result["decomposition"]["uncovered_query_ids"], [2])
        self.assertIn("须明确说明证据不足", result["prompt"])

    def test_active_decomposition_does_not_mix_in_adjacent_expansion(self):
        engine = self.engine()
        engine.search_decomposed = MagicMock(
            return_value=[
                {"book_name": "os", "content": "资料", "query_ids": [1, 2]}
            ]
        )

        result = engine.ask(
            "比较进程和线程",
            book_name="os",
            use_decomposition=True,
            use_adjacent_context=True,
        )

        self.assertEqual(result["decomposition"]["status"], "active")
        self.assertEqual(
            result["context_expansion"],
            {
                "enabled": True,
                "applied": False,
                "reason": "query_decomposition_active",
                "added_candidates": 0,
                "added_sources": 0,
            },
        )

    def test_sdk_planning_has_timeout_and_no_retry(self):
        client = object.__new__(LLMClient)
        client.default_model = "fake"
        client.client = MagicMock()
        client.client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"decompose":false}'))])
        self.assertEqual(client.plan_queries("prompt"), '{"decompose":false}')
        client.client.with_options.assert_called_once_with(timeout=20.0, max_retries=0)
        call = client.client.with_options.return_value.chat.completions.create.call_args
        self.assertEqual(call.kwargs["max_tokens"], 500)
        self.assertEqual(call.kwargs["temperature"], 0)


class EvidenceAllocationTests(unittest.TestCase):
    def test_preserves_long_required_tail_and_skips_optional_block(self):
        rows = [
            {"book_name": "os", "query_ids": [1], "content": "栈的应用。" * 80},
            {"book_name": "os", "query_ids": [2], "content": "队列介绍。" * 100 + "通过取模实现环状移动。"},
            {"book_name": "os", "query_ids": [0], "content": "其他资料。" * 100},
        ]
        context, sources = select_context(rows, max_length=1150, fair_share=True)
        self.assertIn("通过取模实现环状移动。", context)
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(not s["truncated"] for s in sources))
        self.assertLessEqual(len(context), 1150)

    def test_oversized_required_blocks_end_on_sentence_and_warn(self):
        rows = [{"book_name": "os", "query_ids": [i], "content": "这是完整的证据句子。" * 200}
                for i in (1, 2)]
        context, sources = select_context(rows, max_length=500, fair_share=True)
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(s["content"].endswith("[片段未完整装入，请勿推断省略内容]") for s in sources))
        self.assertLessEqual(len(context), 500)

    def test_short_required_block_releases_space_for_long_one(self):
        rows = [{"book_name": "os", "query_ids": [1], "content": "简短证据。"},
                {"book_name": "os", "query_ids": [2], "content": "完整证据。" * 70}]
        context, sources = select_context(rows, max_length=500, fair_share=True)
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(not s["truncated"] for s in sources))
        self.assertEqual(context, "".join(s["context_text"] for s in sources))

    def test_complete_compacted_table_fits_with_rendering_whitespace(self):
        rows = [{"book_name": "os", "query_ids": [1], "content": "证据。"},
                {"book_name": "os", "query_ids": [2],
                 "content": "<table><tr><td>表头</td></tr><tr><td>最后完整一行</td></tr></table>"}]
        context, sources = select_context(rows, max_length=300, fair_share=True)
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(not s["truncated"] for s in sources))
        self.assertIn("最后完整一行", context)

    def test_unbreakable_oversized_prose_is_not_cut_into_fake_evidence(self):
        rows = [{"book_name": "os", "query_ids": [1], "content": "不可分割内容" * 200}]
        context, sources = select_context(rows, max_length=200, fair_share=True)
        self.assertEqual(context, "")
        self.assertEqual(sources, [])


class PlanningGateTests(unittest.TestCase):
    def test_simple_and_dependent_requests_never_call_planner(self):
        for query in ('什么是进程？', '为什么DRAM需要定期刷新？', '根据上一问的答案解释它的局限。'):
            llm = MagicMock()
            result = plan_queries(query, llm, 5)
            self.assertEqual(result['status'], 'not_needed')
            llm.plan_queries.assert_not_called()

    def test_comparisons_and_multiple_questions_reach_planner(self):
        for query in ('什么是进程。线程是什么？', 'Cache和DRAM是什么？', '比较分页与分段。'):
            llm = MagicMock(plan_queries=MagicMock(return_value='{"decompose":false}'))
            plan_queries(query, llm, 5)
            llm.plan_queries.assert_called_once()

    def test_development_routing_examples(self):
        from pathlib import Path

        from rag_textbook_qa.rag.decomposition import planning_gate
        path = Path(__file__).resolve().parents[1] / 'data/evaluation/decomposition_routing_dev.json'
        for case in json.loads(path.read_text())['cases']:
            with self.subTest(query=case['question']):
                self.assertEqual(planning_gate(case['question']), case['expected_reason'])


class PlannerProviderOptionsTests(unittest.TestCase):
    def test_disable_thinking_only_for_verified_provider_model(self):
        for host, model, expected in (
            ("https://api.siliconflow.cn/v1/", "deepseek-ai/DeepSeek-V4-Pro", True),
            ("https://other.example/v1/", "deepseek-ai/DeepSeek-V4-Pro", False),
            ("https://api.siliconflow.cn/v1/", "other-model", False),
        ):
            client = object.__new__(LLMClient)
            client.base_url, client.default_model, client.client = host, model, MagicMock()
            client.client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"decompose":false}'))])
            client.plan_queries("prompt")
            kwargs = client.client.with_options.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs.get("extra_body"), {"enable_thinking": False} if expected else None)

    def test_timeout_phase_is_recorded_without_exception_details(self):
        import httpx
        error = RuntimeError("secret")
        error.__cause__ = httpx.ReadTimeout("private destination")
        client = MagicMock(plan_queries=MagicMock(side_effect=error))
        result = plan_queries("比较进程与线程", client, 5)
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["timeout_phase"], "read")
        self.assertNotIn("secret", str(result))
        self.assertNotIn("private", str(result))
