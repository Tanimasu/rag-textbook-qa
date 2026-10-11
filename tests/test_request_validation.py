"""Reject invalid Python request options before optional model work begins."""

import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

from rag_textbook_qa.rag.engine import RAGEngine


class RequestValidationTests(unittest.TestCase):
    def test_invalid_ask_options_fail_before_planning_or_retrieval(self):
        for options in (
            {"query": " \n "}, {"query": None}, {"book_name": ""}, {"book_name": ["os"]},
            {"max_tokens": 0}, {"context_budget": 0}, {"temperature": float("nan")},
            {"temperature": float("inf")}, {"temperature": "warm"}, {"temperature": True},
            *({field: value} for field in ("top_k", "max_tokens", "context_budget")
              for value in (True, 1.5, "2")),
        ):
            with self.subTest(options=options):
                engine = object.__new__(RAGEngine)
                engine.verbose = False
                engine.context_budget = 4000
                engine.enable_llm = True
                engine.llm = MagicMock()
                engine.vectorizer = MagicMock()
                engine._retrieve = MagicMock(side_effect=AssertionError("retrieval already started"))
                request = {"query": "比较进程和线程", "use_decomposition": True, "use_hyde": True,
                           **options}
                with self.assertRaises(ValueError):
                    engine.ask(**request)
                engine.llm.plan_queries.assert_not_called()
                engine.llm.generate_answer.assert_not_called()
                engine._retrieve.assert_not_called()

    def test_invalid_constructor_budget_fails_before_loading_vectorizer(self):
        for budget in (True, 1.5, "2", 0, -1):
            with (
                self.subTest(budget=budget), contextlib.redirect_stdout(io.StringIO()),
                patch("rag_textbook_qa.rag.engine.MultiBookVectorizer") as vectorizer,
            ):
                with self.assertRaises(ValueError):
                    RAGEngine(context_budget=budget)
                vectorizer.assert_not_called()

    def test_invalid_retrieval_count_fails_before_embedding_or_index_access(self):
        for method, args in (
            ("search_embedding", ("os", "问题")),
            ("search_bm25", ("os", "问题")),
            ("search_single_book", ("os", "问题")),
            ("search_all_books", ("问题",)),
        ):
            for count in (True, 1.5, "2", 0, -1):
                with self.subTest(method=method, count=count):
                    engine = object.__new__(RAGEngine)
                    engine.vectorizer = MagicMock()
                    engine._run_with_index_snapshot = MagicMock(
                        side_effect=AssertionError("index already accessed")
                    )
                    engine.refresh_index_if_changed = MagicMock(
                        side_effect=AssertionError("index already accessed")
                    )
                    field = "top_k_per_book" if method == "search_all_books" else "top_k"
                    with self.assertRaises(ValueError):
                        getattr(engine, method)(*args, **{field: count})
                    engine.vectorizer.client.list_collections.assert_not_called()
                    engine.vectorizer.embedding_provider.embed_queries.assert_not_called()

    def test_valid_request_keeps_text_and_generation_options(self):
        engine = object.__new__(RAGEngine)
        engine.verbose = False
        engine.context_budget = 4000
        engine.enable_llm = True
        engine.llm = MagicMock()
        engine.vectorizer = MagicMock()
        engine._execution_summary = MagicMock(return_value={})
        engine._retrieve = MagicMock(return_value=(
            [{"book_name": "os", "content": "线程是进程中的执行单元。"}], "trace", 0,
        ))
        engine._generate = MagicMock(return_value=({"answer": "回答", "success": True}, None))
        query = " 比较进程和线程 \n"
        result = engine.ask(query, book_name="os", top_k=2, max_tokens=123,
                            context_budget=100, temperature=0)
        self.assertEqual(result["query"], query)
        self.assertIn(query, result["prompt"])
        self.assertLessEqual(len(result["context"]), 100)
        self.assertEqual(engine._retrieve.call_args.args[0], query)
        self.assertEqual(engine._retrieve.call_args.args[2:4], ("os", 2))
        self.assertEqual(engine._generate.call_args.kwargs["max_tokens"], 123)
        self.assertEqual(engine._generate.call_args.kwargs["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
