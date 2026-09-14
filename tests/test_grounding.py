import json
import unittest
from unittest.mock import MagicMock

import test_decomposition as fixtures

from rag_textbook_qa.rag.grounding import BLOCKED, verify_answer


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{'citation_id': 1, 'content': '栈可用于递归。'}]
        self.claim = {'text': '栈可用于递归。', 'evidence': [{'id': 1, 'quote': '栈可用于递归。'}]}

    def client(self, claims=None, verdicts=None):
        return MagicMock(audit_citations=MagicMock(side_effect=[
            json.dumps({'claims': claims or [self.claim]}),
            json.dumps({'verdicts': verdicts or [{'id': 1, 'supported': True}]})]))

    def test_renders_checked_claim_only_not_original_draft(self):
        llm = self.client()
        result = verify_answer('栈的用途', '不支持的返回地址细节', self.sources, llm)
        self.assertEqual(result['status'], 'checked')
        self.assertNotIn('返回地址', result['answer'])
        self.assertIn('【参考资料 1】', result['answer'])
        self.assertEqual(llm.audit_citations.call_count, 2)

    def test_fabricated_quote_unknown_source_and_boolean_id_block(self):
        for ref in ({'id': 1, 'quote': '栈保存返回地址'}, {'id': 2, 'quote': '栈可用于递归。'},
                    {'id': True, 'quote': '栈可用于递归。'}):
            result = verify_answer('q', 'draft', self.sources,
                                   self.client(claims=[{**self.claim, 'evidence': [ref]}]))
            self.assertEqual(result['answer'], BLOCKED)
            self.assertEqual(result['calls'], 1)

    def test_false_verdict_omits_claim_without_rewrite(self):
        claims = [self.claim, {**self.claim, 'text': '不支持的实现细节'}]
        result = verify_answer('q', 'draft', self.sources, self.client(claims, [
            {'id': 1, 'supported': True}, {'id': 2, 'supported': False}]))
        self.assertEqual(result['status'], 'checked')
        self.assertNotIn('实现细节', result['answer'])
        self.assertEqual(result['rejected_claims'], 1)

    def test_missing_duplicate_or_nonboolean_verdicts_block(self):
        for verdicts in ([{'id': 2, 'supported': True}], [{'id': 1, 'supported': 'true'}],
                         [{'id': 1, 'supported': True}, {'id': 1, 'supported': True}]):
            self.assertEqual(verify_answer('q', 'draft', self.sources,
                                          self.client(verdicts=verdicts))['status'], 'blocked')

    def test_timeout_never_returns_draft_or_error_details(self):
        llm = MagicMock(audit_citations=MagicMock(side_effect=TimeoutError('secret')))
        result = verify_answer('q', 'draft-secret', self.sources, llm)
        self.assertEqual(result['answer'], BLOCKED)
        self.assertNotIn('secret', str(result))

    def test_stream_callback_only_receives_checked_final_answer(self):
        engine = fixtures.DecompositionTests().engine()
        engine.llm.generate_answer.return_value['answer'] = 'unchecked draft'
        claim = {'text': '原文证据', 'evidence': [{'id': 1, 'quote': '原文证据'}]}
        engine.llm.audit_citations = self.client([claim]).audit_citations
        callback = MagicMock()
        result = engine.ask('问题', book_name='os', verify_citations=True, on_answer_chunk=callback)
        self.assertTrue(result['success'])
        self.assertEqual(result['grounding']['status'], 'checked')
        engine.llm.stream_answer.assert_not_called()
        callback.assert_called_once_with(result['answer'])
        self.assertNotIn('unchecked draft', str(result))

    def test_failed_audit_blocks_answer_and_success_flag(self):
        engine = fixtures.DecompositionTests().engine()
        engine.llm.audit_citations.side_effect = TimeoutError()
        result = engine.ask('问题', book_name='os', verify_citations=True)
        self.assertFalse(result['success'])
        self.assertEqual(result['answer'], BLOCKED)
        self.assertEqual(result['llm_response']['answer'], BLOCKED)

    def test_all_rejected_is_blocked(self):
        result = verify_answer('q', 'draft', self.sources,
                               self.client(verdicts=[{'id': 1, 'supported': False}]))
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['answer'], BLOCKED)

    def test_invalid_json_and_empty_claims_block(self):
        for raw in ('not json', '{"claims":[]}', '[]'):
            llm = MagicMock(audit_citations=MagicMock(return_value=raw))
            result = verify_answer('q', 'draft', self.sources, llm)
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(result['calls'], 1)

    def test_missing_judge_rows_block(self):
        llm = self.client()
        llm.audit_citations.side_effect = [json.dumps({'claims': [self.claim]}), '{"verdicts":[]}']
        self.assertEqual(verify_answer('q', 'draft', self.sources, llm)['status'], 'blocked')

    def test_audit_sdk_rejects_truncated_output_and_uses_no_retry(self):
        from types import SimpleNamespace

        from rag_textbook_qa.llm.client import LLMClient
        client = object.__new__(LLMClient)
        client.base_url = 'https://api.siliconflow.cn/v1/'
        client.default_model = 'deepseek-ai/DeepSeek-V4-Pro'
        client.client = MagicMock()
        client.client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason='length', message=SimpleNamespace(content='{}'))])
        with self.assertRaises(ValueError):
            client.audit_citations('audit prompt')
        client.client.with_options.assert_called_once_with(timeout=40.0, max_retries=0)
        kwargs = client.client.with_options.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs['extra_body'], {'enable_thinking': False})
