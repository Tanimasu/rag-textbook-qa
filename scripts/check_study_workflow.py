"""Check the study UI through real HTTP/Chrome/Chroma with local model simulators.

Requires API dependencies, requests, Playwright, and an installed Chromium browser.
No remote Worker or paid model is used. Existing output directories are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import threading
import time
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import requests
import uvicorn
from playwright.sync_api import expect, sync_playwright

import rag_textbook_qa
from rag_textbook_qa.api.app import create_api_app
from rag_textbook_qa.api.feedback import FeedbackStore
from rag_textbook_qa.api.guard import AccessGuard, GuardSettings
from rag_textbook_qa.api.imports import ImportManager
from rag_textbook_qa.indexing.snapshot import TemporaryIndexDirectory
from rag_textbook_qa.providers import ModelIdentity
from rag_textbook_qa.rag.engine import RAGEngine

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1],
                    help="Workspace used to label an optional source file in the report")
parser.add_argument("--chunks", type=Path,
                    help="Optional real operating-system chunks JSON; default uses synthetic passages")
parser.add_argument("--output", type=Path, required=True, help="New directory for logs, screenshots and exports")
args = parser.parse_args()
ROOT = args.workspace.expanduser().resolve()
OUT = args.output.expanduser().resolve()
if OUT.exists():
    raise RuntimeError('Use a new output directory; previous evidence is preserved')
source_path = args.chunks.expanduser().resolve() if args.chunks else None
if source_path is not None:
    source_bytes = source_path.read_bytes()
    all_rows = json.loads(source_bytes)
    selected = [row for row in all_rows if '进程是' in row['content'] and len(row['content']) > 150][:3]
    if len(selected) != 3:
        parser.error("The supplied chunks must contain three process passages longer than 150 characters")
    source_label = str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path)
else:
    selected = []
    for index, topic in enumerate(("进程与程序", "进程状态", "进程调度"), 1):
        text = (f"这是用于软件流程验收的合成片段，主题为{topic}，不作为教材内容或正确答案。"
                "进程是此测试的检索关键词。这里只检查上传、索引发布、引用定位和页面交互是否连接正确。"
                "本次模型调用全部由本地确定性模拟器替代，不进行知识准确性或模型能力评测。")
        selected.append({"chunk_id": f"fixture_p{index}", "chapter": "流程测试合成资料",
                         "section_h2": topic, "content": text, "level": 2, "char_count": len(text),
                         "has_code": False, "has_image": False})
    source_bytes = json.dumps(selected, ensure_ascii=False).encode()
    source_label = "builtin:study-workflow-v1"
source_hash = hashlib.sha256(source_bytes).hexdigest()
payload = json.dumps(selected, ensure_ascii=False).encode()
OUT.mkdir(parents=True)
report = {'complete': False, 'version': rag_textbook_qa.__version__, 'package_path': str(Path(rag_textbook_qa.__file__).parent), 'source_file': source_label,
          'synthetic_fixture': source_path is None,
          'source_sha256': source_hash, 'source_chunk_ids': [row['chunk_id'] for row in selected],
          'real_application_and_chroma': True, 'inference': 'local deterministic simulators',
          'paid_model_calls': 0, 'remote_worker_calls': 0, 'steps': []}


class Embedding:
    identity = ModelIdentity(task='embedding', model='offline-ui-embedding', normalized=True, query_instruction='')
    documents = 0
    queries = 0

    def embed_documents(self, texts):
        self.documents += 1
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_queries(self, texts):
        self.queries += 1
        return [[1.0, 0.0, 0.0] for _ in texts]


class Reranker:
    identity = ModelIdentity(task='reranker', model='offline-ui-reranker')
    calls = 0

    def rerank(self, query, documents):
        self.calls += 1
        return [float(len(set(query) & set(document))) for document in documents]


class Generator:
    default_model = 'offline-ui-generator'
    calls = 0

    def generate_answer(self, *args, **kwargs):
        raise AssertionError('Only streaming is expected in this UI check')

    def stream_answer(self, prompt, **kwargs):
        self.calls += 1
        yield '## 本地功能验收\n这是一份模拟回答，用于验证界面流程。【参考资料 1】\n\n'
        yield '| 阶段 | 状态 |\n| --- | --- |\n| 导入与检索 | 已执行 |\n\n```python\nprint("offline check")\n```'


embedding, reranker, generator = Embedding(), Reranker(), Generator()
code = 'offline-ui-check'
headers = {'X-Access-Code': code}
server = None
thread = None
try:
    with TemporaryIndexDirectory() as directory, requests.Session() as client:
        client.trust_env = False
        temporary = Path(directory)
        with RAGEngine(db_path=temporary / 'db', embedding_provider=embedding,
                       reranker_provider=reranker, llm_client=generator, enable_llm=True,
                       enable_hyde=False, verbose=False) as engine:
            def build(path, book, progress):
                engine.vectorizer.vectorize_book(path, book, on_progress=progress)
            importer = ImportManager(temporary / 'imports', build)
            app = create_api_app(engine, AccessGuard(GuardSettings(access_code=code, requests_per_window=100, daily_generations=1)),
                                 [], feedback_store=FeedbackStore(temporary / 'feedback.sqlite3'),
                                 import_manager=importer)
            sock = socket.socket()
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
            base = f'http://127.0.0.1:{port}'
            server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, loop='asyncio',
                                                  http='h11', log_level='error'))
            thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
            thread.start()
            deadline = time.monotonic() + 10
            while not server.started:
                assert time.monotonic() < deadline, 'Server did not start'
                time.sleep(.01)
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True, channel=os.environ.get('RAG_QA_BROWSER_CHANNEL') or None)
                    context = browser.new_context(viewport={'width':1280, 'height':900})
                    page = context.new_page()
                    page.set_default_timeout(15000)
                    errors, external = [], []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    def local_only(route):
                        if route.request.url.startswith(base + '/'):
                            route.continue_()
                        else:
                            external.append(route.request.url)
                            route.abort()
                    context.route('**/*', local_only)
                    page.goto(base + '/imports')
                    expect(page.locator('#import-code')).to_be_visible()
                    expect(page.locator('#import-connect')).to_be_enabled()
                    page.locator('#import-code').fill(code)
                    page.locator('#import-connect').click()
                    expect(page.locator('#import-prepare')).to_be_enabled()
                    page.locator('#import-book').select_option('os')
                    page.locator('#import-file').set_input_files({'name':'离线验收片段.json','mimeType':'application/json','buffer':payload})
                    page.locator('#import-prepare').click()
                    expect(page.locator('#import-job-count')).to_contain_text('3 个片段')
                    assert generator.calls == 0 and embedding.documents == 0
                    page.locator('#import-confirm').check()
                    page.locator('#import-start').click()
                    expect(page.locator('#import-return')).to_be_visible(timeout=30000)
                    page.screenshot(path=str(OUT/'import-complete.png'), full_page=True)
                    report['steps'].append('Actual upload/parse/confirmation/index publication')
                    page.locator('#import-return').click()
                    expect(page.locator('#book')).to_have_value('os')
                    expect(page.locator('#send')).to_be_enabled()
                    page.locator('#input').fill('什么是进程？')
                    page.locator('#send').click()
                    expect(page.locator('.answer')).to_contain_text('已保存到此浏览器')
                    expect(page.locator('.answer table')).to_have_count(1)
                    page.locator('.answer .cite').click()
                    expect(page.locator('.reader-quote')).to_be_visible()
                    page.get_by_role('button', name='读取相邻原文', exact=True).click()
                    expect(page.locator('.reader-current')).to_contain_text('定位片段与引用时的原文一致')
                    page.screenshot(path=str(OUT/'answer-reader.png'), full_page=True)
                    page.get_by_role('button', name='收起对照', exact=True).click()
                    page.locator('.answer').get_by_role('button', name='学习笔记', exact=True).click()
                    page.get_by_role('textbox', name='编辑学习笔记', exact=True).fill('离线学习流程验收笔记')
                    page.get_by_role('button', name='保存笔记', exact=True).click()
                    expect(page.locator('.note-editor')).to_contain_text('笔记已保存')
                    page.locator('.answer').get_by_role('button', name='☆ 收藏', exact=True).click()
                    expect(page.locator('.answer').get_by_role('button', name='★ 已收藏', exact=True)).to_be_visible()
                    with page.expect_download() as download:
                        page.get_by_role('button', name='保存问答（含来源）', exact=True).click()
                    download.value.save_as(OUT/'answer.md')
                    report['steps'].append('Actual SSE answer / indexed reader / notes / Markdown export')
                    page.locator('#toggle-history').click()
                    with page.expect_download() as download:
                        page.locator('#history-export-md').click()
                    download.value.save_as(OUT/'study-collection.md')
                    collection = (OUT/'study-collection.md').read_text()
                    assert '共 1 条记录' in collection and '离线学习流程验收笔记' in collection
                    assert code not in collection
                    report['steps'].append('Filtered Markdown collection preserves saved notes and sources without generation')
                    page.locator('.history-backup summary').click()
                    with page.expect_download() as download:
                        page.locator('#history-backup-all').click()
                    download.value.save_as(OUT/'history-backup.json')
                    backup_text = (OUT/'history-backup.json').read_text()
                    assert code not in backup_text
                    backup = json.loads(backup_text)
                    assert backup['version'] == 2 and backup['records'][0]['note'] == '离线学习流程验收笔记'
                    source = backup['records'][0]['result']['sources'][0]
                    page.locator('#history-review-start').click()
                    expect(page.locator('.answer')).to_have_count(0)
                    page.locator('#review-reveal').click()
                    page.locator('#review-again').click()
                    expect(page.locator('#review-panel')).to_be_hidden()
                    assert generator.calls == 1
                    report['steps'].append('Browser backup and local favorite review without extra generation')
                    page.goto(base+'/retrieval')
                    expect(page.locator('#inspect-send')).to_be_enabled()
                    page.locator('#inspect-book').select_option('os')
                    page.locator('#inspect-query').fill('进程')
                    page.locator('#inspect-send').click()
                    expect(page.locator('#inspect-status')).to_contain_text('检索观察完成')
                    expect(page.locator('#inspect-stages button')).to_have_count(5)
                    page.screenshot(path=str(OUT/'retrieval.png'), full_page=True)
                    assert generator.calls == 1
                    report['steps'].append('Actual five-stage inspection without generation')
                    page.goto(base+'/')
                    expect(page.locator('#send')).to_be_enabled()
                    page.locator('#input').fill('没有生成额度时保留教材资料')
                    page.locator('#send').click()
                    expect(page.locator('.answer')).to_contain_text('已保存到此浏览器')
                    expect(page.locator('.answer')).to_contain_text('今日的生成额度已经用完')
                    with page.expect_download() as download:
                        page.get_by_role('button', name='导出本次记录', exact=True).click()
                    download.value.save_as(OUT/'retrieval-only.md')
                    assert '本次没有完整的生成回答' in (OUT/'retrieval-only.md').read_text()
                    assert generator.calls == 1
                    report['steps'].append('Daily generation allowance exhausted: retrieval-only record export, no additional generation')
                    page.goto(base+'/imports')
                    expect(page.locator('#import-recent')).to_be_visible()
                    page.locator('#import-recent summary').click()
                    expect(page.locator('.recent-job')).to_have_count(1)
                    page.once('dialog', lambda dialog: dialog.accept())
                    page.locator('.recent-job').get_by_role('button', name='清理导入副本').click()
                    expect(page.locator('.recent-job')).to_have_count(0)
                    response = client.get(base+'/v1/books/os/passages', params={'chunk_id':source['chunk_id']}, headers=headers, timeout=5)
                    response.raise_for_status()
                    assert response.json()['source_sha256'] == source['source_sha256']
                    report['steps'].append('Import copy cleanup leaves published index readable')
                    assert not errors, errors
                    assert not external, external
                    context.close()
                    browser.close()
                    report.update(complete=True, generator_simulations=generator.calls,
                                  embedding_document_batches=embedding.documents, embedding_queries=embedding.queries,
                                  reranker_simulations=reranker.calls, browser_errors=errors, external_browser_requests=external)
            finally:
                server.should_exit = True
                thread.join(10)
                sock.close()
                importer.close()
                assert not thread.is_alive(), 'Server did not stop'
    if source_path is not None:
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
finally:
    report['application_sha256'] = {name: hashlib.sha256((Path(rag_textbook_qa.__file__).parent / name).read_bytes()).hexdigest() for name in ['api/app.py','api/imports.py','api/static/index.html','api/static/retrieval.html','api/static/imports.html','api/static/history-backup.js','api/static/markdown.js','rag/engine.py','rag/context.py','indexing/vectorizer.py']}
    (OUT/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    shutil.copyfile(__file__, OUT/'validation_harness.py')
print(json.dumps(report, ensure_ascii=False, indent=2))
