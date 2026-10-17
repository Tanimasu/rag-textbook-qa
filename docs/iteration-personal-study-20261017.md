# 个人学习工作流验收 · 2026-10-17

本轮把 `rag-qa serve` 扩展为可连续使用的个人学习页面，当前版本 **0.3.10**。
从 `main` 的 `6ebc5b8` 开始，功能与修复保存在本地分支 `codex/history-favorites-20261017`；
0.3.10 功能提交为 `10430d6`。具体变更见 [小版本记录](release-notes.md)。

## 可以直接使用的功能

| 入口 | 行为 | 说明 |
| --- | --- | --- |
| 聊天页 | 提问草稿、编辑原问题、流式回答、表格与代码、Markdown 导出 | 提问仍独立检索，历史和笔记不自动发给模型 |
| 历史与收藏 | 自动保存完整结果，按关键词、教材、收藏和笔记筛选，删除与清理 | 保存在当前浏览器，口令不隔离本机历史 |
| 学习笔记 | 每条最多 4000 字，保留草稿，冲突时不覆盖最新内容 | 已保存笔记进入搜索与导出 |
| 备份与恢复 | 完整或按筛选导出 JSON，校验、预览后整批合并 | 相同编号保留现有版本，失败回滚新增记录 |
| 复习收藏 | 先回忆再显示回答，自评后安排明天或一周后再看 | 每轮最多 20 条，不调用模型、不推送通知 |
| 对照原文 | 引用快照与当前原文对照，顺序阅读，原文变化提示 | 无顺序 ID 只展示准确定位片段，不猜相邻内容 |
| `/retrieval` | 查看关键词、向量、融合、重排与最终上下文五阶段 | 不生成回答；Embedding 和重排仍用当前配置 |
| `/imports` | 上传 Markdown 或分块 JSON，先预览后确认建索引，查看进度和清理副本 | 只在本机服务启用，当前支持已有五本教材 |

日常启动：

```bash
rag-qa serve
```

默认打开 `http://127.0.0.1:8000`。如果原环境尚未安装 API 依赖，按 README 的安装步骤使用 `--extra api`。
详细保存范围及限制见 [历史与收藏](history-and-favorites.md)、[原文阅读](reading.md)、
[复习收藏](favorite-review.md)、[检索观察台](retrieval-observatory.md)、[本机导入](textbook-import.md)。

## 最终检查结果

| 检查 | 结果 |
| --- | --- |
| 单元与集成测试 | 652 项：641 通过，11 因未安装 RAGAS `eval` extra 跳过 |
| 原始页面与真实 Chrome | 69 项通过，覆盖桌面、手机、深色、存储故障和交互恢复 |
| 静态检查 | Ruff、compileall、`git diff --check` 通过 |
| 依赖锁 | `uv lock --check --offline` 通过，无新增运行依赖 |
| 发布清单 | wheel 的页面与脚本齐全；wheel / 源码包未包含本地索引、环境文件或运行产物 |
| 安装包串联 | 临时安装 0.3.10 wheel 后，以真实 HTTP、Chrome 和 Chroma 完成下述流程 |

浏览器回归覆盖中断或断流不保存半份答案、历史分页与筛选、跨标签页写入、备份整批回滚、
笔记冲突恢复、收藏复习、旧原文快照、自定义片段 ID、导入进度与旧响应失效、检索取消，
以及表格、代码和 320 像素窄屏页面。这里只验证功能，不据此声称模型回答质量或推理性能改善。

运行命令：

```bash
.venv/bin/python -m unittest discover -s tests -v
RAG_QA_BROWSER_CHANNEL=chrome \
  RAG_QA_BROWSER_SCREENSHOTS=artifacts/evaluations/study-workspace-20261017 \
  .venv/bin/python tests/browser/public_chat_browser.py -v
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m compileall -q src tests scripts
UV_CACHE_DIR=/tmp/rag-study-uv-cache uv lock --check --offline
git diff --check
```

## 安装包串联检查

使用现有 `data/chunks/操作系统_mineru_chunks.json` 的三个片段，只读取原文件，
导入、索引和反馈均放在临时目录。原文件前后 SHA-256 均为
`af0f06fb6d721f6f4741d232c984dfd906cfd2e7284bda772b5c1a060b01a62f`。

实际完成：上传 → 解析预览 → 确认 → Chroma 索引发布 → 提问与 SSE 返回 → 原文对照 →
笔记与收藏 → Markdown / JSON 导出 → 复习 → 五阶段检索观察 → 清理导入副本后继续读取已发布索引。
另将应用每日生成上限设为一次，验证额度用完后第二题仍可导出检索资料，且不会再次调用生成器。
这里的应用上限与 Codex 账户额度无关。

推理由本地确定性模拟器提供：一次模拟生成、一次文档向量化批次、三次查询向量化、三次模拟重排。
复习、原文阅读、导出和清理副本没有增加生成次数。页面错误和浏览器外部请求均为零。
未调用远程 Worker 或付费模型；没有覆盖原教材、现有索引或旧实验结果。

最终原始日志、文件散列及发布清单位于忽略目录
`artifacts/evaluations/study-workspace-20261017/checkpoint-0.3.10/`；
安装包验收脚本、截图、导出样例与 `report.json` 位于同级 `real-http-ui-v0310/`。
旧的阶段记录保留，没有用本轮结果覆盖。

完整串联脚本已整理为 `scripts/check_study_workflow.py`。需要已安装 API 依赖、`requests`、Playwright 和 Chrome，
输出目录必须不存在，脚本拒绝覆盖旧证据：

```bash
RAG_QA_BROWSER_CHANNEL=chrome .venv/bin/python scripts/check_study_workflow.py \
  --output artifacts/evaluations/study-workflow-new-run
```

也可以设置 `PYTHONPATH` 指向临时安装的 wheel 目录，检查发布包的相同流程。
2026-10-18 起，脚本默认使用内置合成资料，便于全新仓库与 CI 复跑；本页上面的历史验收确实使用了实际文件。
重做实际片段检查时加上 `--chunks data/chunks/操作系统_mineru_chunks.json`；`--workspace` 控制报告中显示的相对路径。

## 尚未提供

历史和笔记没有自动跨设备同步，重要记录需要下载备份。页面导入暂不直接解析 PDF，
也不支持任意新增教材名称。学习复习采用简单自评日期，不自动判断掌握程度。
本轮没有运行新的真实模型质量评测；HyDE、查询分解、引用核对和相邻上下文等实验策略仍默认关闭。
