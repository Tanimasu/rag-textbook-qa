# 开发说明

目录结构、持续集成、工具脚本和评测数据集。

## 目录结构

```text
rag-textbook-qa/
├─ pyproject.toml               # Python 版本、依赖分组和命令入口
├─ src/rag_textbook_qa/         # 跨平台包（含 RAG、Provider、评估、CLI 和 Web UI）
├─ project/                     # 尚在迁移的解析工具和兼容入口
├─ data/
│  ├─ raw/                      # 本地 PDF 原书，不提交 Git
│  ├─ parsed/                   # PDF 解析后的 Markdown
│  ├─ cleaned/                  # 清洗后的 Markdown
│  ├─ chunks/                   # 分块 JSON
│  │  └─ previews/              # 分块文本预览
│  └─ evaluation/               # 评估问题集
├─ artifacts/
│  ├─ chunks/                   # 当前分块器生成的本地 chunks，不提交 Git
│  ├─ vector_db/                # 本地 ChromaDB，可重建且不提交 Git
│  └─ evaluations/              # RAGAS 和检索评估结果
├─ docs/                        # 使用手册、服务说明、评测方法与实验记录
├─ scripts/                     # 发布包检查、模型下载、Space 打包与实验脚本
├─ deploy/huggingface/          # Hugging Face Space 的说明卡片
├─ Dockerfile                   # 对外问答服务镜像（内置模型与索引）
└─ tests/                       # 无网络回归与历史资产基线
```

## 持续集成

GitHub Actions 会在每次 push 和 pull request 时执行以下离线验收：

- Windows、macOS、Linux 上的 Python 3.11 / 3.12 测试矩阵
- `src/`、`tests/` 与 `scripts/` 的 Ruff 静态检查
- 完整的无模型、无外部 API 单元与集成测试
- 真实 HTTP 连接验证队列饱和、排队断连及取消线程的执行槽释放
- 回环 Worker 截断响应的整批回退与恢复，以及真实 SDK 的生成终止标记处理
- Chromium 中运行真实聊天页，验证提交、流式回答、引用、重试、停止和反馈
- Python 源码编译检查
- source distribution 与 wheel 构建
- 发布包清单检查：wheel 与源码包中必需文件齐全，且不含本地数据、索引或密钥

本地可运行等价的核心检查：

```bash
uv sync --locked --extra ui --extra worker --extra api
uv run --no-sync python -m ruff check src tests scripts
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/check_serving_queue.py --output artifacts/evaluations/local/queue-http.json
uv run --no-sync python scripts/check_remote_transport.py --output artifacts/evaluations/local/remote-transport.json
uv run --no-sync python scripts/check_generation_transport.py --output artifacts/evaluations/local/generation-transport.json
uv run --no-sync python -m compileall -q src tests project
```

CI 通过 `uv.lock` 安装固定依赖，锁文件与 `pyproject.toml` 不一致时直接失败。
矩阵任务显式设置 `UV_PYTHON` 并断言实际解释器版本，避免 `.python-version` 的本机 3.11 设置覆盖 3.12。
构建使用锁定的 `build` 分组和 `python -m build --no-isolation`，避免构建时另行解析依赖。

浏览器测试独立运行，后端为本机假服务，浏览器执行仓库中的原始 HTML / JavaScript：

```bash
uv sync --locked --only-group browser
uv run --no-sync python -m playwright install chromium
uv run --no-sync python tests/browser/public_chat_browser.py -v
```

Linux CI 使用 `playwright install --with-deps chromium` 安装浏览器与系统库。
本机已有 Chrome 时可设置 `RAG_QA_BROWSER_CHANNEL=chrome`，省去浏览器下载。
`--only-group browser` 创建的环境仅含浏览器测试依赖，运行核心测试前需重新执行上面的完整同步命令。

CI 不读取 `project/.env`，也不会连接远程 Worker、调用 LLM API 或下载模型权重。

### 本机性能验收

`benchmark_retrieval.py` 使用 dev 题、当前索引副本及本机已缓存的模型，测量真实推理耗时。
脚本强制离线、不加载 `.env`；输出路径必须未使用过，以免覆盖旧记录。
复制索引前应停止索引写入；SQLite 备份与原生向量文件复制不是同一个原子操作。
先在已安装 `local-models` 且模型已缓存的环境中运行：

```bash
python scripts/benchmark_retrieval.py --device mps --baseline-ref f4606cc --repeats 2 --output artifacts/evaluations/local/retrieval-mps.json
```

设备可选 `cpu` / `mps`；省略 `--baseline-ref` 时只测当前代码。
`--reranker-batch-size` 默认 32，可用来比较不同推理批大小；历史引擎和当前引擎共用这一设置。
每本教材选一题，交替执行路线；每条路线的样本量较小，P95 只用于描述本次结果。
峰值内存覆盖整次运行，不能用它比较单条路线的内存。
队列脚本使用受控的无模型后端，其人工延迟不能作为真实问答延迟。

### 跨教材候选复核准备

已完成的 dev 对照报告可以导出未评分候选，不需要模型推理：

```bash
python scripts/prepare_crossbook_review.py --report artifacts/evaluations/retrieval-optimization-20261004/report.json --output-dir artifacts/evaluations/local/crossbook-review
```

工具要求索引版本与报告一致，按教材与片段编号合并两条全库路线的候选，保留完整正文与哈希。
`review.json` 随机排列候选，所有评分为空，不含路线与排名；对照关系单独存于 `manifest.json`。
复核时只打开 `review.json`，根据正文填写评分、原文证据和理由，完成后再查看排名。
这是已使用过的 dev 题候选池，不能作为新的独立测试集；未入池内容也不能直接记零分。

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `get_models.py` | 查询当前 API 端点支持的模型列表 |
| `test_llm_api.py` | 验证 LLM API 连通性与模型响应 |
| `extract_images.py` | 从 PDF 中提取图片为 PNG 文件 |
| `clean_db.py` | 管理 ChromaDB 集合（列出 / 删除） |
| `benchmark_retrieval.py` | 离线模型实际推理计时，支持历史 Git 版本对照 |
| `check_serving_queue.py` | 启动本机 HTTP API 验收有界队列与断连取消 |
| `check_index_lifecycle.py` | 连续检查两个 Chroma 场景，索引操作首次失败即退出；临时目录只等待 Windows 文件共享锁释放，最多一秒 |
| `prepare_crossbook_review.py` | 从锁定版本的 dev 对照报告导出未评分跨教材候选 |

## 评估数据集

| 文件 | 题数 | 说明 |
|------|------|------|
| `data/evaluation/test_questions.json` | 50 条 | RAGAS 使用，覆盖五本教材 |
| `data/evaluation/retrieval_questions.json` | 65 条 | 检索评测使用，dev 35 / holdout 30 |
| `data/evaluation/product_acceptance_v1.json` | 15 条 | 产品回归使用，每本教材 3 条，同时支持检索与回答验收 |
| `retrieval_holdout_candidates_v{1..4}.json` | 15 条 | 已并入主集的新增 holdout 题，保留审计记录 |

题集来源、holdout 中 15 题的调参污染范围、以及 2026-09-14 那次标注修正（属于标注修正，
不代表检索或答案质量提升），见
[标注集的由来与污染范围](evaluation-methodology.md#标注集的由来与污染范围)。
