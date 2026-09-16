# 计算机教材 RAG 问答系统

[![CI](https://github.com/Tanimasu/rag-textbook-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanimasu/rag-textbook-qa/actions/workflows/ci.yml)

> 面向计算机课程教材的检索增强生成（RAG）问答系统

基于混合检索策略（语义向量 + BM25）与大语言模型，实现对操作系统、计算机组成原理等教材内容的精准问答。系统采用 HyDE 查询增强与 Cross-Encoder 重排序，在 50 题 RAGAS 评估集（5 本教材各 10 题）上取得如下结果：

> 生成模型：`gemini-3-flash-preview`（Google）；评判模型：`claude-sonnet-4-6`（Anthropic）—— 跨厂商配对以规避自我偏好偏差
>
> 该表为上下文策略调整之前的口径，与当前版本的运行结果不可直接比较。

| 指标 | 得分 | 评级 |
|------|------|------|
| Answer Relevancy | 0.8908 | 优秀 |
| Faithfulness | 0.7984 | 良好 |
| Context Recall | 0.7450 | 良好 |
| Context Precision | 0.6827 | 及格 |

检索层在 65 题标注集上的结果（不调用 LLM；Hybrid + Reranker 一行跨进程有约一道题的抖动，原因尚未定位）：

| 策略 | Recall@5 | MRR |
|------|----------|-----|
| BM25 | 0.677 | 0.484 |
| Embedding | 0.800 | 0.639 |
| Hybrid | 0.800 | 0.660 |
| Hybrid + Reranker | 0.877 | 0.687 |

按 65 题全集统计，数据来自 2026-09-13 的运行，早于次日两道数据库开发题的标注修正。
所有策略跑同一批问题，配对检验后只有「Hybrid + Reranker 优于纯 BM25」达到显著（1:14，p=0.0010），
其余差异都不足以支撑结论——表中数值差异小于阈值时不应解读为提升，依据见
[评测方法与结果](docs/evaluation-methodology.md#差异要用配对检验)。

评测报告里也记录了每题的检索耗时，但该数值受远程 Worker 负载影响波动较大，不宜作为系统指标解读。

---

## 系统概览

```text
PDF ─→ Markdown ─→ 清洗 ─→ 按标题分块 ─→ ChromaDB（每本教材一个集合）
                                              │
       回答 ←─ LLM ←─ 上下文打包 ←─ 重排 ←─ 混合检索（向量 + BM25）
```

检索链路四步：

1. **HyDE**（默认关闭）：先让 LLM 写一段假设性教材原文，用它的向量去检索，改善复杂问题的语义匹配。
   它是文档不是查询，因此用 `embed_documents` 嵌入，关闭时走带指令前缀的 `embed_queries`
2. **混合检索**：向量与 BM25 各自取候选，按**名次**用加权 RRF 融合（BM25 权重默认 0.5），去重并过滤
   习题。按名次而非分数融合是刻意的：余弦相似度有界而 BM25 无界、会随语料漂移。BM25 用词 + 字符
   二元组混合分词，避免 jieba 把教材术语切碎（散列表 → 散/列表）
3. **重排**：`BAAI/bge-reranker-base` 对候选精排，启用时由它决定最终顺序
4. **生成**：按字符预算把片段打包成上下文，拼进提示词交给 LLM，答案标注【参考资料 N】

每次回答会显示 Embedding/Reranker 实际跑在远程 CUDA 还是本地 MPS/CPU，以及检索、生成和总耗时。

**技术栈**：Python + Streamlit；解析用 Docling / MinerU / EasyOCR；检索用 ChromaDB、
sentence-transformers（`BAAI/bge-large-zh-v1.5`）、rank-bm25 + jieba；重排用
`BAAI/bge-reranker-base`；LLM 走 OpenAI-compatible API；评估用 RAGAS。

实现在 `src/rag_textbook_qa/`；`project/` 下的同名脚本是迁移期保留的兼容入口，不是主实现。

---

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
└─ tests/                       # 无网络回归与历史资产基线
```

---

## 快速开始

### 1. 安装依赖

项目要求 Python 3.11 或 3.12。推荐由 Conda 管理 Python 环境、uv 管理项目锁文件和 Python 依赖。请先在各自系统安装 Conda 和 uv，然后运行：

```bash
conda env create -f environment.yml
conda activate rag-textbook-qa
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --inexact
```

`UV_PROJECT_ENVIRONMENT` 让 uv 直接使用当前 Conda 环境，不创建第二个 `.venv`；`--inexact` 保留 Conda 管理的 Python 基础包。Windows PowerShell 对应写法：

```powershell
$env:UV_PROJECT_ENVIRONMENT=$env:CONDA_PREFIX
uv sync --inexact
```

UI、本地模型、远程 Worker、PDF 解析和评估依赖通过 `--extra ui`、`--extra local-models`、`--extra worker`、`--extra docling`、`--extra mineru`、`--extra eval` 按需安装。例如，在 Mac 本地运行 Streamlit 与模型时使用：

```bash
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --inexact --extra ui --extra local-models
```

基础开发环境不会安装 PyTorch，也不会下载模型权重。

### 2. 配置 API

```bash
cp project/.env.example project/.env
```

编辑 `project/.env`：

```env
# 共享配置（所有脚本默认使用）
LLM_API_KEY=your_api_key_here
LLM_API_BASE=https://api.ohmygpt.com/v1
LLM_MODEL=gemini-3-flash-preview

# 可选：为 RAG 引擎和评估器单独指定模型（不设则使用上方共享值）
# RAG_MODEL=gemini-3-flash-preview
# RAGAS_MODEL=claude-sonnet-4-6   # 建议与生成模型使用不同厂商，避免自我偏好偏差
# RAGAS_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

### 3. 验证环境

```bash
rag-qa doctor
rag-qa doctor --index   # 额外检查向量库内容
```

该命令检查 Python、工作目录、基础依赖和可选组件，且不会加载模型或访问网络。安装本地模型组件后，可另外用 `project/check_env.py` 检查 PyTorch、CUDA 与 GPU。

加上 `--index` 会多做一层：列出每个 `textbook_*` 集合的条数与 embedding 模型、标出空集合，
并核对已登记的冲突规则是否仍能命中。默认的 `doctor` 只看配置，`artifacts/vector_db/` 目录存在
就报 ok，空索引和缺书都察觉不到。这一层要导入 chromadb（仍不加载模型），所以做成可选项。

---

## 本地与远程模型计算

Embedding 和 Reranker 走统一 Provider 接口，可在两种模式间切换：`local` 在当前机器上运行
（`cpu`/`cuda`/`mps`）；`remote` 让 Mac 只通过 Tailscale 请求 Windows Worker，而 ChromaDB、
BM25、LLM 和界面仍留在本地。

向量化任务一旦启动就固定使用同一个后端，网络故障不会悄悄换模型。查询可以只在连接超时这类
瞬时故障时回退到相同的本地模型；**认证失败和模型指纹不一致始终直接报错，不降级**。

两端的环境变量、Windows Worker 的一键启动脚本与安全约束、以及 `rag-qa worker check` 的用法，
见 [远程 Worker 部署](docs/remote-worker.md)。


---

## 完整流程

以下命令默认从仓库根目录运行，输入和输出路径不依赖当前操作系统。

### Step 1 — PDF 转 Markdown

将 PDF 放入 `data/raw/`。当前兼容脚本默认处理“数据库原理及应用教程.pdf”：

```bash
python project/parsingPDF_mineru.py   # 推荐：MinerU
python project/parsingPDF.py          # 备选：Docling + EasyOCR
```

MinerU 版本输出到 `data/parsed/*_mineru.md`，Docling 版本输出到 `data/parsed/*.md`。

### Step 2 — 清洗 Markdown

```bash
rag-qa ingest clean data/parsed/教材.md --output data/cleaned/教材_cleaned.md
```

通过 SmartMarkdownCleaner 规范化标题层级，显式写入 `data/cleaned/`。

### Step 3 — 文本分块

```bash
rag-qa ingest chunk data/cleaned/教材_cleaned.md \
  --output artifacts/chunks/教材_chunks.json
rag-qa ingest check artifacts/chunks/教材_chunks.json
```

按标题结构切分 Markdown，并将 JSON 写入 `--output` 指定的位置。

仓库现有的十份 `data/chunks/*.json` 是受测试保护的毕业设计原始资产。验证新版分块器或重新构建当前索引时，应将输出写入 `artifacts/chunks/`，避免覆盖原始基线：

```bash
rag-qa ingest chunk data/cleaned/数据结构_mineru_cleaned.md \
  --output artifacts/chunks/数据结构_mineru_chunks.json
rag-qa ingest check artifacts/chunks/数据结构_mineru_chunks.json
```

新版分块器保留无法安全合并的短块，仅在同一标题路径内合并；普通长文本实际应用重叠窗口。
围栏代码、独立公式和 HTML 表格会作为整体保留，但只有这些片段本身不切分，周围正文照常分割，
因此一个段落不会再因为含有一处公式就整段超长。单张超过块大小上限的表格仍会完整保留，
因为按行切开会破坏表格语义；这类块的尾部超出 embedding 模型输入上限，只能靠 BM25 命中。
修改分块参数后，需要重新生成 chunks 并重建索引才能影响问答；已有索引不会自动更新。

质量检查会同时报告过大/过小块、代码截断、章节编号继承冲突和重复内容。

### Step 4 — 向量化

```bash
rag-qa index build artifacts/chunks/教材_chunks.json
rag-qa index list
rag-qa index check
```

`index build` 根据 `project/.env` 选择本地或远程 embedding Provider，将向量写入 `artifacts/vector_db/`；默认先完整构建临时集合，成功后再替换旧集合。已知教材会根据文件名推断稳定 ID，也可用 `--book database` 显式指定。原来的 `python project/vectorize_chunks.py` 仍保留为批量交互式兼容入口。

`index check` 核对 `rag/conflicts.py` 里已登记的冲突规则：把每条规则的片段ID拿到当前索引里查，
再确认完整原文摘录仍在该片段中，找不到片段或原文已改写就逐条打印并以退出码 1 结束。
片段ID由分块决定，重新分块后同一段原文会换一个ID，冲突保护会就此静默失效而回答看起来一切正常——
这个命令把静默失效变成明确报错。它只读取集合内容，不加载模型、不调用 LLM，也不消耗 token。
`index build` 成功后会自动对刚重建的那本教材跑一遍同样的核对，失效时把警告打到 stderr，
但不改变退出码——索引本身确实建成了，失效的是冲突规则，两件事不该混为一个结果。

### Step 5 — 问答

```bash
rag-qa chat
```

启动交互式问答。输入 `test` 可运行内置测试用例，输入 `quit` 退出；仅检查检索流程时可使用 `rag-qa chat --no-llm --no-hyde`。`RAGEngine` 正式实现位于 `src/rag_textbook_qa/rag/`，原来的 `python project/rag_engine.py` 保留为兼容入口。

### Step 6 — 评估

评测分三层，各自回答不同的问题：确定性的检索评测用来迭代，RAGAS 用来阶段验收，
采样式的生成评测用来判断回答本身。口径、依据和已知限制见
[评测方法与结果](docs/evaluation-methodology.md)。

```bash
rag-qa evaluate-retrieval --strategy all --top-k 5   # 确定性，不调用 LLM，不消耗 token
rag-qa evaluate                                      # RAGAS 验收，一轮一小时以上
rag-qa evaluate-retrieval --split holdout            # 仅在确认最终结论时使用
```

**检索层**读取 `data/evaluation/retrieval_questions.json`（65 题，dev 35 / holdout 30），
比较 BM25、Embedding、Hybrid 和 Hybrid + Reranker，输出 Recall@K、Hit@K、MRR、nDCG@K、
证据保留率与检索耗时。`--split` 默认 dev，调参过程读不到留出集。评测关闭 HyDE、不调用 LLM；
远程 Worker 不可用时直接报错而不静默回退到本地，以保证结果可比。报告写入
`artifacts/evaluations/retrieval/`。

策略间的差异比看上去小，**必须用配对检验判断**，不能对比两个均值就下结论。
含重排的结果跨进程有约一道题的抖动，原因尚未定位。这两点都会改变结论的读法，详见上面的方法文档。

**RAGAS 层**读取 `data/evaluation/test_questions.json`，结果写入 `artifacts/evaluations/`。
测试时务必用 `--output-dir` 重定向，避免覆盖正式结果；依赖按需安装 `uv sync --inexact --extra eval`。
两个容易踩的配置：

- 评判模型用 `RAGAS_MODEL` 单独指定，应与生成模型分属不同厂商以规避自我偏好偏差。若它默认开启
  思考模式（Qwen3 系列如此），须设 `RAGAS_DISABLE_THINKING=true`，否则 faithfulness 会因超时
  全部变成 NaN——实测同一道判断题，开思考耗时 7.2 秒且判错，关闭后 0.5 秒且判对。
- 四项指标里只有 answer_relevancy 依赖向量模型，且单次采样噪声很大（同一批答案重测两次平均
  绝对差 0.033，最差一题 0.367），因此默认独立重复 `RAGAS_RELEVANCY_SAMPLES`（3）轮取平均。

RAGAS 是验收指标而非优化目标，验收标准须在看到数字之前约定，
见[验收标准](docs/evaluation-methodology.md#ragas-验收标准)。
需要无 RAG 基线对比时用 `rag-qa evaluate --baseline`（额外消耗 token）。

**生成层**（`rag-qa evaluate-generation`）针对采样噪声：温度 0.7 下同一个提示词会得出不同回答，
2026-09-14 有一题两轮输入逐字相同而审查结论翻转，所以单次采样的 A/B 对照无法归因。它冻结上下文、
每题重复采样，由另一家族的评判模型把回答拆成事实陈述逐条核对，最后做配对的符号翻转检验。

```bash
rag-qa evaluate-generation --cases <cases.json> \
  --arm t07=baseline@0.7 --arm t02=baseline@0.2 --samples 5 --output-dir <dir>
```

写成 `--arm 名称=上下文@stored` 时不调用生成模型，直接评判保存过的回答。
首次评判器校验（2026-09-15）**未通过**预先登记的门槛，其结论暂不作为决策依据，
见 [评判器校验记录](docs/generation-judge-validation-20260915.md)。

原来的 `python project/ragas_evaluation.py` 保留为兼容入口，延续同时运行 baseline 的旧行为。

### Step 7 — 启动 Web 界面

```bash
rag-qa app
```

`rag-qa app` 会自动定位工作区并读取 `project/.env`，因此不要求终端当前位于仓库根目录。也可以只对本次启动覆盖计算后端，不会改写 `.env`：

```bash
# Mac 本地 CPU
rag-qa app --backend local --device cpu

# Windows 远程 GPU Worker
rag-qa app --backend remote

# 不自动打开浏览器，并覆盖监听地址与端口
rag-qa app --no-browser --host 127.0.0.1 --port 8501
```

Web 问答默认使用流式输出，答案会在模型生成过程中逐步显示。高级参数中的
“启用 HyDE 增强检索”默认关闭；开启后会在每次检索前额外调用一次 LLM，可能
提高部分复杂问题的召回效果，但会增加等待时间和 API 费用。流式回答完成后，
执行摘要还会显示首字等待时间。答案引用区只显示实际送入模型的资料片段；
上下文默认限制为 4000 字符（`--context-budget` 可调），超过剩余预算的正文会截取可容纳的部分。
这个默认值 2026-09-15 由 2000 提高：实测 2000 时 191 条相关片段只有 154 条真正装进上下文，
4000 则全部送达而平均上下文只从 1601 涨到 2336 字符。依据是证据保留率，**答案质量未验证**，
且此后的 RAGAS 分数与更早的运行不可比。

启动命令会检查当前模式所需的依赖并显示不含 token 的配置摘要，但不会主动连接 Worker 或加载模型。远程模式只需安装 `ui`，本地模式以及启用本地回退时还需安装 `local-models`。界面支持教材选择、top-k 调整、对话历史与 RAGAS 评估结果查看。

每次问答完成后，答案下方会显示本次请求的安全执行摘要：Embedding 与 Reranker 的实际后端、设备、Worker 平台、调用次数和耗时，以及检索、回答与总耗时。例如，远程正常时显示“远程 Worker（Windows）· CUDA”；瞬时网络故障触发回退时显示“已回退到本地（macOS）· MPS”。摘要不会包含 Worker URL、token、API Key、问题正文或模型输入。旧版 Worker 未返回平台字段时仍可显示“远程 Worker · CUDA”；Windows 更新代码并重启 Worker 后会补充平台名称。

Web 界面的正式实现位于 `src/rag_textbook_qa/web/`；`project/app.py` 与原来的 `project/ui/`、`project/services/app_services.py` 仅保留为兼容入口。

---

## 对外问答服务

```bash
uv sync --inexact --extra api --extra local-models
rag-qa serve                     # http://127.0.0.1:8000
```

一个进程同时提供聊天页（`/`）、REST API（`/v1/ask`、`/v1/ask/stream`、`/v1/books`、`/v1/feedback`）和自动生成的
接口文档（`/docs`）。公开接口只开放 `query`、`book_id`、`top_k`：查询分解、引用核对和 HyDE 一律关闭，
因为它们未通过验收且会额外调用模型；上游错误文本不会出现在任何响应里。

每次回答会得到一个随机、短期有效的 `answer_id`。页面提供复制、赞和踩；用户主动提交反馈后，
服务才会把这次问题、公开答案、公开引用、冲突提示、耗时和反馈持久保存到
`artifacts/product/feedback.sqlite3`。未提交的回答只在当前进程的有界内存中短暂保留；反馈数据不记录
IP、访问口令、API Key、Worker token 或模型内部提示词。该目录默认被 Git 忽略。

需要检查或分析反馈时，可导出为 JSONL；默认不会覆盖已有文件：

```bash
# 先看不包含问题和答案正文的聚合摘要
rag-qa feedback summary

# 把负面反馈整理为待人工标注的候选，不会修改正式评测集
rag-qa feedback candidates --output artifacts/product/feedback-candidates.json

# 需要逐条分析时再导出明细
rag-qa feedback export --output artifacts/product/feedback-export.jsonl
# 明确需要覆盖同名文件时
rag-qa feedback export --output artifacts/product/feedback-export.jsonl --force
```

反馈用于后续人工归类和离线评测，不会在在线回答中自动运行 RAGAS，也不会自动改变检索参数。
候选文件中的 `relevant_sections`、`ground_truth` 和审核备注默认留空；只有人工对照教材完成标注后，
才应另行迁移到正式评测集，避免随手差评污染实验数据。“速度太慢”会标记为性能检查候选，
不应当迁移成回答质量题。

对外开放前要配好费用控制，全部通过环境变量：

| 变量 | 默认 | 作用 |
|---|---|---|
| `RAG_QA_ACCESS_CODE` | 不设 | 访问口令（仅 ASCII），放在请求头 `X-Access-Code` |
| `RAG_QA_RATE_LIMIT` / `RAG_QA_RATE_WINDOW_SECONDS` | 10 / 600 | 每个 IP 的提问滑动窗口限流 |
| `RAG_QA_FEEDBACK_RATE_LIMIT` | 30 | 同一窗口内单独计算的反馈提交限流，不占用提问次数 |
| `RAG_QA_DAILY_GENERATIONS` | 200 | 每日生成上限，用完后只返回检索到的原文、不调用大模型 |
| `RAG_QA_TRUST_PROXY` | false | 前面恰有一层可信代理时才开启 |

监听非本机地址时，必须设置 `RAG_QA_ACCESS_CODE`，或显式加 `--public` 确认无口令开放。
计数器存在进程内存里，重启即清零，只适合单进程演示。

---

## 持续集成

GitHub Actions 会在每次 push 和 pull request 时执行以下离线验收：

- Windows、macOS、Linux 上的 Python 3.11 / 3.12 测试矩阵
- `src/` 与 `tests/` 的 Ruff 静态检查
- 完整的无模型、无外部 API 单元与集成测试
- Python 源码编译检查
- source distribution 与 wheel 构建
- wheel 中 CLI、Worker 和 Web UI 文件的完整性检查

本地可运行等价的核心检查：

```bash
python -m ruff check src tests
python -m unittest discover -s tests -v
python -m compileall -q src tests project
python -m build
```

CI 不读取 `project/.env`，也不会连接远程 Worker、调用 LLM API 或下载模型权重。

---

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `get_models.py` | 查询当前 API 端点支持的模型列表 |
| `test_llm_api.py` | 验证 LLM API 连通性与模型响应 |
| `extract_images.py` | 从 PDF 中提取图片为 PNG 文件 |
| `clean_db.py` | 管理 ChromaDB 集合（列出 / 删除） |

---

## 评估数据集

| 文件 | 题数 | 说明 |
|------|------|------|
| `data/evaluation/test_questions.json` | 50 条 | RAGAS 使用，覆盖五本教材 |
| `data/evaluation/retrieval_questions.json` | 65 条 | 检索评测使用，dev 35 / holdout 30 |
| `retrieval_holdout_candidates_v{1..4}.json` | 15 条 | 已并入主集的新增 holdout 题，保留审计记录 |

题集来源、holdout 中 15 题的调参污染范围、以及 2026-09-14 那次标注修正（属于标注修正，
不代表检索或答案质量提升），见
[标注集的由来与污染范围](docs/evaluation-methodology.md#标注集的由来与污染范围)。

---

## 当前行为与已知限制

**表格**：完整 HTML 表格在生成前转为紧凑行列文本，合并单元格按占用位置展开，只纳入预算内的完整行，
省略时明确标记；含图片或无法安全解释的结构整块跳过而不是切碎，没有可用证据时不调用生成模型。
只作用于回答上下文，不改索引；长表格在 embedding 阶段的输入截断仍需另行验证。

**默认路径**：普通问答用原问题检索加重排，默认关闭 HyDE、查询分解和引用核对。最新真实 API 验收见
[15 题主线验收](docs/mainline-acceptance-20260914.md)——接口 15/15 正常结束，但答案依据与完整性
仍有待修正项；工作区快照见 [稳定基线](docs/stable-baseline.md)。

**已知教材冲突**：只覆盖数据库教材 UNIQUE／唯一索引的 NULL 数量分歧，且两侧原文都进入上下文才触发；
触发后不拦截回答，而是要求并列给出两种说法。**没有匹配到规则不代表不存在冲突**——这是针对一处已核实
问题的硬编码保护，不是通用冲突检测。详见 [证据复核](docs/database-evidence-review.md)。

**实验功能（默认关闭，都没通过真实效果验收）**：`use_decomposition=True` 最多拆 3 个子问题并保留
原问题检索，合并后统一重排（[记录](docs/query-decomposition-plan.md)）；`verify_citations=True`
在草稿后追加最多两次模型请求核对引用，核对未完成时不展示草稿。引用编号存在、或模型核对通过，
都不等于答案里每个细节都有原文支撑。
