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
```

该命令检查 Python、工作目录、基础依赖和可选组件，且不会加载模型或访问网络。安装本地模型组件后，可另外用 `project/check_env.py` 检查 PyTorch、CUDA 与 GPU。

---

## 本地与远程模型计算

Embedding 和 Reranker 使用统一 Provider 接口，可在两种模式间切换：

- `local`：模型运行在当前机器，可指定 `cpu`、`cuda` 或 `mps`。
- `remote`：Mac 只通过 Tailscale 请求 Windows Worker；ChromaDB、BM25、LLM 和 UI 仍在 Mac 本地。

向量化任务启动后会固定使用同一个后端，网络故障不会悄悄切换模型。查询可选择只在连接超时等瞬时故障时回退到相同的本地模型；认证失败和模型指纹不一致始终直接报错。

### Mac 本地 CPU 模式

先安装本地模型依赖：

```bash
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --inexact --extra local-models
```

在 `project/.env` 中设置：

```env
RAG_QA_COMPUTE_BACKEND=local
RAG_QA_DEVICE=cpu
RAG_QA_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RAG_QA_RERANKER_MODEL=BAAI/bge-reranker-base
```

### Windows 4070 Super Worker

Windows 拉取同一分支后，在 PowerShell 中创建环境并安装 Worker 与本地模型依赖：

```powershell
conda env create -f environment.yml
conda activate rag-textbook-qa
$env:UV_PROJECT_ENVIRONMENT=$env:CONDA_PREFIX
uv sync --inexact --extra worker --extra local-models
```

用 `python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成一个随机 token，并写入 Windows 的 `project/.env`。Worker token 必须是非空 ASCII 字符串，且不能包含首尾空格、内部空白或控制字符：

```env
RAG_QA_WORKER_TOKEN=替换为随机token
RAG_QA_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RAG_QA_RERANKER_MODEL=BAAI/bge-reranker-base
RAG_QA_DEVICE=cuda
```

通过 `tailscale ip -4` 查看台式机 Tailscale IP，然后只监听该地址：

```powershell
rag-qa worker serve --host 100.x.y.z --port 8765 --device cuda
```

配置完成后，也可以从仓库根目录直接运行一键启动脚本，不需要先执行
`conda activate`：

```powershell
.\scripts\windows\start-worker.ps1
```

脚本会根据自身位置定位仓库，自动查找 Conda 和 Tailscale IPv4，检查
`project/.env` 中是否存在格式有效的 Worker token，并在确认 8765 端口空闲后，
通过 `conda run` 启动 CUDA Worker。脚本不会显示 token，也不会修改防火墙、
开机启动项或持久环境变量。即使当前 PowerShell 中残留旧的 Worker 配置，脚本也会
仅为本次子进程清除这些覆盖值，以仓库的 `project/.env` 为准。

脚本也可以通过绝对路径从其他目录启动，或按需覆盖环境名和端口：

```powershell
& "D:\CodeField\rag-textbook-qa-worker\scripts\windows\start-worker.ps1"
.\scripts\windows\start-worker.ps1 -EnvironmentName rag-textbook-qa -Port 8765
```

若希望把首次请求的模型加载等待移到 Worker 启动阶段，可显式启用预热：

```powershell
.\scripts\windows\start-worker.ps1 -Warmup
# 等价的 CLI 参数：rag-qa worker serve ... --warmup
```

预热只会在 Windows 本地分别执行一次最小的 embedding 和 reranker 推理，
不会调用 LLM 或外部 API，也不会修改教材索引。启用后，Worker 会在两个模型
加载完成后再开始监听；未指定 `-Warmup` 时仍保持原有的首次请求懒加载行为。

不要把 Worker 端口映射到公网。监听非 localhost 地址时，程序会强制要求 `RAG_QA_WORKER_TOKEN`。如果 PowerShell 或终端进程中的 token 与 `project/.env` 不同，进程环境变量优先，命令会给出不含 token 内容的警告；修改 token 后应重启 Worker。

### Mac 连接远程 Worker

在 Mac 的 `project/.env` 写入相同 token 和 Windows Tailscale 地址：

```env
RAG_QA_COMPUTE_BACKEND=remote
RAG_QA_REMOTE_URL=http://100.x.y.z:8765
RAG_QA_WORKER_TOKEN=与Windows相同的随机token
RAG_QA_REMOTE_TIMEOUT=120
RAG_QA_QUERY_FALLBACK_TO_LOCAL=false
RAG_QA_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RAG_QA_RERANKER_MODEL=BAAI/bge-reranker-base
```

先执行安全健康检查：

```bash
rag-qa worker check
# 或输出结构化结果
rag-qa worker check --json
```

该命令只请求 `/health`，校验认证、协议版本、设备以及 embedding/reranker 模型指纹，不会调用推理接口，也不会输出 token。`rag-qa doctor` 则只检查当前选择的后端配置，不会连接 Worker。Worker 首次收到 embedding 或 rerank 请求时才会加载并下载模型。若要启用查询回退，Mac 还需安装 `local-models`，并将 `RAG_QA_QUERY_FALLBACK_TO_LOCAL` 改为 `true`。

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

**生成层**（`rag-qa evaluate-generation`）针对采样噪声：同一个提示词在温度 0.7 下会得出不同回答，
2026-09-14 的对照里有一题两轮输入逐字相同而审查结论翻转，所以单次采样的 A/B 对照无法归因。
它冻结生成模型实际看到的上下文，每题重复采样，再由另一家族的评判模型把回答拆成事实陈述逐条
核对；判为有依据的陈述必须附原文摘录，程序在上下文里找到该摘录才算数。方案之间先按题取平均，
再做配对的符号翻转检验。输出目录可断点续跑，首次运行即冻结实验协议，换了参数的重跑会被拒绝。

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
上下文默认限制为 2000 字符，超过剩余预算的正文会截取可容纳的部分。

启动命令会检查当前模式所需的依赖并显示不含 token 的配置摘要，但不会主动连接 Worker 或加载模型。远程模式只需安装 `ui`，本地模式以及启用本地回退时还需安装 `local-models`。界面支持教材选择、top-k 调整、对话历史与 RAGAS 评估结果查看。

每次问答完成后，答案下方会显示本次请求的安全执行摘要：Embedding 与 Reranker 的实际后端、设备、Worker 平台、调用次数和耗时，以及检索、回答与总耗时。例如，远程正常时显示“远程 Worker（Windows）· CUDA”；瞬时网络故障触发回退时显示“已回退到本地（macOS）· MPS”。摘要不会包含 Worker URL、token、API Key、问题正文或模型输入。旧版 Worker 未返回平台字段时仍可显示“远程 Worker · CUDA”；Windows 更新代码并重启 Worker 后会补充平台名称。

Web 界面的正式实现位于 `src/rag_textbook_qa/web/`；`project/app.py` 与原来的 `project/ui/`、`project/services/app_services.py` 仅保留为兼容入口。

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
| `data/evaluation/test_questions.json` | 50 条 | 覆盖五本教材，`ragas_evaluation.py` 默认使用 |
| `data/evaluation/retrieval_questions.json` | 65 条 | 五本教材各13题，dev 35 / holdout 30；holdout中15题曾参与调参，须与新增15题分别解释 |
| `data/evaluation/retrieval_holdout_candidates_v4.json` | 15 条 | 经过人工初览并纳入主标注集的新增holdout题；此文件保留审计记录 |

2026-09-14 修正两道数据库开发题的章节标注，原因及同排序计分对照见 [数据库证据复核](docs/database-evidence-review.md)。这属于标注修正，不代表检索或答案质量提升；holdout未改动。


### 表格回答上下文

完整 HTML 表格在生成前转换为紧凑行列文本，合并单元格按占用位置展开，
只纳入预算内的完整行；省略后续行时明确标记。含图片、标题或无法安全解释的结构暂时跳过，
继续尝试后续资料；如果没有可用证据，不调用生成模型。
界面引用与 RAGAS 使用的仍是实际送入模型的片段，原始检索结果保留不变。
这项改动只作用于回答上下文，不改变索引；长表格在 embedding 阶段的输入截断仍需另行验证。

### 当前稳定使用方式

普通问答使用原问题检索和重排，默认关闭 HyDE、查询分解和答案引用核对。
查询分解、引用核对收纳在侧栏「实验功能（默认关闭）」中。
模型服务或引擎初始化出现常见异常时，页面显示可读的失败提示，可以继续提问。
当前工作区的检查结果、快照与已知限制见 [稳定基线](docs/stable-baseline.md)。
普通问答的最新真实API验收见 [15题主线验收](docs/mainline-acceptance-20260914.md)：接口15/15正常结束，答案依据与完整性仍有待修正项。

### 已知教材冲突保护

普通问答在生成前检查实际上下文中的已登记冲突。目前覆盖数据库教材中UNIQUE／唯一索引的NULL数量分歧；只有经核实的两侧片段及完整原文都在上下文中时触发。
触发后不拦截回答：两侧原文和引用编号会作为附加要求写进提示词，要求回答并列给出两种说法、说明仅凭本次片段无法确定统一结论，其余部分照常作答。
没有匹配到登记规则不代表不存在冲突。这是针对已知教材问题的保护，不是通用语义冲突检测；详见 [证据复核与处理](docs/database-evidence-review.md)。

### 实验功能

- **查询分解**：`engine.ask(..., use_decomposition=True)`。最多3个独立子问题，同时保留原问题检索，合并后统一重排；优先保留每个子问题的完整证据。简单题可本地跳过规划，规划失败恢复原检索方式。已完成小样本回归，尚未证明整体答案准确率提升。见 [实现与验证记录](docs/query-decomposition-plan.md)。
- **答案引用核对**：`engine.ask(..., verify_citations=True)`。草稿生成后最多增加两次模型请求，提取陈述与原文摘录、校验摘录、核对支持关系，最后直接渲染通过的陈述。开启时等待核对完成再显示；格式错误、超时或无通过陈述时不展示草稿。两轮使用同一配置模型，仍可能误判或遗漏信息。目前仅通过离线流程测试，尚无真实效果验收。

`grounding` 返回核对状态、通过的陈述及摘录、调用数与耗时；答案 token 字段仍仅统计草稿生成。
引用编号存在、片段完整或模型核对通过，都不等于答案中的每个细节已被原文充分支持。
实验记录入口见 [稳定基线](docs/stable-baseline.md#历史验证记录)。
