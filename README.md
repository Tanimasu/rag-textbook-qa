# 计算机教材 RAG 问答系统

> 面向计算机课程教材的检索增强生成（RAG）问答系统

基于混合检索策略（语义向量 + BM25）与大语言模型，实现对操作系统、计算机组成原理等教材内容的精准问答。系统采用 HyDE 查询增强与 Cross-Encoder 重排序，在 50 题 RAGAS 评估集（5 本教材各 10 题）上取得如下结果：

> 生成模型：`gemini-3-flash-preview`（Google）；评判模型：`claude-sonnet-4-6`（Anthropic）—— 跨厂商配对以规避自我偏好偏差

| 指标 | 得分 | 评级 |
|------|------|------|
| Answer Relevancy | 0.8908 | 优秀 |
| Faithfulness | 0.7984 | 良好 |
| Context Recall | 0.7450 | 良好 |
| Context Precision | 0.6827 | 及格 |

---

## 项目亮点

- **面向教材问答的完整 RAG 流水线**：覆盖 PDF 解析、Markdown 清洗、分块、向量化、检索、生成与评估
- **混合检索策略**：融合语义向量检索与 BM25 关键词匹配，兼顾语义相关性与术语命中率
- **HyDE 查询增强**：先由 LLM 生成假设性教材原文，再进行向量检索，提升复杂问题的召回效果
- **Cross-Encoder 重排序**：使用 `BAAI/bge-reranker-base` 对候选片段精排，提升最终上下文质量
- **多教材独立向量库**：支持操作系统、计算机组成原理、计算机网络、数据结构、数据库原理及应用等多本教材
- **评估闭环完整**：集成 RAGAS 指标评估，并支持无 RAG baseline 对比
- **可视化交互界面**：基于 Streamlit 提供教材选择、参数调节、问答对话和评估结果查看

---

## 技术栈

- **语言与应用层**：Python、Streamlit
- **文本解析与预处理**：Docling、MinerU、EasyOCR
- **向量化与检索**：ChromaDB、sentence-transformers、`BAAI/bge-large-zh-v1.5`
- **关键词检索**：rank-bm25、jieba
- **重排序模型**：`BAAI/bge-reranker-base`
- **大语言模型接入**：OpenAI-compatible API、openai SDK
- **评估框架**：RAGAS、LangChain OpenAI、datasets
- **数据处理**：pandas、openpyxl、tqdm

---

## 系统架构

```
PDF
 ├─ parsingPDF.py         # Docling + EasyOCR → Markdown
 └─ parsingPDF_mineru.py  # MinerU (推荐，扫描页更完整) → *_mineru.md
     └─ clean_markdown.py     # 标题层级规范化 → *_cleaned.md
         └─ chunk_textbooks.py    # 按标题结构分块 → *_chunks.json
             └─ vectorize_chunks.py   # BAAI/bge-large-zh-v1.5 → ChromaDB
                 └─ rag_engine.py         # 混合检索 + HyDE + Reranker + LLM
                     └─ app.py                # Streamlit 问答界面
```

**检索流程**

1. **HyDE**：用 LLM 将问题改写为假设性教材原文，用其嵌入向量检索，提升语义匹配质量
2. **混合检索**：向量相似度（权重 1.0）与 BM25 关键词匹配（权重 0.3）融合排序
3. **Cross-Encoder 重排序**：`BAAI/bge-reranker-base` 对候选结果精排，取最优 top-k
4. **LLM 生成**：将检索上下文与问题拼接为 Prompt，调用 LLM 生成结构化答案

---

## 目录结构

```text
rag-textbook-qa/
├─ pyproject.toml               # Python 版本、依赖分组和命令入口
├─ src/rag_textbook_qa/         # 新的跨平台包
├─ project/                     # 尚在迁移的 RAG、评估和 Streamlit 兼容入口
├─ data/
│  ├─ raw/                      # 本地 PDF 原书，不提交 Git
│  ├─ parsed/                   # PDF 解析后的 Markdown
│  ├─ cleaned/                  # 清洗后的 Markdown
│  ├─ chunks/                   # 分块 JSON
│  │  └─ previews/              # 分块文本预览
│  └─ evaluation/               # 评估问题集
├─ artifacts/
│  ├─ vector_db/                # 本地 ChromaDB，可重建且不提交 Git
│  └─ evaluations/              # RAGAS 和 baseline 评估结果
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

UI、本地模型、远程 Worker、PDF 解析和评估依赖通过 `--extra ui`、`--extra local-models`、`--extra worker`、`--extra docling`、`--extra mineru`、`--extra eval` 按需安装。例如，本地运行旧版 Streamlit RAG 界面时使用：

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
  --output data/chunks/教材_chunks.json
rag-qa ingest check data/chunks/教材_chunks.json
```

按标题结构切分 Markdown，并将 JSON 写入 `data/chunks/`。

### Step 4 — 向量化

```bash
rag-qa index build data/chunks/教材_chunks.json
rag-qa index list
```

`index build` 根据 `project/.env` 选择本地或远程 embedding Provider，将向量写入 `artifacts/vector_db/`；默认先完整构建临时集合，成功后再替换旧集合。已知教材会根据文件名推断稳定 ID，也可用 `--book database` 显式指定。原来的 `python project/vectorize_chunks.py` 仍保留为批量交互式兼容入口。

### Step 5 — 问答

```bash
rag-qa chat
```

启动交互式问答。输入 `test` 可运行内置测试用例，输入 `quit` 退出；仅检查检索流程时可使用 `rag-qa chat --no-llm --no-hyde`。`RAGEngine` 正式实现位于 `src/rag_textbook_qa/rag/`，原来的 `python project/rag_engine.py` 保留为兼容入口。

### Step 6 — 评估

```bash
python project/ragas_evaluation.py
```

评估问题来自 `data/evaluation/test_questions.json`，结果写入 `artifacts/evaluations/`。

如需同时运行无 RAG 基线对比，在脚本顶部将 `RUN_BASELINE = False` 改为 `True`（会额外消耗 token）。

### Step 7 — 启动 Web 界面

```bash
streamlit run project/app.py
```

在浏览器中打开问答界面，支持教材选择、top-k 调整、对话历史与 RAGAS 评估结果查看。

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
