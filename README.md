# 计算机教材 RAG 问答系统

[![CI](https://github.com/Tanimasu/rag-textbook-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanimasu/rag-textbook-qa/actions/workflows/ci.yml)

> 面向计算机课程教材的检索增强生成（RAG）问答系统

基于混合检索策略（语义向量 + BM25）与大语言模型，实现对操作系统、计算机组成原理等教材内容的精准问答。系统采用 HyDE 查询增强与 Cross-Encoder 重排序，在 50 题 RAGAS 评估集（5 本教材各 10 题）上取得如下结果：

> 生成模型：`gemini-3-flash-preview`（Google）；评判模型：`claude-sonnet-4-6`（Anthropic）—— 跨厂商配对以规避自我偏好偏差

| 指标 | 得分 | 评级 |
|------|------|------|
| Answer Relevancy | 0.8908 | 优秀 |
| Faithfulness | 0.7984 | 良好 |
| Context Recall | 0.7450 | 良好 |
| Context Precision | 0.6827 | 及格 |

检索层在 50 题人工标注集上的结果（确定性，不调用 LLM）：

| 策略 | Recall@5 | MRR |
|------|----------|-----|
| BM25 | 0.677 | 0.484 |
| Embedding | 0.800 | 0.639 |
| Hybrid | 0.800 | 0.660 |
| Hybrid + Reranker | 0.877 | 0.687 |

按 65 题全集统计。所有策略跑同一批问题，因此用配对检验判断差异是否可信：
只有「Hybrid + Reranker 优于纯 BM25」达到显著（分歧 15 题，1:14，p=0.0010）。
重排相对不重排仅 2:7（p=0.18），纯向量与融合为 2:2（p=1.00），均不足以支撑结论。
按当前比例外推，要让重排的优势显著需要约 108 题。表中数值差异小于此阈值时不应解读为提升。

评测报告里也记录了每题的检索耗时，但该数值受远程 Worker 负载影响波动较大，不宜作为系统指标解读。

---

## 项目亮点

- **面向教材问答的完整 RAG 流水线**：覆盖 PDF 解析、Markdown 清洗、分块、向量化、检索、生成与评估
- **混合检索策略**：融合语义向量检索与 BM25 关键词匹配，兼顾语义相关性与术语命中率
- **HyDE 查询增强**：先由 LLM 生成假设性教材原文，再进行向量检索，提升复杂问题的召回效果
- **Cross-Encoder 重排序**：使用 `BAAI/bge-reranker-base` 对候选片段精排，提升最终上下文质量
- **多教材独立向量库**：支持操作系统、计算机组成原理、计算机网络、数据结构、数据库原理及应用等多本教材
- **评估闭环完整**：集成 RAGAS 指标评估，并支持无 RAG baseline 对比
- **可视化交互界面**：基于 Streamlit 提供教材选择、参数调节、问答对话和评估结果查看
- **计算后端可观测**：每次回答显示 Embedding/Reranker 实际运行于远程 CUDA 还是本地 MPS/CPU，以及检索、生成和总耗时

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
2. **混合检索**：通过加权 RRF 按排名融合语义向量与 BM25，并过滤习题、去除重复候选。BM25 分词采用词 + 字符二元组混合，避免 jieba 把教材术语切碎；权重默认 0.3
3. **Cross-Encoder 重排序**：`BAAI/bge-reranker-base` 对候选结果精排，取最优 top-k
4. **LLM 生成**：将检索上下文与问题拼接为 Prompt，调用 LLM 生成结构化答案

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
```

`index build` 根据 `project/.env` 选择本地或远程 embedding Provider，将向量写入 `artifacts/vector_db/`；默认先完整构建临时集合，成功后再替换旧集合。已知教材会根据文件名推断稳定 ID，也可用 `--book database` 显式指定。原来的 `python project/vectorize_chunks.py` 仍保留为批量交互式兼容入口。

### Step 5 — 问答

```bash
rag-qa chat
```

启动交互式问答。输入 `test` 可运行内置测试用例，输入 `quit` 退出；仅检查检索流程时可使用 `rag-qa chat --no-llm --no-hyde`。`RAGEngine` 正式实现位于 `src/rag_textbook_qa/rag/`，原来的 `python project/rag_engine.py` 保留为兼容入口。

### Step 6 — 评估

```bash
rag-qa evaluate
```

评估问题来自 `data/evaluation/test_questions.json`，结果默认写入 `artifacts/evaluations/`。临时测试时可通过 `--questions` 指定小规模问题集，并用 `--output-dir` 将结果写入独立目录，避免覆盖已有正式结果。评估依赖按需安装：`uv sync --inexact --extra eval`。

RAGAS 的上下文现在使用实际发送给生成模型的文本，质量均分仅覆盖成功问题。
每次运行额外输出 `ragas_run_summary.json`，记录总题数、成功率及失败问题和原因类别；
即使全部问题失败，也会保留摘要。该口径与旧版评估不同，不宜直接比较历史分数。

评判模型通过 `RAGAS_MODEL` 单独指定，应与生成模型分属不同家族。若评判模型默认开启思考模式
（Qwen3 系列如此），需设置 `RAGAS_DISABLE_THINKING=true`，否则 faithfulness 会因超时全部变成
NaN；实测同一道判断题，开启思考耗时 7.2 秒且判错，关闭后 0.5 秒且判对。该参数只在端点支持时发送。

四项指标中只有 answer_relevancy 依赖向量模型：它先让评判模型从答案反推出一个问题，再比较
反推问题与原问题的向量相似度。该指标单次采样噪声很大，同一批答案重测两次平均绝对差 0.033、
最差一题 0.367。RAGAS 原本通过一次请求返回多个候选来平均，而 SiliconFlow 拒绝 `n>1`，
因此改为独立重复 `RAGAS_RELEVANCY_SAMPLES` 轮（默认 3）后取平均，设为 1 可恢复单次行为。

RAGAS 是验收指标，不是优化目标。一轮耗时一小时以上且消耗 token，分数又混合了检索、提示词与
生成模型三者，单项涨跌无法归因。检索改动应在确定性的检索评测上迭代，RAGAS 只在阶段末尾跑一次
确认方向。验收标准需在看到该轮数字之前约定，否则任何结果都能被事后解释成合理：

- **Faithfulness 优先。** 教材问答场景下，不编造内容比相关性更重要。这一条有具体理由：提示词
  要求模型在资料不足时明说，而 RAGAS 会把这种诚实拒答的 Answer Relevancy 直接判为 0，两项指标
  结构上对立。因此以相关性下降换取忠实度提升，判定为通过。
- **其余三项容忍 0.03 的回退。** 该数值为暂定。只有 Answer Relevancy 的噪声被实测过（同输入
  重测平均绝对差 0.033，且那是三轮平均生效之前的数据，现在应更低）；Faithfulness、
  Context Precision、Context Recall 均无实测噪声水平，需重测同一批答案后才能确认 0.03 是否合理。
- **绝对下限尚未约定。** 下次触发权衡判定时一并确定。

如需同时运行无 RAG 基线对比，使用 `rag-qa evaluate --baseline`（会额外消耗 token）。原来的 `python project/ragas_evaluation.py` 保留为兼容入口，并延续同时运行 baseline 的旧行为。

在运行会调用 LLM 的 RAGAS 评估前，可以先只比较检索链路：

```bash
rag-qa evaluate-retrieval --strategy all --top-k 5
rag-qa evaluate-retrieval --split holdout   # 仅在确认最终结论时使用
```

标注集现为 65 题，按教材分层切成 dev 35 题与 holdout 30 题（每本 13 题，其中 6 题为 holdout）。`--split` 默认取 dev，调参过程中不会读到
留出集；holdout 只用于最终验证，避免参数被同一批问题反复拟合。需要旧口径的整体数字时用
`--split all`。注意：当前的融合权重是在切分之前用全部 50 题扫出来的，因此这一版参数对
holdout 而言并不干净，今后停止使用这些题调参也不能消除已有的信息泄漏。
这 15 题只能作为历史验证子集，不能支持独立泛化结论。

新增 `data/evaluation/retrieval_holdout_candidates_v1.json` 包含五本教材各 3 题，
由 AI 编写并初步核对章节存在性，尚未调用检索或评分，也不会被默认评测加载。
它是待人工审核的候选题集，不能称为已验收的独立测试集。审核须确认问题可回答、
标注章节提供足够证据，并排除与旧题的语义重复；定稿后冻结内容和参数，再进行一次最终验证。
题集用途与出处见 `data/evaluation/retrieval_holdout_candidates_v1.md`。

后续教材依据审核保存在 `data/evaluation/retrieval_holdout_candidates_v2.json`，保留v1以追溯修改。
v2修正证据章节并替换3道题，每题附答案要点、源文件行号和SHA-256；审核记录见同名 `.md`。
状态为AI依据审核完成、待人工审定，仍不作为默认评测输入。旁附 `.sha256` 标识本次审核版本，
不是已经运行的评测成绩或正式冻结验收证明。

`retrieval_holdout_candidates_v3.json` 进一步替换 3 道题的标注章节。原标签是被分块器提升为
标题的正文列表项（如 `1.SCAN 调度算法`），不是教材的编号小节，已改用同主题规范小节并重新出题。
15 题的标注小节现均存在于索引分块、编号格式规范、证据行段有效、互不重复，与旧 50 题的措辞
重合度最高 0.18。状态不变，仍待人工审定。

`retrieval_holdout_candidates_v4.json` 改写 4 道题的措辞。检索评测集对问题有两条独立要求：
能由所标注小节回答，以及不照抄该小节的原文用语。前者决定标注是否成立，后者决定检索器之间的
对比是否公平——BM25 按字面匹配，问题复用原文词句会让它不经理解就命中。实测同一系统上，
模型据原文生成的题使 BM25 的 Recall@5 达到 0.975，人工措辞的题只有 0.680，而向量检索几乎不变。
改写后候选题的字面重合度中位数由 0.394 降至 0.345，接近人工 50 题的 0.319。详见同名 `.md`。

该命令使用 `data/evaluation/retrieval_questions.json`，依次比较 BM25、Embedding、Hybrid 和 Hybrid + Reranker，输出 Recall@K、Hit@K、MRR、nDCG@K 与平均检索耗时。

相关性按四级评分而非命中与否：命中标注小节记 3 分，命中同一父节下的兄弟小节记 2 分，同章记 1 分，
其余记 0 分。这样设计的原因是失败样本分析显示，检索常落在标注小节的相邻小节上（需要 3.5.3 却返回
3.5.2），二元指标把这种情况与召回到完全无关的章节同等对待。nDCG 的理想排序固定为标注小节排在最前、
其余位置填兄弟小节，使分数有上界 1；覆盖程度仍由 Recall@K 负责。

**重排环节不是完全可复现的。** BM25 与向量路径逐题结果稳定，但交叉编码器运行在 MPS 上，
候选分数接近时浮点微差会改变名次，实测可使 Recall@5 在 0.877 与 0.892 之间波动，幅度约一道题。
解读含重排的结论时须考虑这一抖动。Hybrid 使用 RRF（Reciprocal Rank Fusion）按名次融合两路结果，同时去除重复 chunk 和明确的习题候选，避免直接混合量纲不同的 BM25 与向量分数。评测会关闭 HyDE，不调用 LLM，也不会消耗 LLM API token；为保证结果可比，远程 Worker 不可用时会直接报错，不会静默回退到本地。JSON 报告默认写入 `artifacts/evaluations/retrieval/`。

分词和融合权重都是用这套评测调出来的。

BM25 原先直接用 jieba 默认词典，而它会把教材术语切碎：散列表切成「散 / 列表」，
冯诺依曼切成「冯诺 / 依曼」，十二个抽样术语里十一个被切开。BM25 在本系统里的职责
恰恰是精确命中术语，所以这等于让它带伤工作。现在改为词 + 字符二元组混合分词，并过滤
标点空白；二元组不依赖词典，词典没见过的术语只要查询写法一致就能命中。四种分词方案
在标注集上的对比结果是：纯 jieba 与仅加领域词典均为 Recall@5 0.600，纯二元组 0.700
但会拖低融合后的召回，词 + 二元组混合取得 0.700 且融合表现最好。

融合权重在 50 题标注集上重新扫描后定为 0.5。开启重排时（系统默认配置），Recall@5 在
权重 0.4 到 0.6 之间稳定为 0.900，两侧都会下滑，因此取该平台中部。BM25 现在承担了实际
作用：权重归零时重排后的 Recall@5 掉到 0.820，相当于四道题，而分词修好之前它只值一道题。

检索标注集现为 50 题（每本教材 10 题），复用 `test_questions.json` 里人工撰写的问题，
仅由模型补充「答案位于哪一节」的标签，并排除被误当作正文的目录条目。此前曾尝试由模型
依据小节原文直接出题，但那样生成的问题会沿用原文措辞，BM25 的 Recall@5 因此虚高到 0.975，
而人工问题上只有 0.680；检索评测集必须使用人工措辞的问题，否则关键词检索会被系统性高估。

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
| `data/evaluation/retrieval_questions.json` | 50 条 | 五本教材各 10 条章节标注题，按 dev 35 / holdout 15 分层切分，用于检索策略对比 |
| `data/evaluation/retrieval_holdout_candidates_v4.json` | 15 条 | AI 依据审核完成、待人工审定的候选验证题，默认评测不加载 |


### 表格回答上下文

完整 HTML 表格在生成前转换为紧凑行列文本，合并单元格按占用位置展开，
只纳入预算内的完整行；省略后续行时明确标记。含图片、标题或无法安全解释的结构暂时跳过，
继续尝试后续资料；如果没有可用证据，不调用生成模型。
界面引用与 RAGAS 使用的仍是实际送入模型的片段，原始检索结果保留不变。
这项改动只作用于回答上下文，不改变索引；长表格在 embedding 阶段的输入截断仍需另行验证。
