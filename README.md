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
Graduation_project/
├─ README.md
├─ CLAUDE.md
└─ project/
   ├─ app.py                    # Streamlit 入口
   ├─ rag_engine.py             # RAG 核心引擎：混合检索 / HyDE / 重排序 / 生成
   ├─ vectorize_chunks.py       # 文本分块向量化并写入 ChromaDB
   ├─ chunk_textbooks.py        # Markdown 分块
   ├─ clean_markdown.py         # Markdown 清洗与标题规范化
   ├─ parsingPDF.py             # Docling + EasyOCR 解析 PDF
   ├─ parsingPDF_mineru.py      # MinerU 解析 PDF（推荐）
   ├─ llm_client.py             # OpenAI-compatible LLM 客户端
   ├─ ragas_evaluation.py       # RAGAS 评估与 baseline 对比
   ├─ test_questions.json       # 评估问题集
   ├─ config/                   # 常量配置
   ├─ services/                 # 页面使用的数据加载与服务函数
   ├─ ui/                       # Streamlit 页面与样式模块
   ├─ data/                     # 原始教材 PDF
   ├─ output/                   # 解析、清洗、分块后的中间产物
   └─ vector_db/                # ChromaDB 持久化向量库
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install docling chromadb sentence-transformers rank-bm25 jieba \
            openai python-dotenv pandas openpyxl tqdm \
            ragas langchain-openai langchain-community datasets
```

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
cd project/
python check_env.py
```

验证 PyTorch、CUDA 与 GPU 是否可用。GPU 不可用时，后续 OCR 解析会显著变慢。

---

## 完整流程

所有脚本均在 `project/` 目录下运行。

### Step 1 — PDF 转 Markdown

```bash
python parsingPDF_mineru.py   # 推荐：MinerU，逐页自动判断是否 OCR，扫描页内容更完整
python parsingPDF.py          # 备选：Docling + EasyOCR
```

MinerU 版本输出 `output/*_mineru.md`，Docling 版本输出 `output/*.md`。运行前在脚本顶部修改 PDF 路径。
解析完成后可运行 `check_parsing_quality.py` 检查解析质量。

### Step 2 — 清洗 Markdown

```bash
python clean_markdown.py
```

通过 SmartMarkdownCleaner 规范化标题层级，输出 `output/*_cleaned.md`。

### Step 3 — 文本分块

```bash
python chunk_textbooks.py
```

按标题结构将清洗后的 Markdown 切分为 JSON 块（最大 800 字符），输出 `output/*_chunks.json`。
分块完成后可运行 `check_quality.py` 检查分块质量。

### Step 4 — 向量化

```bash
python vectorize_chunks.py
```

使用 `BAAI/bge-large-zh-v1.5` 生成嵌入向量并存入 ChromaDB（`vector_db/`）。交互式运行，每本教材单独询问是否处理。完成后可运行 `test_vector_db.py` 核查各集合的向量数量。

### Step 5 — 问答

```bash
python rag_engine.py
```

启动交互式问答。输入 `test` 可运行内置测试用例，输入 `quit` 退出。

### Step 6 — 评估

```bash
python ragas_evaluation.py
```

使用 RAGAS 框架计算 Faithfulness、Answer Relevancy、Context Precision、Context Recall 四项指标，结果保存为 `ragas_evaluation_results.csv`。评估数据集来自 `test_questions.json`（50 条，覆盖五本教材，每本 10 题）。

如需同时运行无 RAG 基线对比，在脚本顶部将 `RUN_BASELINE = False` 改为 `True`（会额外消耗 token）。

### Step 7 — 启动 Web 界面

```bash
streamlit run app.py
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
| `test_questions.json` | 50 条 | 覆盖五本教材，`ragas_evaluation.py` 默认使用 |
