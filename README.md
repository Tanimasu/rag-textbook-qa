# 计算机教材 RAG 问答系统

> 面向计算机课程教材的检索增强生成（RAG）问答系统

基于混合检索策略（语义向量 + BM25）与大语言模型，实现对操作系统、计算机组成原理等教材内容的精准问答。系统采用 HyDE 查询增强与 Cross-Encoder 重排序，在 RAGAS 标准评估集上取得如下结果：

| 指标 | 得分 |
|------|------|
| Faithfulness | 0.9409 |
| Answer Relevancy | 0.9335 |
| Context Recall | 0.8456 |
| Context Precision | 0.7290 |

---

## 系统架构

```
PDF
 └─ parsingPDF.py       # Docling + EasyOCR → Markdown
     └─ clean_markdown.py    # 标题层级规范化 → *_cleaned.md
         └─ chunk_textbooks.py   # 按标题结构分块 → *_chunks.json
             └─ vectorize_chunks.py  # BAAI/bge-large-zh-v1.5 → ChromaDB
                 └─ rag_engine.py        # 混合检索 + HyDE + Reranker + LLM
```

**检索流程**

1. **HyDE**：用 LLM 将问题改写为假设性教材原文，用其嵌入向量检索，提升语义匹配质量
2. **混合检索**：向量相似度（权重 1.0）与 BM25 关键词匹配（权重 0.3）融合排序
3. **Cross-Encoder 重排序**：`BAAI/bge-reranker-base` 对候选结果精排，取最优 top-k
4. **LLM 生成**：将检索上下文与问题拼接为 Prompt，调用 LLM 生成结构化答案

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
LLM_MODEL=gemini-3.1-flash-lite-preview

# 可选：为 RAG 引擎和评估器单独指定模型（不设则使用上方共享值）
# RAG_MODEL=
# RAGAS_MODEL=
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
python parsingPDF.py
```

使用 Docling + EasyOCR 将 `data/` 目录下的 PDF 转换为 Markdown，输出至 `output/*.md`。
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

**选项 A：轻量评估**（无需外部服务）

```bash
python simple_evaluation.py
```

基于关键词重叠计算综合得分（Jaccard 相似度 × 0.4 + 关键词覆盖率 × 0.4 + 长度得分 × 0.2），结果保存为 `simple_eval_results.json` 和 `.xlsx`。

**选项 B：RAGAS 评估**（需要 LLM API）

```bash
python ragas_evaluation.py
```

使用 RAGAS 框架计算 Faithfulness、Answer Relevancy、Context Precision、Context Recall 四项指标，结果保存为 `ragas_evaluation_results.csv`。

> 如需使用难度更高的 `test_questions.json`（16 条），在脚本底部取消对应注释即可。

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
| `eval_dataset.json` | ~10 条 | 基础题，`simple_evaluation.py` 默认使用 |
| `test_questions.json` | 16 条 | 难度较高，覆盖操作系统与计算机组成原理 |
