# 计算机教材 RAG 问答系统

面向计算机课程教材的检索增强生成（RAG）问答系统，完整流程分为 5 个阶段：
PDF 解析 → Markdown 清洗 → 文本分块 → 向量化 → 问答/评估

所有脚本均在 `project/` 目录下运行。

---

## 第一阶段：环境准备（一次性）

```bash
cd project/
python check_env.py
```

验证 PyTorch、CUDA、GPU 是否正常。如果不能用 GPU，后续 OCR 会很慢。

---

## 第二阶段：PDF 转 Markdown

```bash
python parsingPDF.py
```

- 使用 Docling + EasyOCR 将 PDF 转成 Markdown
- 每本教材跑一次，输出到 `output/*.md`
- 跑完后用 `check_parsing_quality.py` 检查解析质量（字符数、标题数、图片占位符等）

---

## 第三阶段：清洗 Markdown

```bash
python clean_markdown.py
```

- 规范化标题层级（SmartMarkdownCleaner）
- 输出 `output/*_cleaned.md`

---

## 第四阶段：文本分块

```bash
python chunk_textbooks.py
```

- 将 `_cleaned.md` 按标题结构切成 JSON chunks（每块最大 800 字符）
- 输出 `output/*_chunks.json` 和 `*_chunks_preview.txt`
- 跑完后用 `check_quality.py` 检查分块质量（截断代码块、行号残留、过大/过小等）

---

## 第五阶段：向量化

```bash
python vectorize_chunks.py
```

- 使用 `shibing624/text2vec-base-chinese` 生成向量，存入 ChromaDB
- 交互式，每本教材会单独询问是否向量化
- 跑完后用 `test_vector_db.py` 确认数据库中各集合的向量数量

---

## 第六阶段：问答测试

```bash
python rag_engine.py
```

- 启动交互式问答（输入 `test` 运行内置测试用例）
- 混合检索：Embedding（权重 1.0）+ BM25（权重 0.1）
- 检索结果送入 LLM 生成结构化答案

---

## 第七阶段：评估（二选一）

### 选项 A — 轻量评估（无需外部服务）

```bash
python simple_evaluation.py
```

读取 `eval_dataset.json`，用关键词重叠计算指标：

- 整体相似度（Jaccard）x 0.4
- 关键词覆盖率 x 0.4
- 长度合适度 x 0.2

结果保存为 `simple_eval_results.json` + `.xlsx`

### 选项 B — RAGAS 评估（需要 LLM API）

```bash
python ragas_evaluation.py
```

用 RAGAS 框架计算专业指标：

- Faithfulness（答案忠实度）
- Answer Relevancy（答案相关性）
- Context Precision（上下文精确率）
- Context Recall（上下文召回率，需 ground_truth）

结果保存为 `ragas_evaluation_results.csv`

> 如需使用 `test_questions.json`（16 条更难的题目），在文件底部取消对应注释即可。

---

## 工具脚本（按需使用）

| 脚本 | 用途 | 典型使用时机 |
|------|------|------------|
| `get_models.py` | 查询 API 支持的模型列表 | 换模型前确认名称 |
| `test_llm_api.py` | 验证 LLM API 连通性 | 配置 API 后首次使用 |
| `extract_images.py` | 从 PDF 提取图片为 PNG | 需要查看 PDF 中图片时 |
| `clean_db.py` | 删除 ChromaDB 中的集合 | 重新向量化前清理旧数据 |

---

## 评估数据集说明

| 文件 | 题数 | 格式 | 特点 |
|------|------|------|------|
| `eval_dataset.json` | ~10 条 | `{"questions": [...]}` | 基础题，`simple_evaluation.py` 默认使用 |
| `test_questions.json` | 16 条 | `[...]` | 更难更深，覆盖操作系统 + 计算机组成原理 |

---

## 依赖

```bash
pip install docling chromadb sentence-transformers rank-bm25 jieba openai pandas openpyxl tqdm ragas langchain-openai langchain-community datasets
```
