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
- Python 源码编译检查
- source distribution 与 wheel 构建
- 发布包清单检查：wheel 与源码包中必需文件齐全，且不含本地数据、索引或密钥

本地可运行等价的核心检查：

```bash
python -m ruff check src tests scripts
python -m unittest discover -s tests -v
python -m compileall -q src tests project
python -m build
```

CI 不读取 `project/.env`，也不会连接远程 Worker、调用 LLM API 或下载模型权重。

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `get_models.py` | 查询当前 API 端点支持的模型列表 |
| `test_llm_api.py` | 验证 LLM API 连通性与模型响应 |
| `extract_images.py` | 从 PDF 中提取图片为 PNG 文件 |
| `clean_db.py` | 管理 ChromaDB 集合（列出 / 删除） |

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
