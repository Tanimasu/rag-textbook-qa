# 产品验收基线

这套验收回答的是“当前默认产品路径能否稳定地检索、回答并给出可核对依据”，不用于调参，
也不替代完整的 65 题检索评测或 50 题 RAGAS 评测。

阶段性运行结果单独记录在[产品验收结果](product-acceptance-results.md)，避免把协议与某一次机器环境下的
数字混在一起。

## 冻结题集

`data/evaluation/product_acceptance_v1.json` 固定 15 题，每本教材 3 题。它来自已经完成章节、
答案要点和教材行段核对的 `retrieval_holdout_candidates_v4.json`，并保留原始证据文件哈希。
当前 `review_status` 为 `human_reviewed_glance`：可以用于产品冒烟和回归检查，但不应描述为
经过独立专家审核的最终金标准。

同一份 JSON 同时包含：

- `relevant_sections`：用于确定性的检索评测；
- `ground_truth`：用于回答质量验收；
- `evidence`：教材文件、行段和 SHA-256，防止教材变化后继续沿用旧答案。

题集属于 holdout。看到结果后不得根据这 15 题调参数，再回头把同一结果称作无偏验收；需要调参时，
只能使用 `retrieval_questions.json` 的 dev 划分，改完后再运行一次冻结验收。

## 第一阶段：零 API 费用

先检查工作区、索引和数据集：

```bash
rag-qa doctor --json --index
python -m unittest tests.test_product_acceptance
```

Windows Worker 在线时，用固定的远程模型运行四种检索策略：

```bash
rag-qa worker check --json
rag-qa evaluate-retrieval \
  --questions data/evaluation/product_acceptance_v1.json \
  --split holdout \
  --strategy all \
  --top-k 5 \
  --context-budget 4000 \
  --output-dir artifacts/evaluations/product-acceptance-v1/retrieval
```

这一步不调用 LLM，不产生生成 API 费用。评测会禁止远程到本地的静默回退；Worker 不可用时应停止，
等同一计算后端恢复后再跑，避免把 Windows CUDA 与 Mac MPS 的结果混在一份基线里。

## 第二阶段：回答质量与产品交互

第二阶段会调用生成与评判 API，必须在运行前确认费用和模型配置：

```bash
rag-qa evaluate \
  --questions data/evaluation/product_acceptance_v1.json \
  --output-dir artifacts/evaluations/product-acceptance-v1/ragas \
  --dry-run

# 确认预检中的模型、后端、题数、费用影响和空输出目录后，再去掉 --dry-run 正式运行
rag-qa evaluate \
  --questions data/evaluation/product_acceptance_v1.json \
  --output-dir artifacts/evaluations/product-acceptance-v1/ragas
```

该命令默认复现公开产品路径：Top 5、HyDE 关闭、查询分解关闭、引用核对关闭。运行后必须检查
`ragas_run_summary.json` 的 `product_path` 与上述配置一致；需要试验 HyDE 或其他 Top K 时，使用
`--hyde` / `--top-k` 并写入另一输出目录，不得覆盖产品验收结果。

同时在产品页面逐题抽查以下项目：

1. 回答完整结束，没有把截断文本显示成成功；
2. 引用章节能打开，教材片段与问题相关；
3. 回答的关键陈述能在引用原文中找到依据；
4. 页面能显示检索、首字和总耗时；
5. “有帮助 / 需要改进”可以提交，问题分类与实际缺陷一致；
6. Windows Worker 关闭后，只抽 1–2 题确认 Mac MPS 回退，不把慢速结果混入 CUDA 基线。

完成后运行：

```bash
rag-qa feedback summary
rag-qa feedback candidates \
  --output artifacts/product/feedback-candidates.json \
  --force
```

下一轮只处理最高频、证据最充分的一类问题。实验性的 HyDE、查询分解和引用核对继续保持默认关闭，
直到独立评测证明收益大于额外延迟与成本。
