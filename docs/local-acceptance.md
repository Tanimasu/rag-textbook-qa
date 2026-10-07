# 本地验收与评分复核

所有命令从仓库根目录运行。输出路径必须未使用过；原教材、索引和评测集保持原样。

## 缓存模型 HTTP 验收

使用装有 `local-models`、`api` 和 `requests` 的环境，并确保两个 BGE 模型已缓存。
脚本强制离线加载，不读取 dotenv，也不调用远程 Worker 或 LLM。

```bash
python scripts/benchmark_serving.py --device mps --output artifacts/evaluations/local-http/mps.json
python scripts/benchmark_serving.py --device cpu --output artifacts/evaluations/local-http/cpu.json
```

每个设备单独运行，避免其他推理任务影响计时。包括 5 道 dev 题的顺序请求、容量 3 的八请求并发、
排队和活跃断连取消、后续恢复以及 RSS 采样。默认重复两轮，可用 `--repeats` 调整。
报告的 `complete=true` 才表示全部检查结束；异常后的部分结果继续保留。
原生推理调用会运行到返回，活跃取消不保证立即归还执行槽。
P95 仅为小样本描述；系统 `ps` 不可用时 RSS 为 null，不把它解释为零。

## 候选预算正文差异

已有 dev 预算报告时，可在不加载模型的情况下核对 50 / 75 的前五名：

```bash
python scripts/audit_candidate_budget.py \
  --report artifacts/evaluations/serving-performance-20261004/candidate-budget-dev.json \
  --dataset data/evaluation/retrieval_questions.json \
  --db-path artifacts/vector_db \
  --output artifacts/evaluations/candidate-budget/body-audit.json
```

脚本检查数据集和索引版本，读索引副本，保存退出 / 新增前五名的完整正文与哈希。
不赋相关性分数；只有数据集中已审核的正文证据跨度才计算逐字覆盖率，没有标注时保持 null。
候选退出前五名不等于整个答案失去该知识点。

## 导入跨教材评分副本

用 `scripts/prepare_crossbook_review.py` 导出的 `review.json` 和 `manifest.json` 作为原始参照，
保留在同一目录。审核者编辑 `review.json` 的副本：

```bash
cp artifacts/evaluations/serving-performance-20261004/crossbook-review/review.json \
  artifacts/evaluations/crossbook-reviewed.json
python scripts/score_crossbook_review.py \
  --template-dir artifacts/evaluations/serving-performance-20261004/crossbook-review \
  --review artifacts/evaluations/crossbook-reviewed.json --validate-only
```

可以调整问题与候选的排列顺序，但只修改 `grade`、`evidence_quotes`、`rationale` 和顶层审核状态 / 人员 / 时间。
分数必须是整数 0–3；大于零的分数需引用候选正文，所有已评分候选都需要理由。
完成后填写 `review_status="complete"`、非空 `reviewer` 和带 UTC 时区的 `reviewed_at_utc`。

```bash
python scripts/score_crossbook_review.py \
  --template-dir artifacts/evaluations/serving-performance-20261004/crossbook-review \
  --review artifacts/evaluations/crossbook-reviewed.json \
  --output artifacts/evaluations/crossbook-scores.json --top-k 5
```

空白或未完成评分会被拒绝，不产生质量报告。已有输出也会被拒绝。
`--top-k` 不能超过任一路线记录的排名长度。
指标只覆盖导出的 dev 候选并集，不能作为独立测试集或全库 recall。
原始模板与排名清单必须由操作者可信保存：哈希用于记录与核对输入，不提供数字签名或审核者身份认证。

本次实际验收结果与限制见 [2026-10-07 记录](optimization-20261007.md)。

## 真实远程检索与传输恢复

配置既有 Worker 地址、token 和模型后运行：

```bash
python scripts/benchmark_remote.py --output artifacts/evaluations/remote-acceptance/http-frozen.json
```

脚本读取环境变量和 `project/.env`，环境变量优先。只调用配置的 embedding / reranker Worker，
不调用 LLM，强制关闭本地回退。使用索引副本，完成 dev 小样本 REST / SSE 计时和冻结 15 题
的四策略检索验收；连接凭据不进入报告。冻结结果用于验收，不用于调参数。
首次模型加载可能较慢，Worker 应已备妥所需模型缓存；脚本本身不安装或下载本地模型。

单独复核传输中断与恢复，不需要远程设备或任何模型：

```bash
python scripts/check_remote_transport.py --output artifacts/evaluations/remote-acceptance/transport.json
```

这里的 HTTP Worker 是本机模拟器，验证截断响应的整批回退、错误分类和恢复。
真实 CUDA 与模拟故障的结果分别记录在 [远程验收记录](iteration-20261007-remote.md)。

## 生成流结束条件

使用实际 SDK 请求本机模拟的完成接口，不需要模型或 API 凭据：

```bash
python scripts/check_generation_transport.py --output artifacts/evaluations/generation-transport.json
```

验证正常 / 截断终止标记无需等待上游关闭连接，以及缺少结束标记的响应被判为不完整。
只测试传输和结束处理，不衡量生成质量。
