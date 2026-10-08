# 本地验收与评分复核

所有命令从仓库根目录运行。输出路径必须未使用过；原教材、索引和评测集保持原样。

RAGAS 正式运行也会拒绝非空输出目录；该检查在 CLI 加载引擎、评估器创建客户端和生成回答前执行。
请使用 `evaluate --output-dir artifacts/evaluations/新的实验名`。保存 JSON / CSV 时采用独占创建，
即使运行期间出现同名文件，也会报错并保留已有文件；失败目录不自动重用。
Streamlit 每次评估保存到 `artifacts/evaluations/ragas-runs/` 下独立的运行目录，并读取最近保存的
结果 CSV；仍兼容根目录的历史 CSV，未产生结果的空目录不会遮住已有结果。

`evaluate-generation` 的续跑协议也绑定实际题目、答案要点、资料和存档答案。
这些输入改变或旧协议缺少 `frozen_cases_sha256` 时，请使用新输出目录；不要给历史协议补写哈希。
已有结果仍保留，新的检查不会修改评判模板或重新判分历史答案。

## 已存回答的离线复核

先复用已保存的问答，不调用生成或评判 API：

```bash
python scripts/prepare_answer_review.py \
  --questions data/evaluation/product_acceptance_v1.json \
  --saved-answers artifacts/evaluations/product-acceptance-v1/ragas-20260918/ragas_qa_comparison.json \
  --output-dir artifacts/evaluations/answer-quality-20261008/product-review \
  --run-label '2026-09-18 历史回答，不代表当前版本'
```

输出 `review.json`、`review.md` 与 `manifest.json`。按问题正文精确对应已存回答，检查答案要点、
教材编号、可用的原题号、证据文件 SHA-256 和行段。重复问题、不同答案要点或陈旧证据会阻止导出；
缺失回答、未匹配回答、未记录上下文与结束状态会单独计数，所有质量判定保持空白。
现有输出目录不可覆盖，导出失败不发布半份复核包。

教材审核行段不是生成时的上下文，不能用它补认旧答案的引用编号。新 RAGAS 问答导出同时保存
实际上下文、资料编号与正文、提示词、原题号、生成模型与结束状态；评分输入 schema 保持不变。
保留导出的模板，另存复核副本。模型复核须标明来源，不能描述为独立人工金标准。
冻结验收题只用于验收和问题归类，不用于调整检索参数或评判规则。

## 已存评判的一致性复算

已有陈述拆分、逐条复核标签与判定时，可以离线重算一致性：

```bash
python scripts/compare_judge_review.py \
  --extracted artifacts/evaluations/generation-temperature-20260915/validation-v3/extracted.json \
  --labels artifacts/evaluations/generation-temperature-20260915/validation-v3/labels.json \
  --verified artifacts/evaluations/generation-temperature-20260915/validation-v3/verified.json \
  --output artifacts/evaluations/answer-quality-20261008/judge-agreement.json
```

工具核对全部题号、事实陈述编号与正文；缺失、重复、额外判定和非整数 0/1 标签均被拒绝。
输出混淆计数、Kappa、召回率、精确率、拆分遗漏和逐题分歧；无定义指标为 null。
未调用模型，不产生新标签，不自动改变原验收门槛。结果只表示与提供标签的一致性，
不认证审核者身份、独立性或盲标时序，也不能把历史 v3 结果称为当前 v4 验收。

## RAGAS 请求重试边界

安装 `eval` 依赖后，可验证真实 RAGAS / OpenAI SDK 的请求次数，全程使用内存 HTTP 传输：

```bash
python -m unittest tests.test_ragas_transport -v
```

429 和 503 最多首次请求加三次 SDK 重试，401 仅一次；三轮相关性评分各保留相同边界。
RAGAS 外层不再叠加五轮重试。解析修复与其他指标仍可能发起独立调用，这不是整批费用上限。
同一测试模块还并发运行两个评估器，验证三个引用型指标各自使用本次评测的服务与模型。
真实本机 HTTP 回归还验证连续五次评分复用连接、异步调用方兼容和连接关闭。
未安装 `eval` 时该模块会跳过；CI 构建任务在已安装的锁定 eval 依赖和 wheel 上实际执行它们。

`run_evaluation` 无论成功还是异常都会关闭评判连接。直接使用评估器时采用上下文管理器：

```python
with RAGASEvaluator(...) as evaluator:
    result = evaluator.evaluate(dataset)
```

同一实例的多次评分共用其专属线程和事件循环；退出时关闭 HTTP 池、事件循环和线程。

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

重复观察 HTTP 路径时，可跳过冻结题集：

```bash
python scripts/benchmark_remote.py --http-only --repeats 20 \
  --output artifacts/evaluations/remote-acceptance/http-continuous.json
```

每轮五道 dev 题、单教材 / 全库各一次；上述命令共 200 次计时请求，另检查 SSE。
报告采样客户端进程 RSS 和文件描述符（不可用时为 null），不测远程 GPU 内存，也不作为泄漏证明。

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
