# DeepSeek 生成、Kimi 辅助核查：调用前准备（2026-10-29）

用户已确定：**DeepSeek 生成答案，Kimi 辅助核查，核查结果暂不作为独立验收标准；排除硅基流动。** 本轮已完成离线请求准备和结构回归，实际模型调用为 0。选择服务/分工不等于授予费用额度，执行仍禁用。

## 固定的比较范围

从冻结候选 `focused_parent` 的保存排名回放中提取五道已查看开发题：文件目录 dev-06、Cache dev-08、输入设备 dev-12、栈 dev-22、第三范式 dev-30。

每题比较原始默认打包与新候选，各生成 3 次，共 **30 次 DeepSeek 生成**。两边均使用原产品提示词和同一模型/参数，只有实际上下文不同；不复用旧生成基线、不同时改提示词。使用问题×重复轮次作为区组，随机排列 15 个区组，区组内再随机排列两个方案，串行调用。三次生成仍是固定上下文的采样，不能称为三次独立上下文选择。

每份正常完成的 DeepSeek 回答最多触发 **1 次 Kimi 核查**，最多 30 次。因此计划最多 60 次模型 HTTP 尝试；无自动重试或付费 JSON 修复。缺失、截断、无效核查不记为通过。超时/用量未知、用量落盘失败、认证/服务身份错误或额度无法预留时停止后续调用。

这些题和 15 项答案要点均来自已经查看过的 Codex 开发诊断，不能称为独立新题或专家金标准。答案要点只用于覆盖检查，不发送给 DeepSeek，不允许 Kimi 将其当作实际上下文的事实依据。

## 服务与参数方案

|角色|拟用官方服务|模型与参数|
|---|---|---|
|生成|`https://api.deepseek.com/chat/completions`|`deepseek-v4-pro`，温度 0.7，`max_tokens=2000`，显式 `thinking.type=disabled`，非流式|
|辅助核查|`https://api.moonshot.ai/v1/chat/completions`，**账号站点仍待确认**|`kimi-k3.1-pro`，`reasoning_effort=low`，`max_completion_tokens=8192`，JSON 模式，缓存 implicit/5m，非流式；省略固定采样参数|

DeepSeek 文档标称该模型对应 DeepSeek-V4-Pro-0813，实际账户权限和响应版本尚未验证。Kimi K3.1 Pro 强制思考，温度固定，不能照搬此前硅基流动的 `enable_thinking=false` 或生成评判器的 `temperature=0` 请求。[DeepSeek 参数](https://api-docs.deepseek.com/api/create-chat-completion/)、[DeepSeek 模型信息](https://api-docs.deepseek.com/quick_start/pricing/)、[Kimi 参数](https://platform.kimi.ai/docs/api/models-overview)、[Kimi API](https://platform.kimi.ai/docs/api/chat)支持这些接口和参数约束。

两端分别使用本机 `DEEPSEEK_API_KEY`、`MOONSHOT_API_KEY`，不继承旧的 `LLM_*`/`RAGAS_*` 服务配置。当前未发现这两份独立密钥；未展示密钥值、修改本机 `.env` 或调用余额/模型发现接口。若账号使用国内 Kimi 站点，须先确认匹配的官方地址、模型和价格，再生成新协议，不把当前国际站方案悄悄换址执行。

产品的现有服务配置未切换；本报告只准备独立实验。工具本身没有 HTTP 发送功能，即使本机已有密钥也不会开始付费运行。

## 实际冻结材料与核查边界

证据目录 **E** = `artifacts/evaluations/deepseek-kimi-paired-20261029/`。

- `frozen-requests-final/generation-cases.json`：五题、答案要点、两方案实际完整上下文和来源。
- `generation-requests.json`：30 条实际 DeepSeek JSON 请求、顺序和逐条 SHA-256。
- `screening-spec.json`：Kimi 提示词、响应结构、模型配置与构造规则。新答案尚未产生，因此尚无 30 条含真实新答案的核查请求，不能预先补造。
- `protocol.json` / `plan.json`：输入、模型、随机种子、预算/停止规则和实现散列；`execution_enabled=false`、`budget_authorized=false`。
- `manifest.json`：六份计划文件的 SHA-256。旧候选源码与回放、教材要点原文的散列均在创建前核对。

Kimi 请求由一份身份匹配、`finish_reason=stop` 且非空的实际生成响应构造。请求绑定样本、生成请求哈希、新答案哈希、实际上下文哈希；核查来源保留生成模型看到的资料编号、标题和正文。回答正文排除生成的参考章节列表，原始回答另行保存。

整份核查提示词限 32768 个 UTF-8 字节，超出时记核查缺失，**不截断回答或来源**。8192 的输出额度包含思考和最终内容；仅保存思考字符数和用量，不保存原始思考文本。

核查结果必须正常结束、严格 JSON、主张原句在回答中可唯一定位、引用标记属于同段、摘录逐字属于声明资料、指定引用摘录不能从其他资料借用，且覆盖检查编号完整。结构无效不修复、不生成质量分。`validate-screening` 返回的“绑定有效”不表示语义判断正确；仍标记 `semantic_validity_verified=false`，没有自动放行结论。

模型拆分可能遗漏主张，条件保留和引用与子句的语义关系仍需阅读者复核。这是单次模型辅助筛查，不是已独立校准的评判器，也不升级/替代历史 v7 生成评分协议。

## 费用情景

已核对 [DeepSeek 官方价格](https://api-docs.deepseek.com/quick_start/pricing/)和 [Kimi 官方价格](https://platform.kimi.ai/docs/pricing/chat)，公开页面散列及 Kimi 动态价格表的数值提取保存在 `official-price-check.json`。

|模型|输入单价（美元/百万 token）|输出单价（美元/百万 token）|
|---|---:|---:|
|DeepSeek V4 Pro|高峰、未缓存 1.32|高峰 3.96|
|Kimi K3.1 Pro|未缓存 2.00；5m 缓存写入 2.00；缓存读取 0.20|10.00，含思考和回答|

Kimi 的缓存读取、缓存写入和未缓存部分划分同一个 `prompt_tokens` 总数，不能把输入和写入费用重复计到同一 token 上；本轮按三类输入中的最高单价估算。[官方计费字段说明](https://platform.kimi.ai/docs/api/chat)也明确输出用量包含思考和回答。

`cost-scenario.json` 采用偏保守的“每个 UTF-8 字节估 1 token、每请求另留 128 token 格式开销”情景，并按完整输出额度计算：

- 30 次 DeepSeek：约 **0.551525 美元**。
- 最多 30 次 Kimi：约 **4.431360 美元**。
- 合计约 **4.982885 美元**。

这是规划情景，不是实际 tokenizer 计数、已扣费用或可保证的账单硬上限；税费、支付手续费及汇率影响未计入，也没有折算成人民币。正式调用前仍需核实账号站点、当前单价和可执行的本轮总额控制。当前没有费用授权，历史预算不自动沿用。

## 离线复现与验证

```bash
.venv/bin/python scripts/prepare_paired_generation.py prepare \
  --replay-report artifacts/evaluations/context-development-20261029/replay-final/report.json \
  --candidate-freeze artifacts/evaluations/context-development-20261029/candidate-freeze.json \
  --evidence artifacts/evaluations/context-development-20261029/required-evidence-final.json \
  --output-dir artifacts/evaluations/deepseek-kimi-新的目录
```

生成后，可对已存的真实响应使用 `screen-request` 离线构造核查请求，或使用 `validate-screening` 校验已存核查。它们也不会发送 HTTP；`--help` 给出参数。已有输出会被拒绝，不覆盖旧协议。

最终 56 项相关回归、76 个子测试通过，0 跳过；见 `tests-complete.json` / `.log`。涵盖30次区组配对、提示词一致、来源/答案绑定、协议/要点变更隔离、错误引用保留、截断/重复 JSON/陈旧响应拒绝、超长输入不裁剪、缓存输入不重复计费、无网络导出和旧输出保留。Ruff、编译及差异检查另存 `final-validation.json`。

本轮没有生成成功率、核查成功率或新答案质量分数。实际服务可达性、账户模型权限、Kimi 账号站点、真实检索与 Worker 链路仍未验证。开始付费执行前需要本机密钥、账号站点和本轮费用上限；这三项不会用文档历史记录补认。
