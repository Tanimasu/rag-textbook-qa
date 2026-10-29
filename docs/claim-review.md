# 已存回答的逐主张引用复核

这个工具整理实际问题、提示词、回答、来源及引用标记，并校验复核记录的绑定关系。
它不调用模型、不自动拆分或赋予语义标签，也不认证专家身份、拆分完整性或独立验收通过。

## 导出未评分材料

当前支持 `context-selection-20261011` 等段落选择实验的 `generation-cases.json`、
`generation-requests.json` 和 `responses/gen-*.json` 格式。先在仓库环境中运行：

```bash
.venv/bin/python scripts/prepare_claim_review.py prepare \
  --batch-dir artifacts/evaluations/cache-scope-20261011 \
  --run-label '已存开发回答；不代表新候选质量' \
  --output-dir artifacts/evaluations/claim-review-新的目录
```

可重复传入 `--batch-dir` 合并批次。输出目录必须不存在；输入、旧输出均不修改。
程序校验冻结来源属于相应上下文块、请求散列及对应的样本/响应，重复采样即使请求散列相同也分别绑定。
原始 JSON 的重复字段或非有限数值会被拒绝。响应缺失、截断或未记录正常结束会保留其状态。

输出：

- `packet.json`：不可修改的实际材料及输入 SHA-256；保留完整回答和生成提示词。
- `review.md`：回答正文、原始引用偏移、实际上下文的并排阅读材料。
- `review-template.json`：所有样本初始为 pending，`claims` 为空。

`answer_body` 排除了回答生成的参考章节列表；所有偏移以该字段为准，不以排版后的 Markdown 为准。
代码中的示例引用保留但标为 `in_code`，不能借来支持正文主张。组合编号、编号范围和无效标记均保留原文。

## 填写复核副本

另存模板副本，填写 `reviewer`、`reviewer_kind`（model/human）、UTC 时间和 `review_status`。
只核对部分主张时使用 partial，样本使用 targeted；没有审完的样本继续 pending。
完整复核要求每份样本明确声明 `extraction_complete=true`；这是复核者声明，不是程序证明。

每个 `claims` 项包含：

|字段|含义|
|---|---|
|`id`|样本内唯一字符串编号|
|`start`、`end`|`answer_body` 中同一段落内的原句/子句精确字符跨度，左闭右开|
|`statement`|可独立理解的单一主张；保留条件、数值、符号及适用范围|
|`answer_qualifiers`|从该回答跨度逐字复制的条件/数值/符号片段数组，可为空；没有写出的条件应在理由中说明遗漏|
|`citation_marker_offsets`|该段实际引用标记的 `start` 偏移数组；无引用填空，不得从别段或章节列表借编号|
|`context_support`|实际全部上下文能否支持|
|`citation_support`|主张指定的原引用能否支持，和上一项分开填写|
|`context_evidence`、`citation_evidence`|`{"source_id": 2, "quote": "逐字原文"}` 数组；后者仅可使用原引用中的来源|
|`reason`|说明拆分、证据关系、缺失条件和引用范围歧义；非空|

两项标签均可为 supported、partial、unsupported、contradicted、insufficient_materials。
支持/部分支持必须附相应来源原文。引用来源不存在或没有引用时不能标为支持。
同段多个引用究竟支持哪一子句仍需阅读者判断，程序只验证来源与位置绑定。

例如“总金额不变且余额不为负”应拆成两个主张。不能因另一份来源支持非负余额，就把原回答错误的引用改成正确编号。
例子只给出非负十进制整数转八进制时，应记录这些条件；没有展开某项内容只记省略，不称为引用修复。

## 校验并保存结果

```bash
.venv/bin/python scripts/prepare_claim_review.py validate \
  --packet artifacts/evaluations/claim-review-新的目录/packet.json \
  --review artifacts/evaluations/claim-reviewed-copy.json \
  --output artifacts/evaluations/claim-validation-新的文件.json
```

保留原模板、packet 和复核副本。结果列出计划样本、完整生成、缺失/不完整生成、pending/targeted/complete 复核分母、
逐回答与逐主张标签；不输出自动放行结论，不把定向选择的主张比例解释为整份回答正确率。
散列用于一致性检查，不认证操作者身份；原 packet 必须可信保管。

这条离线流程与生成评判器分开。它没有改变 v7 评分语义或让 v7 自动具备回答引用完整性核证能力。
本轮实际使用结果见 [开发修复记录](context-development-20261029.md)。
