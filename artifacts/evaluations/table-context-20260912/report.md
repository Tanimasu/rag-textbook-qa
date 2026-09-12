# 表格上下文离线验证

固定2000字符预算，每个长块单独作为最高排名结果，使用五教材审计目录中的49个长块。
本次不运行检索、embedding、reranker或LLM，不代表端到端召回率或答案质量。

- 完整纳入转换后文本：44块。
- 部分纳入完整行：1块。
- 结构不支持而跳过：4块。

所有输出均符合预算，context_text拼接与生成上下文一致。HTML属性被去除，合并单元格展开；完整纳入是本转换规则下的判定，不是对原始PDF表格语义的人工验收。

跳过样本：
- artifacts/chunks/five-book-audit-20260912/数据库原理及应用教程_mineru_chunks.json / ch15_s15_2_p938
- artifacts/chunks/five-book-audit-20260912/数据库原理及应用教程_mineru_chunks.json / ch15_s15_3_p973
- artifacts/chunks/five-book-audit-20260912/计算机组成原理_mineru_chunks.json / ch5_p690
- artifacts/chunks/five-book-audit-20260912/操作系统_mineru_chunks.json / ch12_s12_7_p095

上述表格保留原始分块，未自动修复；生成时继续尝试下一条证据，全无可用资料则不调用LLM。
原始HTML仍进入既有向量化链路，该阶段的长输入问题尚未解决。
