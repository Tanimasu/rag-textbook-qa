---
title: CS 教材问答
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: 基于五本计算机教材的检索增强问答，回答标注原文出处
---

# CS 教材问答

选一本计算机专业教材提问，服务先检索教材原文，再让大模型只依据检索到的片段作答，
每个结论都标注对应的教材片段，点击编号即可查看原文。

- 教材：操作系统、计算机组成原理、计算机网络、数据结构、数据库；
- 检索：BM25 与 `BAAI/bge-large-zh-v1.5` 混合召回，`BAAI/bge-reranker-base` 重排，均在本机 CPU 上运行；
- 接口：页面之外还有 REST API 和 `/docs` 接口文档。

回答由大模型生成，可能出错，请以原文为准。为控制费用，每个 IP 有提问频率限制，每天的生成次数也有
上限；额度用完后只返回检索到的教材原文。

源码、评测方法和已知局限见 GitHub：<https://github.com/Tanimasu/rag-textbook-qa>
