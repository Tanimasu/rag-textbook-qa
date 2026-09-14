"""Pack retrieved evidence into exactly the context handed to generation."""

from __future__ import annotations

import re
from typing import Any

from rag_textbook_qa.rag.decomposition import context_budgets
from rag_textbook_qa.rag.tables import evidence_excerpt

SOURCE_SUFFIX = "\n---\n"
_HEADING_FIELDS = ("chapter", "section_h2", "section_h3", "section_h4")


def _source_prefix(result: dict[str, Any], index: int) -> str:
    """Render one citation header.

    Defined once on purpose: the fair-share pre-pass budgets against exactly the
    characters the packing loop later emits, so a format that drifted between the
    two would misallocate every slot without raising anything.
    """

    heading = " - ".join(
        str(result.get(field, "")) for field in _HEADING_FIELDS if result.get(field)
    )
    return f"【参考资料 {index}】\n教材: {result['book_name']}\n章节: {heading}\n内容:\n"


def select_context(
    results: list[dict[str, Any]],
    max_length: int = 2000,
    *, fair_share: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Pack evidence and return exactly the excerpts supplied to generation."""
    if max_length <= 0:
        raise ValueError("上下文预算必须大于 0")
    blocks = []
    sources = []
    remaining = max_length
    slots = None
    if fair_share:
        lengths = []
        for index, result in enumerate(results, 1):
            content = str(result.get("content", "")).strip()
            full, _, compacted = evidence_excerpt(
                content, max(len(content) * 100, max_length)
            )
            # Table rendering uses boundary newlines removed by excerpt.strip().
            estimated = len(_source_prefix(result, index)) + len(full)
            lengths.append(estimated + len(SOURCE_SUFFIX) + (2 if compacted else 0))
        slots = context_budgets(results, lengths, max_length)
    for result_index, result in enumerate(results):
        content = str(result.get("content", "")).strip()
        if not content:
            continue
        index = len(sources) + 1
        prefix = _source_prefix(result, index)
        slot = min(remaining, slots[result_index]) if slots is not None else remaining
        available = slot - len(prefix) - len(SOURCE_SUFFIX)
        if available <= 0:
            continue
        excerpt, truncated, table_compacted = evidence_excerpt(content, available)
        if fair_share and truncated and not table_compacted:
            notice = "\n[片段未完整装入，请勿推断省略内容]"
            end = max((m.end() for m in re.finditer(r"[。！？!?]|\n\n", excerpt)
                       if m.end() + len(notice) <= available), default=0)
            excerpt = excerpt[:end].rstrip() + notice if end else ""
        if not excerpt:
            continue
        block = prefix + excerpt + SOURCE_SUFFIX
        blocks.append(block)
        sources.append(
            {
                **result,
                "content": excerpt,
                "citation_id": index,
                "context_text": block,
                "truncated": truncated,
                "table_compacted": table_compacted,
                "char_count": len(excerpt),
            }
        )
        remaining -= len(block)
    return "".join(blocks), sources


def build_prompt(query: str, context: str) -> str:
    system_prompt = """你是一个计算机课程的专业 AI 助教，请严格依据教材内容回答问题。

要求：
1. 不要编造教材没有的内容
2. 先给出简明答案（2-3句话），再给出详细解释
3. 如有多个要点，使用编号列表
4. 在相关论述后标注【参考资料 N】，仅引用提供的资料编号，并在最后标注章节
5. 如果资料不足以回答，请明确说明教材证据不足，不要补造答案

回答格式示例：
## 简明答案
[2-3句话的核心答案]

## 详细解释
1. ...
2. ...

## 参考章节
📚 [章节信息]
"""
    return f"""{system_prompt}

## 学生问题
{query}

## 相关教材内容
{context}

请开始你的回答：
"""
