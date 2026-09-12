"""Chinese tokenization for BM25 keyword retrieval.

BM25 exists in this system to catch exact textbook terms that vector search
misses. jieba's general-purpose dictionary works against that: it splits
散列表 into 散/列表 and 冯诺依曼 into 冯诺/依曼, so the terms the metric is
supposed to match never survive tokenization. The strategies here are all
attempts to keep those terms intact, and are chosen by measurement rather
than by argument.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import jieba

BM25_MODES = ("jieba", "dictionary", "bigram", "hybrid")
DEFAULT_BM25_MODE = "hybrid"

_WORD = re.compile(r"[一-鿿]+|[A-Za-z][A-Za-z0-9+#.]*|\d+")
_CJK_RUN = re.compile(r"^[一-鿿]+$")
_HEADING_NUMBER = re.compile(r"^[\s\d.、）)]*")
_CONNECTIVES = ("的", "与", "和", "及", "或", "在", "对")


def heading_terms(headings: Iterable[str], *, min_length: int = 2) -> set[str]:
    """Mine candidate terms from the heading paths already stored as metadata."""

    terms: set[str] = set()
    for heading in headings:
        text = _HEADING_NUMBER.sub("", str(heading or "")).strip()
        if not text:
            continue
        for fragment in re.split(r"[\s、，,：:（）()【】\[\]/]+", text):
            for piece in _split_on_connectives(fragment):
                if len(piece) < min_length or not _WORD.fullmatch(piece):
                    continue
                # A dictionary entry starting or ending with a connective
                # would make jieba mis-segment ordinary prose elsewhere.
                if piece.startswith(_CONNECTIVES) or piece.endswith(_CONNECTIVES):
                    continue
                terms.add(piece)
    return terms


def _split_on_connectives(fragment: str) -> list[str]:
    pieces = [fragment]
    for connective in _CONNECTIVES:
        nxt: list[str] = []
        for piece in pieces:
            nxt.extend(part for part in piece.split(connective) if part)
        pieces = nxt
    return [fragment, *pieces] if pieces != [fragment] else pieces


def install_terms(tokenizer: jieba.Tokenizer, terms: Iterable[str]) -> int:
    """Teach an isolated tokenizer the textbook vocabulary."""

    added = 0
    for term in terms:
        if term:
            tokenizer.add_word(term)
            added += 1
    return added


def _bigrams(text: str) -> list[str]:
    """Character bigrams for CJK runs, whole tokens for latin and digits.

    Bigrams need no dictionary, so a term jieba has never seen still matches
    as long as the query spells it the same way.
    """

    tokens: list[str] = []
    for match in _WORD.finditer(text):
        run = match.group()
        if not _CJK_RUN.match(run):
            tokens.append(run.lower())
        elif len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _words(text: str, tokenizer: jieba.Tokenizer) -> list[str]:
    return [token for token in tokenizer.cut(text) if _WORD.fullmatch(token)]


class BM25Tokenizer:
    """Tokenize documents and queries identically for one retrieval run."""

    def __init__(self, mode: str = DEFAULT_BM25_MODE, terms: Sequence[str] = ()) -> None:
        if mode not in BM25_MODES:
            raise ValueError(f"未知的 BM25 分词模式: {mode}")
        self.mode = mode
        self.terms = tuple(terms)
        self._tokenizer = jieba.Tokenizer()
        if mode in {"dictionary", "hybrid"} and terms:
            install_terms(self._tokenizer, terms)

    def __call__(self, text: str) -> list[str]:
        if self.mode == "bigram":
            return _bigrams(text)
        words = _words(text, self._tokenizer)
        if self.mode == "hybrid":
            return words + _bigrams(text)
        return words
