"""Deterministic lexical tokenization shared by SearchUnit and BM25."""

from __future__ import annotations

import re
from dataclasses import dataclass


_BASE_TOKEN = re.compile(
    r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]"
    r"|[^\W\d_]+",
    re.UNICODE,
)
_CJK = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]$")


@dataclass(frozen=True)
class TokenSpan:
    term: str
    start: int
    end: int


def chunk_token_spans(text: str) -> list[TokenSpan]:
    """Return non-overlapping tokens used only for deterministic slicing."""

    return [
        TokenSpan(match.group(0).casefold(), match.start(), match.end())
        for match in _BASE_TOKEN.finditer(text)
    ]


def bm25_token_spans(text: str) -> list[TokenSpan]:
    """Return lexical tokens plus contiguous CJK bigrams for BM25."""

    base = chunk_token_spans(text)
    result = list(base)
    for left, right in zip(base, base[1:]):
        if (
            _CJK.fullmatch(left.term)
            and _CJK.fullmatch(right.term)
            and left.end == right.start
        ):
            result.append(
                TokenSpan(
                    term=left.term + right.term,
                    start=left.start,
                    end=right.end,
                )
            )
    return sorted(result, key=lambda item: (item.start, item.end, item.term))
