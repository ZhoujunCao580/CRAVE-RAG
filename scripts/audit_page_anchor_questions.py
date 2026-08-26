"""Audit page-reference language in the local MMLongBench-Doc questions.

This is a read-only diagnostic.  It does not change Exact Lookup semantics.
It distinguishes explicit printed-page-like references (``Page 12``) from
document-order references (``the second page``), then compares both with the
benchmark Gold evidence pages and any available SoftDoc printed-page labels.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "data" / "raw" / "mmlongbench_doc" / "questions.json"
DEFAULT_SOFTDOC_ROOT = ROOT / "data" / "processed" / "representative_28" / "softdoc"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "representative_28"
    / "reports"
    / "page_anchor_question_audit.json"
)


_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}
_CHINESE_NUMERALS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_EXAMPLE_CUE = re.compile(r"\b(?:for\s+example|e\.g\.)", re.IGNORECASE)
_FORMAT_CONTEXT = re.compile(r"\b(?:answer|output|format|formatted|list)\b", re.IGNORECASE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--softdoc-root", type=Path, default=DEFAULT_SOFTDOC_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    document_cache: dict[str, dict[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
    for question_index, source in enumerate(questions):
        text = str(source.get("question") or "")
        mentions = _extract_page_mentions(text)
        if not mentions:
            continue
        doc_id = str(source.get("doc_id") or "")
        stem = Path(doc_id).stem
        if stem not in document_cache:
            document_path = args.softdoc_root / stem / "document.json"
            document_cache[stem] = (
                json.loads(document_path.read_text(encoding="utf-8"))
                if document_path.is_file()
                else None
            )
        document = document_cache[stem]
        gold_pages = _evidence_pages(source.get("evidence_pages"))
        enriched = [
            _enrich_mention(mention, gold_pages=gold_pages, document=document)
            for mention in mentions
        ]
        rows.append(
            {
                "question_index": question_index,
                "doc_id": doc_id,
                "doc_type": source.get("doc_type"),
                "question": text,
                "answer": source.get("answer"),
                "gold_pages": gold_pages,
                "softdoc_available": document is not None,
                "mentions": enriched,
            }
        )

    mention_rows = [mention for row in rows for mention in row["mentions"]]
    kind_counts = Counter(item["kind"] for item in mention_rows)
    alignment_counts = Counter(item["gold_alignment"] for item in mention_rows)
    summary = {
        "total_questions": len(questions),
        "questions_with_page_language": len(rows),
        "total_page_mentions": len(mention_rows),
        "mention_kind_counts": dict(sorted(kind_counts.items())),
        "gold_alignment_counts": dict(sorted(alignment_counts.items())),
        "ordinal_question_count": sum(
            any(mention["semantic_class"] == "document_order" for mention in row["mentions"])
            for row in rows
        ),
        "explicit_numeric_question_count": sum(
            any(mention["kind"] in {"explicit_numeric", "chinese_numeric"} for mention in row["mentions"])
            for row in rows
        ),
        "representative_softdoc_question_count": sum(row["softdoc_available"] for row in rows),
    }
    payload = {
        "audit_version": "page-anchor-question-audit-v0.1",
        "summary": summary,
        "questions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(args.output)
    print(markdown_path)
    return 0


def _extract_page_mentions(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ignored_spans = _answer_example_spans(text)
    patterns: list[tuple[str, str, re.Pattern[str]]] = [
        (
            "explicit_numeric",
            "printed_label_or_physical_fallback",
            re.compile(r"(?<!\w)pages?\s+(?P<label>[0-9]+)(?![0-9])", re.IGNORECASE),
        ),
        (
            "numeric_ordinal",
            "document_order",
            re.compile(r"(?<!\w)(?P<label>[0-9]+)(?:st|nd|rd|th)\s+page\b", re.IGNORECASE),
        ),
        (
            "word_ordinal",
            "document_order",
            re.compile(
                r"\b(?:the\s+)?(?P<label>" + "|".join(_ORDINAL_WORDS) + r")\s+page\b",
                re.IGNORECASE,
            ),
        ),
        (
            "last_page",
            "document_order",
            re.compile(r"\b(?:the\s+)?(?P<label>last)\s+page\b", re.IGNORECASE),
        ),
        (
            "chinese_numeric",
            "printed_label_or_physical_fallback",
            re.compile(r"第\s*(?P<label>[0-9]+)\s*页"),
        ),
        (
            "chinese_ordinal",
            "document_order",
            re.compile(r"第\s*(?P<label>[一二三四五六七八九十])\s*页"),
        ),
    ]
    for order, (kind, semantic_class, pattern) in enumerate(patterns):
        for match in pattern.finditer(text):
            if any(start <= match.start() and match.end() <= end for start, end in ignored_spans):
                continue
            raw_label = match.group("label")
            if kind == "word_ordinal":
                number = _ORDINAL_WORDS[raw_label.casefold()]
            elif kind == "chinese_ordinal":
                number = _CHINESE_NUMERALS[raw_label]
            elif kind == "last_page":
                number = None
            else:
                number = int(raw_label)
            candidates.append(
                {
                    "text": match.group(0),
                    "kind": kind,
                    "semantic_class": semantic_class,
                    "number": number,
                    "span": [match.start(), match.end()],
                    "pattern_order": order,
                }
            )
    candidates.sort(key=lambda item: (item["span"][0], item["span"][1], item["pattern_order"]))
    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        start, end = candidate["span"]
        if any(start < old_end and old_start < end for old_start, old_end in (item["span"] for item in accepted)):
            continue
        candidate.pop("pattern_order")
        accepted.append(candidate)
    return accepted


def _answer_example_spans(text: str) -> list[tuple[int, int]]:
    """Mirror Exact Lookup's boundary for output-format examples."""

    spans: list[tuple[int, int]] = []
    for cue in _EXAMPLE_CUE.finditer(text):
        left_context = text[max(0, cue.start() - 140) : cue.start()]
        search_end = min(len(text), cue.end() + 240)
        bracket_start = text.find("[", cue.end(), search_end)
        if bracket_start >= 0:
            bracket_end = text.find("]", bracket_start + 1, search_end)
            if bracket_end >= 0:
                spans.append((cue.start(), bracket_end + 1))
                continue
        if not _FORMAT_CONTEXT.search(left_context):
            continue
        sentence_end = len(text)
        for terminator in ("\n", "?", "!"):
            position = text.find(terminator, cue.end())
            if position >= 0:
                sentence_end = min(sentence_end, position + 1)
        period = text.find(".", cue.end() + 4)
        if period >= 0:
            sentence_end = min(sentence_end, period + 1)
        spans.append((cue.start(), sentence_end))
    return spans


def _enrich_mention(
    mention: dict[str, Any],
    *,
    gold_pages: list[int],
    document: dict[str, Any] | None,
) -> dict[str, Any]:
    enriched = dict(mention)
    number = mention["number"]
    if number is None:
        physical_targets: list[int] = []
        printed_targets: list[int] = []
        gold_alignment = "not_numbered"
    else:
        physical_targets = [number]
        printed_targets = []
        if document is not None:
            printed_targets = sorted(
                {
                    int(page["page_number"])
                    for page in document.get("pages", [])
                    if str(number) in {str(value) for value in page.get("page_label_aliases", [])}
                }
            )
        physical_gold = number in gold_pages
        printed_gold = bool(set(printed_targets).intersection(gold_pages))
        if physical_gold and printed_gold:
            gold_alignment = "physical_and_printed"
        elif physical_gold:
            gold_alignment = "physical"
        elif printed_gold:
            gold_alignment = "printed"
        elif gold_pages:
            gold_alignment = "neither"
        else:
            gold_alignment = "no_gold_pages"
    enriched.update(
        {
            "physical_target_page_numbers": physical_targets,
            "printed_label_target_page_numbers": printed_targets,
            "gold_alignment": gold_alignment,
            "recommended_resolution": (
                "physical_document_order"
                if mention["semantic_class"] == "document_order"
                else "printed_label_then_physical_fallback"
            ),
        }
    )
    return enriched


def _evidence_pages(value: object) -> list[int]:
    if value is None:
        return []
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        return []
    return [int(item) for item in parsed]


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Page-reference language audit",
        "",
        "This report is diagnostic only; it does not modify Exact Lookup.",
        "",
        "## Summary",
        "",
        f"- Dataset questions: {summary['total_questions']}",
        f"- Questions containing audited page language: {summary['questions_with_page_language']}",
        f"- Page mentions: {summary['total_page_mentions']}",
        f"- Ordinal/document-order questions: {summary['ordinal_question_count']}",
        f"- Explicit numeric-page questions: {summary['explicit_numeric_question_count']}",
        f"- Questions with a local representative SoftDoc: {summary['representative_softdoc_question_count']}",
        "",
        "Mention kinds:",
        "",
    ]
    for key, value in summary["mention_kind_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Questions", ""])
    for row in payload["questions"]:
        lines.extend(
            [
                f"### Q{row['question_index']}",
                "",
                f"- Document: `{row['doc_id']}` ({row.get('doc_type')})",
                f"- Question: {row['question']}",
                f"- Gold pages: {row['gold_pages']}",
            ]
        )
        for mention in row["mentions"]:
            lines.append(
                "- Mention "
                f"`{mention['text']}`: kind=`{mention['kind']}`, "
                f"semantic=`{mention['semantic_class']}`, number={mention['number']}, "
                f"printed targets={mention['printed_label_target_page_numbers']}, "
                f"Gold alignment=`{mention['gold_alignment']}`, "
                f"recommended=`{mention['recommended_resolution']}`"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
