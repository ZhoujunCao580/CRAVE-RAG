from __future__ import annotations

from scripts.audit_page_anchor_questions import _enrich_mention, _extract_page_mentions


def test_page_language_audit_distinguishes_numeric_and_ordinal_semantics() -> None:
    numeric = _extract_page_mentions("Read Page 14.")
    ordinal = _extract_page_mentions("Read the second page and the 6th page.")

    assert [(item["kind"], item["number"]) for item in numeric] == [
        ("explicit_numeric", 14)
    ]
    assert [(item["kind"], item["number"]) for item in ordinal] == [
        ("word_ordinal", 2),
        ("numeric_ordinal", 6),
    ]
    assert all(item["semantic_class"] == "document_order" for item in ordinal)


def test_ordinal_page_recommends_physical_order_even_with_printed_alias() -> None:
    mention = _extract_page_mentions("Read the first page.")[0]
    document = {
        "pages": [
            {"page_number": 1, "page_label_aliases": []},
            {"page_number": 4, "page_label_aliases": ["1"]},
        ]
    }

    result = _enrich_mention(mention, gold_pages=[1], document=document)

    assert result["printed_label_target_page_numbers"] == [4]
    assert result["gold_alignment"] == "physical"
    assert result["recommended_resolution"] == "physical_document_order"


def test_numeric_page_audit_preserves_printed_label_preference() -> None:
    mention = _extract_page_mentions("Read Page 1.")[0]
    document = {
        "pages": [
            {"page_number": 1, "page_label_aliases": []},
            {"page_number": 4, "page_label_aliases": ["1"]},
        ]
    }

    result = _enrich_mention(mention, gold_pages=[4], document=document)

    assert result["gold_alignment"] == "printed"
    assert result["recommended_resolution"] == "printed_label_then_physical_fallback"


def test_answer_format_page_examples_are_not_counted_as_query_targets() -> None:
    mentions = _extract_page_mentions(
        'List the relevant pages; format the answer, for example, ["Page 17", "Page 25"].'
    )

    assert mentions == []
