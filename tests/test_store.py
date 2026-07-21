from __future__ import annotations

from softdoc.ids import relation_id
from softdoc.models import Relation, RelationEvidence, RelationSource, RelationStatus, RelationType
import pytest

from softdoc.store import DocumentStore, DocumentValidationError


def test_store_lookup_and_follow_relation(parsed_document) -> None:
    store = DocumentStore(parsed_document)
    relation = next(
        item for item in parsed_document.relations if item.relation_type == RelationType.CAPTION_OF
    )
    assert store.get_document() is parsed_document
    assert store.get_element(relation.source_id)
    targets = store.follow_relation(relation.source_id, RelationType.CAPTION_OF)
    assert [target.element_id for target in targets] == [relation.target_id]
    assert store.validate_references() == []


def test_invalid_relation_reference_is_reported(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    source = document.elements[0].element_id
    document.relations.append(
        Relation(
            relation_id=relation_id(
                RelationType.REFERS_TO.value,
                source,
                "missing:element",
                RelationStatus.CONFIRMED.value,
                RelationSource.EXPLICIT_REFERENCE.value,
            ),
            source_id=source,
            target_id="missing:element",
            relation_type=RelationType.REFERS_TO,
            confidence=1.0,
            status=RelationStatus.CONFIRMED,
            created_by=RelationSource.EXPLICIT_REFERENCE,
            evidence=[RelationEvidence(rule="test", description="invalid fixture")],
        )
    )
    errors = DocumentStore(document).validate_references()
    assert any("missing target" in error for error in errors)


def test_duplicate_ids_are_rejected(parsed_document) -> None:
    document = parsed_document.model_copy(deep=True)
    document.elements.append(document.elements[0].model_copy(deep=True))
    with pytest.raises(DocumentValidationError, match="Duplicate ID"):
        DocumentStore(document)
