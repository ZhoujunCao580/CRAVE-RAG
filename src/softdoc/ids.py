"""Deterministic, readable identifier helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def slug(value: str, *, fallback: str = "item") -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return normalized or fallback


def stable_digest(*parts: Any, length: int = 12) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def document_id(source_name: str, declared_id: str | None = None) -> str:
    identity = declared_id or source_name
    return f"doc:{slug(identity, fallback='document')}:{stable_digest(identity, length=8)}"


def page_id(doc_id: str, page_index: int) -> str:
    return f"{doc_id}:page:{page_index:04d}"


def element_id(doc_id: str, page_index: int, source_index: int, element_type: str, role: str = "main") -> str:
    return f"{doc_id}:page:{page_index:04d}:element:{source_index:04d}:{slug(element_type)}:{slug(role)}"


def section_id(doc_id: str, heading_element_id: str) -> str:
    return f"{doc_id}:section:{stable_digest(heading_element_id)}"


def provenance_id(adapter: str, source_path: str, source_locator: str) -> str:
    return f"prov:{slug(adapter)}:{stable_digest(source_path, source_locator)}"


def bbox_id(owner_id: str) -> str:
    return f"bbox:{stable_digest(owner_id)}"


def relation_id(
    relation_type: str,
    source_id: str,
    target_id: str,
    status: str,
    created_by: str,
) -> str:
    return (
        f"rel:{slug(relation_type)}:"
        f"{stable_digest(source_id, target_id, relation_type, status, created_by)}"
    )


def root_question_id(question_text: str, external_id: str | None = None) -> str:
    """Return a stable root-question ID without embedding the full question."""

    identity = external_id or question_text
    label = slug(external_id, fallback="root") if external_id else "root"
    return f"question:{label}:{stable_digest(identity, question_text, length=12)}"


def reading_session_id(question_id: str, run_key: str) -> str:
    """Namespace one repeatable reading run for a root question."""

    return f"reading:{stable_digest(question_id, run_key, length=16)}"


def action_id(reading_session_id: str, step_index: int) -> str:
    if step_index < 0:
        raise ValueError("Action step_index must be non-negative")
    namespace = stable_digest(reading_session_id, length=12)
    return f"action:{namespace}:{step_index:04d}"


def read_input_id(input_index: int) -> str:
    """Return the local alias used inside one read action (I1, I2, ...)."""

    if input_index < 0:
        raise ValueError("Read input_index must be non-negative")
    return f"I{input_index + 1}"


def observation_id(action_id: str, observation_index: int) -> str:
    """Return a stable global Observation ID owned by one action."""

    if observation_index < 0:
        raise ValueError("Observation index must be non-negative")
    return f"obs:{stable_digest(action_id, length=12)}:{observation_index:02d}"


def evidence_id(action_id: str, evidence_index: int) -> str:
    """Return a stable Evidence ID for an addition proposed by one action."""

    if evidence_index < 0:
        raise ValueError("Evidence index must be non-negative")
    namespace = stable_digest(action_id, length=12)
    return f"evidence:{namespace}:{evidence_index:04d}"
