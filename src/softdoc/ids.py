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

