"""Scope-aware numbered-label registry for explicit document references."""

from __future__ import annotations

import re
from dataclasses import dataclass

from softdoc.models import Document, Element
from softdoc.profiles import DocumentProfile


@dataclass(frozen=True)
class ReferenceTarget:
    target_id: str
    kind: str
    number: str
    resolution_rule: str
    label_source_id: str | None
    label_text: str
    priority: int
    page_index: int | None
    section_id: str | None
    section_root: str | None


class LabelRegistry:
    """Retain every label candidate and resolve it in source scope."""

    def __init__(self, document: Document) -> None:
        self.document = document
        self.profile = DocumentProfile(
            str(
                document.metadata.get("document_profile", {}).get(
                    "profile", DocumentProfile.REPORT.value
                )
            )
        )
        self.pages = {
            page.page_id: page.page_index for page in document.pages
        }
        self.elements = {
            element.element_id: element for element in document.elements
        }
        self.sections = {
            section.section_id: section for section in document.sections
        }
        self._targets: dict[tuple[str, str], list[ReferenceTarget]] = {}

    def add(
        self,
        *,
        kind: str,
        number: str,
        target_id: str,
        resolution_rule: str,
        label_source_id: str | None,
        label_text: str,
        priority: int,
    ) -> None:
        element = self.elements.get(target_id)
        section = self.sections.get(target_id)
        if element is not None:
            page_index = self.pages.get(element.page_id)
            section_id = element.section_id
            section_root = (
                element.section_path[0]
                if element.section_path
                else None
            )
        elif section is not None:
            heading = self.elements.get(section.heading_element_id)
            page_index = self.pages.get(heading.page_id) if heading else None
            section_id = section.section_id
            section_root = (
                section.section_path[0] if section.section_path else None
            )
        else:
            return
        candidate = ReferenceTarget(
            target_id=target_id,
            kind=kind,
            number=number,
            resolution_rule=resolution_rule,
            label_source_id=label_source_id,
            label_text=label_text,
            priority=priority,
            page_index=page_index,
            section_id=section_id,
            section_root=section_root,
        )
        key = (kind, number)
        current = self._targets.setdefault(key, [])
        if not any(
            item.target_id == candidate.target_id
            and item.label_source_id == candidate.label_source_id
            for item in current
        ):
            current.append(candidate)

    def resolve(
        self,
        source: Element,
        kind: str,
        number: str,
        *,
        matched_text: str,
    ) -> ReferenceTarget | None:
        candidates = self._targets.get((kind, number), [])
        if not candidates:
            return None
        source_page = self.pages.get(source.page_id)
        source_root = source.section_path[0] if source.section_path else None
        source_tokens = self._tokens(source.text or "")

        def rank(candidate: ReferenceTarget) -> tuple[int, int, int, int, str]:
            same_section = int(
                bool(
                    source.section_id
                    and candidate.section_id == source.section_id
                )
            )
            same_root = int(
                bool(source_root and candidate.section_root == source_root)
            )
            distance = (
                abs(source_page - candidate.page_index)
                if source_page is not None
                and candidate.page_index is not None
                else 1_000_000
            )
            return (
                -same_section,
                -same_root,
                -candidate.priority,
                distance,
                candidate.target_id,
            )

        selected = min(candidates, key=rank)
        if (
            self.profile
            not in {DocumentProfile.ACADEMIC, DocumentProfile.SLIDES}
            and source_page is not None
            and selected.page_index is not None
            and abs(source_page - selected.page_index) > 5
            and source_root
            and selected.section_root
            and source_root != selected.section_root
        ):
            label_tokens = self._tokens(selected.label_text)
            shared = source_tokens & label_tokens
            meaningful = {
                token
                for token in shared
                if token not in {"figure", "fig", "table", "section"}
                and len(token) > 2
            }
            if not meaningful:
                return None
        return selected

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+", text)
            if len(token) > 1
        }
