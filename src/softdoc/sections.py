"""Build semantic Sections from already-normalized heading elements."""

from __future__ import annotations

import re

from softdoc.hierarchy import HeadingAction
from softdoc.ids import section_id
from softdoc.models import Element, ElementType, Page, RelationSource, Section


class SectionBuilder:
    """Create a section tree without making heading-classification decisions."""

    def build(
        self,
        document_id: str,
        pages: list[Page],
        elements: list[Element],
    ) -> list[Section]:
        page_indexes = {page.page_id: page.page_index for page in pages}
        terminal_checklist_anchor = self._terminal_checklist_anchor(
            pages,
            elements,
            page_indexes,
        )
        ordered = sorted(
            elements,
            key=lambda item: (
                page_indexes[item.page_id],
                item.reading_order,
            ),
        )
        for element in ordered:
            element.section_id = None
            element.section_path = None

        elements_by_id = {
            element.element_id: element for element in elements
        }
        normalized_heading_titles = {
            self._normalized_title(element.text or "")
            for element in ordered
            if element.element_type == ElementType.HEADING
        }
        stack: list[Section] = []
        sections: list[Section] = []
        for element in ordered:
            hierarchy = element.metadata.get("heading_hierarchy", {})
            repeated_region = element.metadata.get("repeated_region")
            if (
                hierarchy.get("action") == HeadingAction.DOCUMENT_TITLE.value
                or repeated_region in {"page_header", "page_footer"}
                or element.metadata.get("excluded_from_section_hierarchy") is True
            ):
                continue

            if element.element_type == ElementType.HEADING:
                level = element.heading_level or 1
                while stack and stack[-1].level >= level:
                    stack.pop()
                path = [section.title for section in stack] + [element.text or ""]
                current = Section(
                    section_id=section_id(document_id, element.element_id),
                    document_id=document_id,
                    title=element.text or "",
                    level=level,
                    heading_element_id=element.element_id,
                    parent_section_id=stack[-1].section_id if stack else None,
                    section_path=path,
                    page_ids=[element.page_id],
                    element_ids=[element.element_id],
                    provenance=element.provenance,
                    metadata={
                        "created_by": RelationSource.DETERMINISTIC_RULE.value,
                        "heading_confidence": hierarchy.get("confidence"),
                        "heading_evidence": hierarchy.get("evidence", {}),
                    },
                )
                sections.append(current)
                stack.append(current)
                element.section_id = current.section_id
                element.section_path = path
                continue

            caption_section_title = self._functional_caption_section_title(
                element,
                elements_by_id,
                normalized_heading_titles,
            )
            if caption_section_title is not None:
                is_financial_statement = bool(
                    re.search(
                        r"\b(?:balance sheet|profit and loss(?: account)?|"
                        r"income statement|cash flow statement|"
                        r"statement of cash flows)\b",
                        caption_section_title,
                        re.IGNORECASE,
                    )
                )
                matching_ancestor = next(
                    (
                        section
                        for section in reversed(stack)
                        if self._normalized_title(section.title)
                        == self._normalized_title(
                            caption_section_title
                        )
                    ),
                    None,
                )
                floating_ancestor = next(
                    (
                        section
                        for section in reversed(stack)
                        if section.metadata.get("section_scope")
                        == "floating_table_block"
                    ),
                    None,
                )
                level = 1 if is_financial_statement else (
                    matching_ancestor.level
                    if matching_ancestor is not None
                    else (
                        floating_ancestor.level
                        if floating_ancestor is not None
                        else (stack[-1].level if stack else 1)
                    )
                )
                while stack and stack[-1].level >= level:
                    stack.pop()
                path = [
                    section.title for section in stack
                ] + [caption_section_title]
                current = Section(
                    section_id=section_id(
                        document_id,
                        element.element_id,
                    ),
                    document_id=document_id,
                    title=caption_section_title,
                    level=level,
                    heading_element_id=element.element_id,
                    parent_section_id=(
                        stack[-1].section_id if stack else None
                    ),
                    section_path=path,
                    page_ids=[element.page_id],
                    element_ids=[],
                    provenance=element.provenance,
                    metadata={
                        "created_by": (
                            RelationSource.DETERMINISTIC_RULE.value
                        ),
                        "section_scope": "floating_table_block",
                        "anchor_element_type": element.element_type.value,
                        "anchor_rule": (
                            "financial_statement_caption"
                            if is_financial_statement
                            else "short_all_caps_table_caption"
                        ),
                    },
                )
                sections.append(current)
                stack.append(current)
                target = elements_by_id.get(
                    str(element.metadata.get("target_element_id") or "")
                )
                if target is not None:
                    target.section_id = current.section_id
                    target.section_path = list(path)
                element.section_id = current.section_id
                element.section_path = list(path)
                element.metadata["section_anchor"] = {
                    "rule": "short_all_caps_table_caption",
                    "section_id": current.section_id,
                    "title": caption_section_title,
                }
                continue

            if not stack:
                continue
            current = stack[-1]
            element.section_id = current.section_id
            element.section_path = list(current.section_path)
            current.element_ids.append(element.element_id)
            if element.page_id not in current.page_ids:
                current.page_ids.append(element.page_id)
        if terminal_checklist_anchor is not None:
            self._build_terminal_checklist_section(
                document_id,
                terminal_checklist_anchor,
                ordered,
                page_indexes,
                sections,
            )
        self._synchronize_sections(sections, ordered)
        return sections

    @classmethod
    def _functional_caption_section_title(
        cls,
        element: Element,
        elements_by_id: dict[str, Element],
        normalized_heading_titles: set[str],
    ) -> str | None:
        if element.element_type != ElementType.CAPTION:
            return None
        target = elements_by_id.get(
            str(element.metadata.get("target_element_id") or "")
        )
        if target is None or target.element_type != ElementType.TABLE:
            return None
        text = re.sub(r"<[^>]+>", " ", element.text or "")
        text = " ".join(text.split())
        financial_statement = re.search(
            r"\b(?:balance sheet|profit and loss(?: account)?|"
            r"income statement|cash flow statement|"
            r"statement of cash flows)\b.*",
            text,
            re.IGNORECASE,
        )
        if financial_statement is not None:
            title = financial_statement.group(0).strip(" (:-\u2013\u2014")
            return title[:90] if title else None
        if not text or re.match(
            r"^(?:Table|Figure|Fig\.?|Listing|Source|Note|Notes|Exhibit)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        text = re.sub(r"(?<=[A-Z])(?=Test\s)", " ", text)
        stop = re.search(
            r"(?:\bTest\s*Conditions?|\bAll\s+parameter|"
            r"\bT\s*A\s*=|\bTA\s*=|\bNote\s*:|\bNotes\s*:)",
            text,
            re.IGNORECASE,
        )
        if stop is not None:
            text = text[: stop.start()]
        text = re.sub(r"\s*\(\s*TA\s*=.*$", "", text, flags=re.IGNORECASE)
        title = cls._deduplicate_prefix(
            text.strip(" (:-\u2013\u2014")
        )
        if not title or len(title) > 90:
            return None
        normalized = cls._normalized_title(title)
        if normalized in normalized_heading_titles:
            return None
        words = re.findall(r"[A-Za-z]+", title)
        if (
            not words
            or len(words) > 10
            or (len(words) == 1 and len(words[0]) < 6)
        ):
            return None
        letters = "".join(words)
        uppercase_ratio = (
            sum(character.isupper() for character in letters)
            / len(letters)
        )
        return title if uppercase_ratio >= 0.82 else None

    @staticmethod
    def _normalized_title(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    @staticmethod
    def _deduplicate_prefix(value: str) -> str:
        for prefix_length in range(8, len(value) // 2 + 1):
            prefix = value[:prefix_length]
            if value[prefix_length:].startswith(prefix):
                remainder = value[prefix_length * 2 :]
                return (prefix + " " + remainder.lstrip()).strip()
        return value

    @classmethod
    def _terminal_checklist_anchor(
        cls,
        pages: list[Page],
        elements: list[Element],
        page_indexes: dict[str, int],
    ) -> Element | None:
        """Find a parser-labelled page header that starts a terminal checklist.

        This deliberately requires both a title-like page-header anchor and
        several numbered checklist questions. A body sentence that merely
        mentions a checklist therefore cannot create a Section.
        """

        if not pages:
            return None
        last_page_index = max(page.page_index for page in pages)
        earliest_terminal_page = max(0, last_page_index - 1)
        candidates = [
            element
            for element in elements
            if page_indexes[element.page_id] >= earliest_terminal_page
            and element.bbox is not None
            and element.bbox.normalized[1] <= 0.15
            and 8 <= len((element.text or "").strip()) <= 100
            and re.search(
                r"\bcheck[\s-]?list\b",
                element.text or "",
                re.IGNORECASE,
            )
            and (
                element.metadata.get("mineru_type") == "page_header"
                or element.metadata.get("repeated_region") == "page_header"
            )
        ]
        for candidate in sorted(
            candidates,
            key=lambda item: (
                page_indexes[item.page_id],
                item.bbox.normalized[1] if item.bbox else 1.0,
            ),
        ):
            start_page = page_indexes[candidate.page_id]
            terminal_elements = [
                element
                for element in elements
                if page_indexes[element.page_id] >= start_page
                and element.element_id != candidate.element_id
            ]
            checklist_item_count = sum(
                cls._looks_like_numbered_checklist_item(
                    element.text or ""
                )
                for element in terminal_elements
            )
            has_later_heading = any(
                element.element_type == ElementType.HEADING
                and element.metadata.get("repeated_region")
                not in {"page_header", "page_footer"}
                and element.metadata.get(
                    "excluded_from_section_hierarchy"
                )
                is not True
                for element in terminal_elements
            )
            if checklist_item_count >= 3 and not has_later_heading:
                return candidate
        return None

    @staticmethod
    def _looks_like_numbered_checklist_item(value: str) -> bool:
        text = re.sub(r"<[^>]+>", " ", value)
        text = " ".join(text.split())
        return bool(
            re.search(
                r"(?:^|[\s\x03✓✗])(?:[A-Z]\d+|[A-Z])\.\s+",
                text,
            )
            and "?" in text
        )

    @classmethod
    def _build_terminal_checklist_section(
        cls,
        document_id: str,
        anchor: Element,
        ordered: list[Element],
        page_indexes: dict[str, int],
        sections: list[Section],
    ) -> None:
        title = " ".join((anchor.text or "").split())
        identifier = section_id(document_id, anchor.element_id)
        start_page = page_indexes[anchor.page_id]
        path = [title]
        section = Section(
            section_id=identifier,
            document_id=document_id,
            title=title,
            level=1,
            heading_element_id=anchor.element_id,
            parent_section_id=None,
            section_path=path,
            page_ids=[],
            element_ids=[],
            provenance=anchor.provenance,
            metadata={
                "created_by": RelationSource.DETERMINISTIC_RULE.value,
                "section_scope": "terminal_checklist",
                "anchor_element_type": anchor.element_type.value,
                "anchor_rule": (
                    "terminal_page_header_with_numbered_checklist_items"
                ),
            },
        )
        sections.append(section)
        for element in ordered:
            if page_indexes[element.page_id] < start_page:
                continue
            repeated_region = element.metadata.get("repeated_region")
            parser_type = element.metadata.get("mineru_type")
            if (
                element.element_id != anchor.element_id
                and repeated_region in {"page_header", "page_footer"}
            ):
                continue
            if (
                element.element_id != anchor.element_id
                and parser_type in {"page_header", "page_footer", "page_number"}
            ):
                continue
            element.section_id = identifier
            element.section_path = list(path)
        anchor.section_id = identifier
        anchor.section_path = list(path)
        anchor.metadata["section_anchor"] = {
            "rule": "terminal_page_header_with_numbered_checklist_items",
            "section_id": identifier,
            "title": title,
        }

    @staticmethod
    def _synchronize_sections(
        sections: list[Section],
        ordered: list[Element],
    ) -> None:
        by_id = {section.section_id: section for section in sections}
        for section in sections:
            section.element_ids = []
            section.page_ids = []
        for element in ordered:
            section = by_id.get(element.section_id or "")
            if section is None:
                continue
            section.element_ids.append(element.element_id)
            if element.page_id not in section.page_ids:
                section.page_ids.append(element.page_id)
