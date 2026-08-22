"""Parser-neutral element normalization before hierarchy and relation building."""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

from pydantic import Field

from softdoc.ids import bbox_id, stable_digest
from softdoc.models import (
    BoundingBox,
    Document,
    Element,
    ElementType,
    Provenance,
    RelationSource,
    SoftDocModel,
)
from softdoc.profiles import DocumentProfile


class ElementNormalizationDecision(SoftDocModel):
    element_id: str
    page_id: str
    page_number: int
    original_type: ElementType
    normalized_type: ElementType
    changed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rule: str
    created_by: RelationSource = RelationSource.DETERMINISTIC_RULE
    evidence: dict[str, object] = Field(default_factory=dict)


class ElementNormalizer:
    """Apply bounded, auditable corrections without discarding parser payload."""

    _EXPLICIT_ALGORITHM = re.compile(
        r"^\s*algorithm\s+\d+\b", re.IGNORECASE
    )
    _STRUCTURAL_SECTION = re.compile(
        r"\b(?:section|appendix)\s+[A-Z0-9]+(?:\.[0-9]+)*\s*:"
        r"\s*[^|]{0,180}",
        re.IGNORECASE,
    )
    _CAPTION_LABEL = re.compile(
        r"(?:^|[\s)])(?:Figure|Fig\.?|Table|Listing|Algorithm|图|表)\s*"
        r"\d+(?:\s*(?:\([a-z](?:\s*[-–—]\s*[a-z])?\)|[a-z](?:\s*[-–—]\s*[a-z])?))?"
        r"\s*[:.\-–—]",
        re.IGNORECASE,
    )
    _FOOTNOTE_PREFIX = re.compile(
        r"^\s*(?:<sup>\s*)?(?:\d+|[*†‡])(?:\s*</sup>)?(?=\s|[).:\-])"
        r"|^\s*(?:note|notes|source|sources)\s*[:：]",
        re.IGNORECASE,
    )
    _PROCEDURE_OR_CONTACT = re.compile(
        r"^\s*(?:step\s+\d+|it takes\b|to (?:use|install|connect|set)|"
        r"\d+\s+(?:on|click|choose|press|select|open|insert|connect)\b|"
        r"(?:tel|phone|email|contact)\b|www\.|https?://)",
        re.IGNORECASE,
    )
    _VISUAL_ATTRIBUTION = re.compile(
        r"\b(?:picture|photo|chart)\s+by\b|"
        r"source\s*:|\bfrom\s+(?:https?://|www\.)|"
        r"\bcopyright\b",
        re.IGNORECASE,
    )
    _OFFICIAL_OR_ADDRESS = re.compile(
        r"\b(?:governor|secretary|director|president|chief officer|"
        r"street|avenue|road|telephone|fax)\b",
        re.IGNORECASE,
    )
    _UI_TEXT = re.compile(
        r"\b(?:control panel|system preferences|enable|checkbox|"
        r"ask me before|sharing|click|button)\b",
        re.IGNORECASE,
    )
    _FINANCIAL_STATEMENT = re.compile(
        r"\b(?:balance sheet|profit and loss|income statement|"
        r"cash flow statement|statement of cash flows)\b",
        re.IGNORECASE,
    )
    _SLIDE_VISUAL_NOTE = re.compile(
        r"^\s*(?:source\s*:|notes?\s*:|"
        r"\d*\s*(?:EB|PB|TB|GB|MB|KB)\s*=)",
        re.IGNORECASE,
    )
    _NUMBERED_QUESTION = re.compile(
        r"^\s*(?P<marker>(?:\d{1,3}\s*[.)、]|[（(]\s*\d{1,3}\s*[)）]))\s*"
        r"(?P<body>\S.*)$"
    )
    _ALPHA_TABLE_NOTE = re.compile(
        r"^\s*(?P<marker>[a-d])\s*[.)]\s*(?P<body>\S.*)$",
        re.IGNORECASE,
    )
    _QUESTION_SIGNAL = re.compile(
        r"[?？]|\b(?:what|which|who|where|when|why|how|do|does|did|is|are|"
        r"can|could|should|would|will|have|has)\b|"
        r"(?:请|是否|什么|哪些|如何|为什么|有无|你|您|贵)",
        re.IGNORECASE,
    )
    _FORM_FIELD_SIGNAL = re.compile(
        r"(?:_{3,}|□|\[\s*\]|（\s*）|\(\s*\))"
    )
    _FORM_LABEL_SIGNAL = re.compile(
        r"\b(?:name|date|agency|bureau|identifier|investment|project|program|"
        r"rating|level|type|address|description|status|amount|cost|funding|"
        r"approval)\b[^:]{0,140}:",
        re.IGNORECASE,
    )
    _TABLE_COMPONENT_SIGNAL = re.compile(
        r"\b(?:table|field|column|row|component|parameter|attribute|"
        r"indicator|option|value|code|label)\b|"
        r"(?:表|字段|列|行|组件|参数|属性|指标|选项|取值|编码|标签)",
        re.IGNORECASE,
    )

    def normalize(
        self,
        document: Document,
        profile: DocumentProfile,
    ) -> list[ElementNormalizationDecision]:
        decisions: list[ElementNormalizationDecision] = []
        self._normalize_form_question_blocks(document)
        self._normalize_table_component_notes(document)
        elements = {
            element.element_id: element for element in document.elements
        }
        next_element: dict[str, Element | None] = {}
        by_page: dict[str, list[Element]] = defaultdict(list)
        for item in document.elements:
            by_page[item.page_id].append(item)
        for page_elements in by_page.values():
            ordered_page = sorted(
                page_elements, key=lambda item: item.reading_order
            )
            for index, item in enumerate(ordered_page):
                next_element[item.element_id] = (
                    ordered_page[index + 1]
                    if index + 1 < len(ordered_page)
                    else None
        )
        for element in document.elements:
            preset = element.metadata.get("_normalization_preset")
            original = (
                ElementType(str(preset["original_type"]))
                if isinstance(preset, dict)
                and preset.get("original_type") is not None
                else element.element_type
            )
            rule = "preserve_parser_type"
            confidence = 1.0
            evidence: dict[str, object] = {"profile": profile.value}
            if isinstance(preset, dict):
                rule = str(preset["rule"])
                confidence = float(preset["confidence"])
                evidence.update(dict(preset.get("evidence") or {}))
            text = self._plain_text(element.text or "")
            corrections: list[dict[str, str]] = []
            if profile == DocumentProfile.SLIDES and text:
                corrected_text, corrections = self._correct_slide_ocr(text)
                if corrections:
                    element.metadata.setdefault(
                        "normalization_original_text",
                        element.text,
                    )
                    element.metadata["ocr_corrections"] = corrections
                    element.text = corrected_text
                    text = corrected_text
            following = next_element.get(element.element_id)

            structural = self._STRUCTURAL_SECTION.search(text)
            slide_visual_note_target = (
                self._slide_visual_note_target(
                    element,
                    text,
                    by_page[element.page_id],
                )
                if profile == DocumentProfile.SLIDES
                else None
            )
            if (
                structural is not None
                and element.element_type
                in {ElementType.PARAGRAPH, ElementType.CAPTION}
                and (
                    element.element_type == ElementType.CAPTION
                    or structural.start() <= 20
                )
            ):
                original_text = element.text
                element.element_type = ElementType.HEADING
                element.text = structural.group(0).strip()
                element.heading_level = 1
                element.metadata["normalization_original_text"] = original_text
                element.metadata["embedded_structural_heading"] = True
                element.metadata.pop("target_element_id", None)
                rule = "embedded_section_label_promoted"
                confidence = 0.99
                evidence["matched_text"] = structural.group(0)

            elif slide_visual_note_target is not None:
                element.element_type = ElementType.FOOTNOTE
                element.metadata["target_element_id"] = (
                    slide_visual_note_target.element_id
                )
                element.metadata["slide_visual_note"] = True
                rule = "slide_visual_note_to_footnote"
                confidence = 0.97
                evidence["target_element_id"] = (
                    slide_visual_note_target.element_id
                )

            elif element.element_type in {
                ElementType.CODE,
                ElementType.ALGORITHM,
            }:
                normalized_type, code_rule, code_confidence, code_evidence = (
                    self._normalize_code_like(element, text)
                )
                element.element_type = normalized_type
                rule = code_rule
                confidence = code_confidence
                evidence.update(code_evidence)

            elif (
                element.element_type == ElementType.FOOTNOTE
                and not isinstance(preset, dict)
            ):
                target = elements.get(
                    str(element.metadata.get("target_element_id") or "")
                )
                (
                    normalized_type,
                    footnote_rule,
                    footnote_confidence,
                    footnote_evidence,
                ) = self._normalize_footnote(
                    element,
                    target,
                    text,
                    following=following,
                    profile=profile,
                )
                element.element_type = normalized_type
                rule = footnote_rule
                confidence = footnote_confidence
                evidence.update(footnote_evidence)

            elif element.element_type == ElementType.CAPTION:
                target = elements.get(
                    str(element.metadata.get("target_element_id") or "")
                )
                if (
                    target is not None
                    and self._is_visual(target)
                    and self._VISUAL_ATTRIBUTION.search(text)
                ):
                    element.element_type = ElementType.FOOTNOTE
                    element.metadata["visual_attribution"] = True
                    rule = "visual_attribution_to_footnote"
                    confidence = 0.97
                    evidence["target_element_id"] = target.element_id
                elif self._is_external_visual_title(element, target, text):
                    rule = "external_visual_title_preserved"
                    confidence = 0.97
                    evidence["target_element_id"] = target.element_id
                    evidence["vertical_gap"] = round(
                        self._vertical_gap(element, target), 4
                    )
                    evidence["horizontal_overlap"] = round(
                        self._horizontal_overlap(element, target), 4
                    )
                elif (
                    profile == DocumentProfile.SLIDES
                    and target is not None
                    and self._is_visual(target)
                    and not self._CAPTION_LABEL.search(text)
                    and not re.search(
                        r"\b(?:picture|photo|chart)\s+by\b|"
                        r"\bsource\s*:|\bfrom\s+\S+|©",
                        text,
                        re.IGNORECASE,
                    )
                    and (
                        self._overlap_fraction(element, target) >= 0.05
                        or len(text.split()) <= 12
                        or self._PHONE_OR_CONTACT.search(text)
                    )
                ):
                    element.element_type = ElementType.PARAGRAPH
                    element.metadata["visual_internal_text"] = True
                    element.metadata["excluded_from_relations"] = True
                    rule = "slide_local_text_not_caption"
                    confidence = 0.94
                    evidence["target_element_id"] = target.element_id
                    evidence["overlap_fraction"] = round(
                        self._overlap_fraction(element, target), 4
                    )
                elif (
                    target is not None
                    and self._is_visual(target)
                    and not self._CAPTION_LABEL.search(text)
                    and self._overlap_fraction(element, target) >= 0.45
                ):
                    element.element_type = ElementType.PARAGRAPH
                    element.metadata["visual_internal_text"] = True
                    element.metadata["excluded_from_relations"] = True
                    rule = "visual_internal_label_not_caption"
                    confidence = 0.95
                    evidence["target_element_id"] = target.element_id
                    evidence["overlap_fraction"] = round(
                        self._overlap_fraction(element, target), 4
                    )
                elif (
                    profile == DocumentProfile.SLIDES
                    and target is not None
                    and self._is_visual(target)
                    and not self._CAPTION_LABEL.search(text)
                    and len(text.split()) <= 8
                    and (
                        self._overlap_fraction(element, target) >= 0.15
                        or re.search(
                            r"\b(?:feature|axis|input|output|layer|node)\w*\b",
                            text,
                            re.IGNORECASE,
                        )
                    )
                ):
                    element.element_type = ElementType.PARAGRAPH
                    element.metadata["visual_internal_text"] = True
                    element.metadata["excluded_from_relations"] = True
                    rule = "slide_diagram_label_not_caption"
                    confidence = 0.93
                    evidence["target_element_id"] = target.element_id

            if (
                element.element_type == ElementType.LIST
                and self._is_empty_list(element)
            ):
                element.metadata["excluded_from_relations"] = True
                element.metadata["empty_list"] = True
                if rule == "preserve_parser_type":
                    rule = "empty_list_excluded_from_semantic_relations"
                    confidence = 1.0

            if isinstance(preset, dict):
                # These transformations are deliberately terminal: a form
                # prompt or table-field note should not be reinterpreted by a
                # generic caption/footnote heuristic later in this pass.
                rule = str(preset["rule"])
                confidence = float(preset["confidence"])
                evidence.update(dict(preset.get("evidence") or {}))

            if element.element_type != original:
                element.metadata.setdefault(
                    "parser_element_type", original.value
                )
            if corrections and rule == "preserve_parser_type":
                rule = "high_confidence_slide_ocr_correction"
                confidence = 0.99
                evidence["ocr_corrections"] = corrections
            decision = ElementNormalizationDecision(
                element_id=element.element_id,
                page_id=element.page_id,
                page_number=element.page_number,
                original_type=original,
                normalized_type=element.element_type,
                changed=element.element_type != original,
                confidence=confidence,
                rule=rule,
                evidence=evidence,
            )
            element.metadata["element_normalization"] = decision.model_dump(
                mode="json"
            )
            decisions.append(decision)

        if profile == DocumentProfile.SLIDES:
            decisions.extend(self._normalize_slide_primary_titles(document))
        decisions.extend(
            self._normalize_chart_titles(document, profile, elements)
        )
        decisions.extend(
            self._normalize_slide_chart_titles(document, profile, elements)
        )
        self._mark_duplicate_code_blocks(document)
        self._mark_visual_groups(document)
        return decisions

    def _normalize_form_question_blocks(self, document: Document) -> None:
        """Canonicalize parser-inconsistent questionnaire prompts.

        MinerU occasionally emits the same numbered form as a mixture of list
        and paragraph blocks.  We only touch contiguous, clearly interrogative
        numbered blocks, which leaves ordinary numbered lists intact.
        """

        by_page: dict[str, list[Element]] = defaultdict(list)
        for element in document.elements:
            by_page[element.page_id].append(element)

        replacements: dict[str, list[Element]] = {}
        affected_pages: set[str] = set()
        for page_id, page_elements in by_page.items():
            ordered = sorted(page_elements, key=lambda item: item.reading_order)
            candidates: list[tuple[Element, list[dict[str, object]]]] = []
            for element in ordered:
                units = self._numbered_question_units(element)
                if units:
                    candidates.append((element, units))
                else:
                    candidates.append((element, []))

            index = 0
            while index < len(candidates):
                if not candidates[index][1]:
                    index += 1
                    continue
                end = index
                unit_count = 0
                while end < len(candidates) and candidates[end][1]:
                    unit_count += len(candidates[end][1])
                    end += 1
                normalize_segment = unit_count >= 2 or any(
                    element.element_type == ElementType.LIST
                    for element, _ in candidates[index:end]
                )
                if normalize_segment:
                    for element, units in candidates[index:end]:
                        replacement = self._canonical_form_elements(
                            element,
                            units,
                            document,
                        )
                        if replacement is not None:
                            replacements[element.element_id] = replacement
                            affected_pages.add(page_id)
                index = end

        if not replacements:
            return
        rewritten: list[Element] = []
        for element in document.elements:
            rewritten.extend(replacements.get(element.element_id, [element]))
        document.elements[:] = rewritten
        self._reindex_pages(document, affected_pages)

    def _numbered_question_units(
        self,
        element: Element,
    ) -> list[dict[str, object]]:
        if element.element_type not in {ElementType.LIST, ElementType.PARAGRAPH}:
            return []
        units = self._list_item_units(element)
        if not units:
            return []
        for unit in units:
            match = self._NUMBERED_QUESTION.match(str(unit["text"]))
            if match is None or not self._is_question_prompt(match.group("body")):
                return []
            unit["marker"] = match.group("marker").strip()
            unit["body"] = match.group("body").strip()
        return units

    def _list_item_units(self, element: Element) -> list[dict[str, object]]:
        raw_content = element.provenance.raw_payload.get("content")
        raw_items = (
            raw_content.get("list_items")
            if isinstance(raw_content, dict)
            and isinstance(raw_content.get("list_items"), list)
            else []
        )
        grouped_items = element.metadata.get("grouped_items")
        layout_payload = element.provenance.metadata.get("layout_payload")
        layout_lines = (
            layout_payload.get("lines", [])
            if isinstance(layout_payload, dict)
            and isinstance(layout_payload.get("lines"), list)
            else []
        )
        line_units = self._split_numbered_lines(
            element.text or "",
            layout_lines,
        )
        if line_units:
            for index, unit in enumerate(line_units):
                if len(raw_items) == len(line_units):
                    unit["raw_item"] = raw_items[index]
                elif len(raw_items) == 1:
                    # MinerU can wrap several numbered prompts in one list
                    # item.  Keep that original parser item as lineage for
                    # each recovered atomic prompt.
                    unit["raw_item"] = raw_items[0]
            return line_units
        units: list[dict[str, object]] = []
        if raw_items:
            for index, item in enumerate(raw_items):
                if not isinstance(item, dict):
                    continue
                text = self._raw_text(item.get("item_content", item.get("text")))
                if text:
                    line = (
                        layout_lines[index]
                        if index < len(layout_lines)
                        and isinstance(layout_lines[index], dict)
                        else {}
                    )
                    units.append(
                        {
                            "text": text,
                            "raw_item": item,
                            "raw_bbox": item.get("bbox", line.get("bbox")),
                            "layout_line_index": index if line else None,
                        }
                    )
        elif isinstance(grouped_items, list) and grouped_items:
            for item in grouped_items:
                if not isinstance(item, dict):
                    continue
                text = self._plain_text(str(item.get("text") or ""))
                if text:
                    units.append(
                        {
                            "text": text,
                            "raw_item": item,
                            "raw_bbox": item.get("raw_bbox"),
                            "normalized_bbox": item.get("normalized_bbox"),
                        }
                    )
        else:
            text = self._plain_text(element.text or "")
            matches = list(
                re.finditer(
                    r"(?m)^\s*(?:\d{1,3}\s*[.)、]|[（(]\s*\d{1,3}\s*[)）])\s*\S.*$",
                    text,
                )
            )
            if matches:
                units = [{"text": match.group(0).strip()} for match in matches]
            elif text:
                units = [{"text": text}]
        return units

    @classmethod
    def _split_numbered_lines(
        cls,
        text: str,
        layout_lines: list[object],
    ) -> list[dict[str, object]]:
        """Split a wrapped MinerU list on numbered line starts.

        ``layout_payload.lines`` is authoritative geometry for ordinary
        ``list`` blocks (unlike ``index``, which has ``grouped_items``).  The
        contiguous line boxes belonging to each recovered prompt are unioned,
        preserving multi-line question geometry.
        """

        lines = text.splitlines()
        starts = [
            index
            for index, line in enumerate(lines)
            if cls._NUMBERED_QUESTION.match(line)
        ]
        if not starts:
            return []
        units: list[dict[str, object]] = []
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(lines)
            raw_boxes = [
                line.get("bbox")
                for line in layout_lines[start:end]
                if isinstance(line, dict)
                and isinstance(line.get("bbox"), (list, tuple))
                and len(line["bbox"]) == 4
            ]
            unit: dict[str, object] = {
                "text": " ".join(part.strip() for part in lines[start:end]).strip(),
                "layout_line_index": start if raw_boxes else None,
            }
            if raw_boxes:
                unit["raw_bbox"] = [
                    min(float(box[0]) for box in raw_boxes),
                    min(float(box[1]) for box in raw_boxes),
                    max(float(box[2]) for box in raw_boxes),
                    max(float(box[3]) for box in raw_boxes),
                ]
            units.append(unit)
        return units

    @classmethod
    def _raw_text(cls, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return cls._plain_text(value)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dict):
            return cls._raw_text(value.get("content", value.get("text")))
        if isinstance(value, list):
            return "".join(cls._raw_text(item) for item in value).strip()
        return cls._plain_text(str(value))

    def _canonical_form_elements(
        self,
        element: Element,
        units: list[dict[str, object]],
        document: Document,
    ) -> list[Element] | None:
        original_type = element.element_type
        if len(units) == 1:
            element.element_type = ElementType.PARAGRAPH
            element.text = str(units[0]["text"])
            self._mark_form_question(element, units[0], original_type)
            return None

        pages = {page.page_id: page for page in document.pages}
        page = pages.get(element.page_id)
        result: list[Element] = []
        for index, unit in enumerate(units):
            text = str(unit["text"])
            marker = str(unit["marker"])
            question_id = (
                f"{element.element_id}:form-question:{index:02d}:"
                f"{stable_digest(element.element_id, index, marker, text, length=10)}"
            )
            raw_item = unit.get("raw_item")
            raw_payload = {
                "derived_from": element.provenance.raw_payload,
                "list_item": raw_item,
            }
            question = Element(
                element_id=question_id,
                document_id=element.document_id,
                page_id=element.page_id,
                page_number=element.page_number,
                element_type=ElementType.PARAGRAPH,
                reading_order=element.reading_order * 1000 + index,
                bbox=self._unit_bbox(unit, element, page),
                column_index=element.column_index,
                text=text,
                parse_status=element.parse_status,
                provenance=Provenance(
                    provenance_id=(
                        f"prov:softdoc:{stable_digest(question_id, length=12)}"
                    ),
                    adapter=element.provenance.adapter,
                    source_path=element.provenance.source_path,
                    source_locator=(
                        f"{element.provenance.source_locator}:form-question[{index}]"
                    ),
                    parser_version=element.provenance.parser_version,
                    raw_payload=raw_payload,
                    metadata={"parent_provenance_id": element.provenance.provenance_id},
                ),
                metadata=dict(element.metadata),
            )
            question.metadata["derived_from_element_id"] = element.element_id
            question.metadata["derived_item_index"] = index
            self._mark_form_question(question, unit, original_type)
            result.append(question)
        return result

    def _mark_form_question(
        self,
        element: Element,
        unit: dict[str, object],
        original_type: ElementType,
    ) -> None:
        raw_item = unit.get("raw_item")
        nesting = None
        if isinstance(raw_item, dict):
            nesting = {
                key: raw_item[key]
                for key in ("level", "depth", "indent", "list_level")
                if key in raw_item
            }
        element.metadata.setdefault("parser_element_type", original_type.value)
        element.metadata["form_question"] = True
        element.metadata["form_question_marker"] = str(unit["marker"])
        element.metadata["form_question_nesting"] = nesting or {}
        if isinstance(raw_item, dict):
            element.metadata["raw_list_item"] = raw_item
        if unit.get("layout_line_index") is not None:
            element.metadata["layout_line_index"] = unit["layout_line_index"]
        element.metadata["_normalization_preset"] = {
            "original_type": original_type.value,
            "rule": "numbered_form_question_to_paragraph",
            "confidence": 0.98,
            "evidence": {
                "marker": str(unit["marker"]),
                "atomic_question": True,
            },
        }

    def _unit_bbox(
        self,
        unit: dict[str, object],
        parent: Element,
        page: object | None,
    ) -> BoundingBox | None:
        normalized = unit.get("normalized_bbox")
        raw = unit.get("raw_bbox")
        raw_item = unit.get("raw_item")
        if isinstance(raw_item, dict):
            raw = raw or raw_item.get("bbox")
        if isinstance(normalized, list) and len(normalized) == 4:
            normalized_values = tuple(float(value) for value in normalized)
            if page is not None:
                raw_values = tuple(
                    float(value) * dimension
                    for value, dimension in zip(
                        normalized_values,
                        (page.width, page.height, page.width, page.height),
                        strict=True,
                    )
                )
            else:
                raw_values = normalized_values
        elif isinstance(raw, (list, tuple)) and len(raw) == 4 and page is not None:
            raw_values = tuple(float(value) for value in raw)
            normalized_values = tuple(
                value / dimension
                for value, dimension in zip(
                    raw_values,
                    (page.width, page.height, page.width, page.height),
                    strict=True,
                )
            )
        else:
            return parent.bbox
        owner_id = stable_digest(parent.element_id, unit.get("text"), length=16)
        return BoundingBox(
            bbox_id=bbox_id(owner_id),
            raw=raw_values,
            normalized=normalized_values,
        )

    @classmethod
    def _is_question_prompt(cls, body: str) -> bool:
        return bool(
            cls._QUESTION_SIGNAL.search(body)
            or cls._FORM_FIELD_SIGNAL.search(body)
            or cls._FORM_LABEL_SIGNAL.search(body)
        )

    def _normalize_table_component_notes(self, document: Document) -> None:
        by_page: dict[str, list[Element]] = defaultdict(list)
        for element in document.elements:
            by_page[element.page_id].append(element)
        for page_elements in by_page.values():
            ordered = sorted(page_elements, key=lambda item: item.reading_order)
            for index, table in enumerate(ordered):
                if table.element_type != ElementType.TABLE:
                    continue
                notes: list[tuple[Element, str]] = []
                expected = ord("a")
                for candidate in ordered[index + 1 :]:
                    if candidate.element_type == ElementType.TABLE:
                        break
                    if candidate.element_type != ElementType.PARAGRAPH:
                        break
                    match = self._ALPHA_TABLE_NOTE.match(candidate.text or "")
                    if match is None or ord(match.group("marker").lower()) != expected:
                        break
                    if not self._references_table_components(match.group("body"), table):
                        break
                    notes.append((candidate, match.group("marker").lower()))
                    expected += 1
                    if expected > ord("d") + 1:
                        break
                if len(notes) < 2:
                    continue
                for note, marker in notes:
                    note.element_type = ElementType.FOOTNOTE
                    note.metadata.setdefault(
                        "parser_element_type", ElementType.PARAGRAPH.value
                    )
                    note.metadata["target_element_id"] = table.element_id
                    note.metadata["table_component_note"] = True
                    note.metadata["table_note_marker"] = marker
                    note.metadata["_normalization_preset"] = {
                        "original_type": ElementType.PARAGRAPH.value,
                        "rule": "table_component_paragraph_to_footnote",
                        "confidence": 0.97,
                        "evidence": {
                            "target_element_id": table.element_id,
                            "marker": marker,
                        },
                    }

    def _references_table_components(self, body: str, table: Element) -> bool:
        if self._TABLE_COMPONENT_SIGNAL.search(body):
            return True
        table_terms = {
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", table.text or "")
            if token.casefold() not in {"table", "this", "that", "with", "from"}
        }
        normalized_body = self._normalized_text(body)
        return any(term.casefold() in normalized_body for term in table_terms)

    def _is_empty_list(self, element: Element) -> bool:
        if self._plain_text(element.text or ""):
            return False
        raw_content = element.provenance.raw_payload.get("content")
        return not (
            isinstance(raw_content, dict)
            and any(
                self._raw_text(item.get("item_content", item.get("text")))
                for item in raw_content.get("list_items", [])
                if isinstance(item, dict)
            )
        )

    @staticmethod
    def _reindex_pages(document: Document, page_ids: set[str]) -> None:
        by_page: dict[str, list[Element]] = defaultdict(list)
        for element in document.elements:
            if element.page_id in page_ids:
                by_page[element.page_id].append(element)
        for page in document.pages:
            if page.page_id not in page_ids:
                continue
            existing_order = {
                element_id: index
                for index, element_id in enumerate(page.reading_order)
            }

            def order_key(element: Element) -> tuple[int, int, str]:
                parent_id = element.metadata.get("derived_from_element_id")
                anchor_id = (
                    str(parent_id)
                    if parent_id is not None
                    else element.element_id
                )
                return (
                    existing_order.get(anchor_id, len(existing_order)),
                    int(element.metadata.get("derived_item_index", 0)),
                    element.element_id,
                )

            ordered = sorted(by_page[page.page_id], key=order_key)
            for reading_order, element in enumerate(ordered):
                element.reading_order = reading_order
            element_ids = [element.element_id for element in ordered]
            object.__setattr__(page, "element_ids", element_ids)
            object.__setattr__(page, "reading_order", list(element_ids))

    def _slide_visual_note_target(
        self,
        element: Element,
        text: str,
        page_elements: list[Element],
    ) -> Element | None:
        if (
            element.element_type != ElementType.PARAGRAPH
            or element.bbox is None
            or not self._SLIDE_VISUAL_NOTE.match(text)
        ):
            return None
        candidates = [
            visual
            for visual in page_elements
            if self._is_visual(visual)
            and visual.bbox is not None
            and visual.bbox.normalized[3]
            <= element.bbox.normalized[1] + 0.02
            and self._vertical_gap(visual, element) <= 0.12
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda visual: (
                self._vertical_gap(visual, element),
                -self._horizontal_overlap(visual, element),
            ),
        )

    def _normalize_slide_primary_titles(
        self,
        document: Document,
    ) -> list[ElementNormalizationDecision]:
        """Unify a repeated slide-title style mislabeled as page headers."""

        candidate_groups: dict[tuple[int, int, int], list[Element]] = (
            defaultdict(list)
        )
        for element in document.elements:
            if (
                element.element_type != ElementType.PARAGRAPH
                or element.bbox is None
                or element.metadata.get("mineru_type")
                not in {"page_header", "header"}
                or not (element.text or "").strip()
            ):
                continue
            x1, y1, x2, y2 = element.bbox.normalized
            if (
                x1 <= 0.18
                and y1 <= 0.09
                and y2 <= 0.20
                and y2 - y1 >= 0.05
                and x2 - x1 >= 0.15
                and re.search(r"[A-Za-z\u4e00-\u9fff]", element.text or "")
            ):
                style_key = (
                    round(x1 / 0.04),
                    round(y1 / 0.03),
                    round((y2 - y1) / 0.02),
                )
                candidate_groups[style_key].append(element)
        # A single parser header may really be page furniture.  Three or more
        # matching title-band instances establish a document-level style.
        candidates = [
            element
            for group in candidate_groups.values()
            if len(group) >= 3
            for element in group
        ]
        if len(candidates) < 3:
            return []

        decisions: list[ElementNormalizationDecision] = []
        promoted_pages: set[str] = set()
        for element in candidates:
            original = element.element_type
            element.element_type = ElementType.HEADING
            element.heading_level = 1
            element.metadata["forced_heading_level"] = 1
            element.metadata["semantic_marginal_override"] = "slide_title"
            promoted_pages.add(element.page_id)
            decision = ElementNormalizationDecision(
                element_id=element.element_id,
                page_id=element.page_id,
                page_number=element.page_number,
                original_type=original,
                normalized_type=element.element_type,
                changed=True,
                confidence=0.98,
                rule="repeated_slide_title_style_promoted",
                evidence={
                    "matching_style_instances": len(candidates),
                    "mineru_type": element.metadata.get("mineru_type"),
                    "normalized_bbox": list(element.bbox.normalized),
                },
            )
            element.metadata["element_normalization"] = decision.model_dump(
                mode="json"
            )
            decisions.append(decision)

        candidate_ids = {element.element_id for element in candidates}
        by_page: dict[str, list[Element]] = defaultdict(list)
        for element in document.elements:
            if element.page_id in promoted_pages:
                by_page[element.page_id].append(element)
        for page in document.pages:
            if page.page_id not in promoted_pages:
                continue
            ordered = sorted(
                by_page[page.page_id],
                key=lambda element: (
                    0 if element.element_id in candidate_ids else 1,
                    (
                        element.bbox.normalized[1]
                        if element.bbox is not None
                        else 2.0
                    ),
                    (
                        element.bbox.normalized[0]
                        if element.bbox is not None
                        else 2.0
                    ),
                    element.element_id,
                ),
            )
            for reading_order, element in enumerate(ordered):
                element.reading_order = reading_order
            ordered_ids = [element.element_id for element in ordered]
            object.__setattr__(page, "element_ids", ordered_ids)
            object.__setattr__(
                page,
                "reading_order",
                list(ordered_ids),
            )
        return decisions

    @staticmethod
    def _correct_slide_ocr(
        text: str,
    ) -> tuple[str, list[dict[str, str]]]:
        corrected = text
        rules = (
            (r"(?<=\$)[Il](?=\d)", "1", "currency_leading_one"),
            (r"(?<=\d\.)[Il](?=[KMBT]\b)", "1", "decimal_unit_one"),
            (
                r"(?<![A-Za-z])(['’‘])[Il]{2}\b",
                r"\g<1>11",
                "two_digit_year",
            ),
            (
                r"(?<![A-Za-z])(['’‘])[Il](\d)\b",
                r"\g<1>1\g<2>",
                "mixed_digit_year",
            ),
            (
                r"\bin(?=\d{2}\b)",
                "in '",
                "missing_year_apostrophe",
            ),
            (r"(?<=#)[Il]\b", "1", "numbered_hash_label"),
            (
                r"\b[Il](?=(?:EB|PB|TB|GB|MB|KB|B)\b)",
                "1",
                "data_unit_one",
            ),
            (r"\b[Il](?=\s+Billion\b)", "1", "billion_count_one"),
        )
        corrections: list[dict[str, str]] = []
        for pattern, replacement, rule in rules:
            updated, count = re.subn(pattern, replacement, corrected)
            if count:
                corrections.append(
                    {
                        "rule": rule,
                        "before": corrected,
                        "after": updated,
                    }
                )
                corrected = updated
        return corrected, corrections

    def _normalize_chart_titles(
        self,
        document: Document,
        profile: DocumentProfile,
        elements: dict[str, Element],
    ) -> list[ElementNormalizationDecision]:
        if profile not in {DocumentProfile.REPORT, DocumentProfile.ACADEMIC}:
            return []
        by_page: dict[str, list[Element]] = defaultdict(list)
        for element in document.elements:
            by_page[element.page_id].append(element)
        decisions: list[ElementNormalizationDecision] = []
        for page_elements in by_page.values():
            ordered = sorted(page_elements, key=lambda item: item.reading_order)
            for heading in ordered:
                if (
                    heading.element_type != ElementType.HEADING
                    or heading.metadata.get("embedded_structural_heading")
                    or not heading.text
                    or self._STRUCTURAL_SECTION.search(heading.text)
                ):
                    continue
                following_visuals = [
                    item
                    for item in ordered
                    if item.reading_order > heading.reading_order
                    and item.reading_order - heading.reading_order <= 3
                    and item.element_type in {
                        ElementType.CHART,
                        ElementType.FIGURE,
                    }
                    and item.bbox is not None
                ]
                preceding_narrative = any(
                    item.element_type == ElementType.PARAGRAPH
                    and item.reading_order < heading.reading_order
                    and len((item.text or "").split()) >= 20
                    for item in ordered
                )
                if not following_visuals or not preceding_narrative:
                    continue
                target = min(
                    following_visuals,
                    key=lambda item: item.reading_order,
                )
                if (
                    heading.bbox is None
                    or target.bbox is None
                    or heading.bbox.normalized[3]
                    > target.bbox.normalized[1] + 0.02
                ):
                    continue
                original = heading.element_type
                heading.element_type = ElementType.CAPTION
                heading.heading_level = None
                heading.metadata["target_element_id"] = target.element_id
                heading.metadata["normalized_visual_title"] = True
                heading.metadata["excluded_from_heading_hierarchy"] = True
                heading.metadata["excluded_from_section_hierarchy"] = True
                decision = ElementNormalizationDecision(
                    element_id=heading.element_id,
                    page_id=heading.page_id,
                    page_number=heading.page_number,
                    original_type=original,
                    normalized_type=heading.element_type,
                    changed=True,
                    confidence=0.90,
                    rule="report_chart_title_to_caption",
                    evidence={
                        "profile": profile.value,
                        "target_element_id": target.element_id,
                        "preceding_narrative": True,
                    },
                )
                heading.metadata["element_normalization"] = (
                    decision.model_dump(mode="json")
                )
                decisions.append(decision)
                elements[heading.element_id] = heading
        return decisions

    def _normalize_slide_chart_titles(
        self,
        document: Document,
        profile: DocumentProfile,
        elements: dict[str, Element],
    ) -> list[ElementNormalizationDecision]:
        """Promote a strongly aligned paragraph above a slide chart."""

        if profile != DocumentProfile.SLIDES:
            return []
        by_page: dict[str, list[Element]] = defaultdict(list)
        for element in document.elements:
            by_page[element.page_id].append(element)
        decisions: list[ElementNormalizationDecision] = []
        for page_elements in by_page.values():
            charts = [
                element
                for element in page_elements
                if element.element_type == ElementType.CHART
                and element.bbox is not None
                and element.metadata.get("excluded_from_relations") is not True
            ]
            for chart in charts:
                if any(
                    element.element_type == ElementType.CAPTION
                    and element.metadata.get("target_element_id")
                    == chart.element_id
                    and element.metadata.get("excluded_from_relations") is not True
                    for element in page_elements
                ):
                    continue
                candidates: list[Element] = []
                for element in page_elements:
                    text = self._plain_text(element.text or "")
                    if (
                        element.element_type != ElementType.PARAGRAPH
                        or element.bbox is None
                        or not text
                        or element.metadata.get("repeated_region")
                        or element.metadata.get("excluded_from_relations") is True
                        or element.metadata.get("mineru_type")
                        in {"page_header", "page_footer", "header", "footer"}
                        or element.reading_order >= chart.reading_order
                        or chart.reading_order - element.reading_order > 2
                        or not 2 <= len(text.split()) <= 24
                        or self._VISUAL_ATTRIBUTION.search(text)
                        or re.search(
                            r"\b(?:In|Out)\s*\[\s*\d+\s*\]|"
                            r"\b(?:plt|imshow|plot)\s*\.",
                            text,
                            re.IGNORECASE,
                        )
                    ):
                        continue
                    source_left, _, source_right, source_bottom = (
                        element.bbox.normalized
                    )
                    _, target_top, _, _ = chart.bbox.normalized
                    width_ratio = (
                        source_right - source_left
                    ) / chart.bbox.width
                    if (
                        source_bottom <= target_top + 0.01
                        and self._vertical_gap(element, chart) <= 0.08
                        and self._horizontal_overlap(element, chart) >= 0.50
                        and width_ratio >= 0.30
                    ):
                        candidates.append(element)
                if not candidates:
                    continue
                title = min(
                    candidates,
                    key=lambda item: (
                        self._vertical_gap(item, chart),
                        chart.reading_order - item.reading_order,
                    ),
                )
                original = title.element_type
                title.element_type = ElementType.CAPTION
                title.metadata["target_element_id"] = chart.element_id
                title.metadata["normalized_visual_title"] = True
                decision = ElementNormalizationDecision(
                    element_id=title.element_id,
                    page_id=title.page_id,
                    page_number=title.page_number,
                    original_type=original,
                    normalized_type=title.element_type,
                    changed=True,
                    confidence=0.95,
                    rule="slide_paragraph_chart_title_to_caption",
                    evidence={
                        "profile": profile.value,
                        "target_element_id": chart.element_id,
                        "vertical_gap": round(
                            self._vertical_gap(title, chart), 4
                        ),
                        "horizontal_overlap": round(
                            self._horizontal_overlap(title, chart), 4
                        ),
                    },
                )
                title.metadata["element_normalization"] = (
                    decision.model_dump(mode="json")
                )
                decisions.append(decision)
                elements[title.element_id] = title
        return decisions

    def _normalize_code_like(
        self,
        element: Element,
        text: str,
    ) -> tuple[ElementType, str, float, dict[str, object]]:
        if self._EXPLICIT_ALGORITHM.match(text):
            return (
                ElementType.ALGORITHM,
                "explicit_numbered_algorithm",
                0.99,
                {},
            )
        if (
            element.element_type == ElementType.ALGORITHM
            and self._UI_TEXT.search(text)
            and not self._has_code_syntax(text)
        ):
            return (
                ElementType.FIGURE,
                "ui_screenshot_not_algorithm",
                0.95,
                {"ui_signal": True},
            )
        if re.search(
            r"(?:^|\n)\s*#\s*(?:answer|question|verify|predict)\b"
            r"|<input_[a-z_]+>",
            text,
            re.IGNORECASE,
        ):
            return (
                ElementType.CODE,
                "prompt_template_normalized_to_code",
                0.96,
                {"prompt_template_signal": True},
            )
        code_score = self._code_score(text, element)
        if code_score >= 2:
            return (
                ElementType.CODE,
                "runnable_or_prompt_code_normalized",
                min(0.98, 0.78 + code_score * 0.04),
                {"code_signal_score": code_score},
            )
        if (
            element.element_type == ElementType.ALGORITHM
            and len(text.split()) <= 8
            and not self._has_code_syntax(text)
        ):
            return (
                ElementType.PARAGRAPH,
                "short_output_not_algorithm",
                0.90,
                {"code_signal_score": code_score},
            )
        return (
            element.element_type,
            "preserve_code_like_type",
            0.70,
            {"code_signal_score": code_score},
        )

    def _normalize_footnote(
        self,
        element: Element,
        target: Element | None,
        text: str,
        *,
        following: Element | None = None,
        profile: DocumentProfile | None = None,
    ) -> tuple[ElementType, str, float, dict[str, object]]:
        evidence: dict[str, object] = {
            "target_element_id": target.element_id if target else None
        }
        if self._VISUAL_ATTRIBUTION.search(text):
            element.metadata["visual_attribution"] = True
            return (
                ElementType.FOOTNOTE,
                "visual_attribution_preserved",
                0.97,
                evidence,
            )
        if (
            self._PROCEDURE_OR_CONTACT.search(text)
            or self._OFFICIAL_OR_ADDRESS.search(text)
        ):
            return (
                ElementType.PARAGRAPH,
                "procedure_contact_or_official_text_not_footnote",
                0.94,
                evidence,
            )
        if (
            profile == DocumentProfile.SLIDES
            and not self._FOOTNOTE_PREFIX.search(text)
            and len(text.split()) <= 12
        ):
            element.metadata["excluded_from_relations"] = True
            return (
                ElementType.PARAGRAPH,
                "slide_local_text_not_footnote",
                0.92,
                evidence,
            )
        if self._FOOTNOTE_PREFIX.search(text):
            return (
                ElementType.FOOTNOTE,
                "explicit_footnote_or_source_prefix",
                0.98,
                evidence,
            )
        if (
            not self._FOOTNOTE_PREFIX.search(text)
            and following is not None
            and following.element_type == ElementType.TABLE
            and following.page_id == element.page_id
            and following.reading_order - element.reading_order <= 2
        ):
            evidence["following_table_id"] = following.element_id
            return (
                ElementType.PARAGRAPH,
                "table_introduction_before_following_table",
                0.95,
                evidence,
            )
        if target is None:
            element.metadata["footnote_anchor_status"] = "unresolved"
            return (
                ElementType.FOOTNOTE,
                "unbound_page_footnote_preserved",
                0.70,
                evidence,
            )
        overlap = self._overlap_fraction(element, target)
        evidence["overlap_fraction"] = round(overlap, 4)
        if self._is_visual(target) and overlap >= 0.30:
            element.metadata["visual_internal_text"] = True
            element.metadata["excluded_from_relations"] = True
            return (
                ElementType.PARAGRAPH,
                "visual_internal_text_not_footnote",
                0.96,
                evidence,
            )
        if target.element_type == ElementType.TABLE:
            # MinerU sometimes binds introductory prose above a table as a
            # footnote.  A markerless line below the table, however, is a
            # legitimate table note often enough that it must be preserved for
            # the relation validator to score instead of being discarded here.
            source_above_target = bool(
                element.bbox
                and target.bbox
                and element.bbox.normalized[3]
                <= target.bbox.normalized[1] + 0.01
            )
            evidence["source_above_target"] = source_above_target
            if source_above_target:
                return (
                    ElementType.PARAGRAPH,
                    "table_introduction_without_note_marker",
                    0.92,
                    evidence,
                )
            element.metadata["footnote_anchor_status"] = "needs_validation"
            return (
                ElementType.FOOTNOTE,
                "markerless_table_note_preserved_for_validation",
                0.68,
                evidence,
            )
        if self._is_visual(target):
            vertical_gap = self._vertical_gap(element, target)
            evidence["normalized_vertical_gap"] = round(vertical_gap, 4)
            if vertical_gap <= 0.10 and 2 <= len(text.split()) <= 100:
                return (
                    ElementType.CAPTION,
                    "nearby_visual_description_to_caption",
                    0.86,
                    evidence,
                )
        return (
            ElementType.FOOTNOTE,
            "ambiguous_footnote_preserved",
            0.55,
            evidence,
        )

    def _mark_duplicate_code_blocks(self, document: Document) -> None:
        code_like = [
            element
            for element in document.elements
            if element.element_type in {
                ElementType.CODE,
                ElementType.ALGORITHM,
            }
            and element.bbox is not None
            and element.text
        ]
        for index, source in enumerate(code_like):
            for target in code_like[index + 1 :]:
                if source.page_id != target.page_id:
                    continue
                if self._bbox_iou(source, target) < 0.45:
                    continue
                similarity = SequenceMatcher(
                    None,
                    self._normalized_text(source.text or ""),
                    self._normalized_text(target.text or ""),
                ).ratio()
                if similarity < 0.72:
                    continue
                duplicate = (
                    target
                    if target.reading_order > source.reading_order
                    else source
                )
                canonical = source if duplicate is target else target
                duplicate.metadata["duplicate_of"] = canonical.element_id
                duplicate.metadata["excluded_from_relations"] = True
                duplicate.metadata["duplicate_similarity"] = round(
                    similarity, 4
                )

    def _mark_visual_groups(self, document: Document) -> None:
        visuals_by_page: dict[str, list[Element]] = defaultdict(list)
        for element in document.elements:
            if self._is_visual(element) and element.bbox is not None:
                visuals_by_page[element.page_id].append(element)
        for caption in document.elements:
            if (
                caption.element_type != ElementType.CAPTION
                or not caption.text
                or not self._CAPTION_LABEL.search(caption.text)
                or caption.bbox is None
            ):
                continue
            candidates = [
                visual
                for visual in visuals_by_page[caption.page_id]
                if self._vertical_gap(caption, visual) <= 0.10
                and self._horizontal_overlap(caption, visual) >= 0.20
            ]
            if len(candidates) < 2:
                continue
            group_id = (
                f"visual-group:{stable_digest(caption.element_id, length=12)}"
            )
            caption.metadata["visual_group_id"] = group_id
            caption.metadata["visual_group_member_ids"] = [
                item.element_id for item in candidates
            ]
            for visual in candidates:
                visual.metadata["visual_group_id"] = group_id
                visual.metadata["visual_group_caption_id"] = (
                    caption.element_id
                )

    @staticmethod
    def _code_score(text: str, element: Element) -> int:
        score = 0
        patterns = (
            r"\b(?:def|class|import|from|return|lambda)\s+",
            r"\b(?:if|for|while)\s+.+:",
            r"\b\w+\s*=\s*[^=]",
            r"(?:^|\n)\s*(?:#|//)",
            r"\bIn\s*\[\s*\*?\d*\s*\]\s*:",
            r"\b(?:Question|Verify|Predict)\s*\(",
            r"<input_[a-z_]+>",
            r"\bThe answer is\s*:",
        )
        score += sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in patterns)
        if element.metadata.get("code_language"):
            score += 2
        return score

    @staticmethod
    def _has_code_syntax(text: str) -> bool:
        return bool(
            re.search(
                r"(?:[{}[\]();]|==|:=|=>|\bdef\s+|\bimport\s+|\w+\s*=\s*\S)",
                text,
            )
        )

    @staticmethod
    def _is_visual(element: Element) -> bool:
        return element.element_type in {
            ElementType.FIGURE,
            ElementType.CHART,
        }

    @classmethod
    def _is_external_visual_title(
        cls,
        source: Element,
        target: Element | None,
        text: str,
    ) -> bool:
        """Keep a compact title immediately above a visual as its caption.

        MinerU often emits chart titles as unnumbered captions.  On slides,
        the previous short-text rule could mistake these for labels embedded
        inside the image and suppress their functional relation.
        """

        if (
            target is None
            or not cls._is_visual(target)
            or source.bbox is None
            or target.bbox is None
            or not text
        ):
            return False
        _, _, _, source_bottom = source.bbox.normalized
        _, target_top, _, _ = target.bbox.normalized
        return (
            source_bottom <= target_top + 0.01
            and cls._vertical_gap(source, target) <= 0.08
            and cls._horizontal_overlap(source, target) >= 0.50
            and len(text.split()) <= 24
        )

    @staticmethod
    def _plain_text(text: str) -> str:
        return " ".join(re.sub(r"<[^>]+>", " ", text).split())

    @classmethod
    def _normalized_text(cls, text: str) -> str:
        return re.sub(r"\W+", " ", cls._plain_text(text).casefold()).strip()

    @staticmethod
    def _vertical_gap(source: Element, target: Element) -> float:
        if source.bbox is None or target.bbox is None:
            return 1.0
        _, sy1, _, sy2 = source.bbox.normalized
        _, ty1, _, ty2 = target.bbox.normalized
        if sy1 >= ty2:
            return sy1 - ty2
        if ty1 >= sy2:
            return ty1 - sy2
        return 0.0

    @staticmethod
    def _horizontal_overlap(source: Element, target: Element) -> float:
        if source.bbox is None or target.bbox is None:
            return 0.0
        sx1, _, sx2, _ = source.bbox.normalized
        tx1, _, tx2, _ = target.bbox.normalized
        overlap = max(0.0, min(sx2, tx2) - max(sx1, tx1))
        width = min(sx2 - sx1, tx2 - tx1)
        return overlap / width if width > 0 else 0.0

    @staticmethod
    def _overlap_fraction(source: Element, target: Element) -> float:
        if source.bbox is None or target.bbox is None:
            return 0.0
        sx1, sy1, sx2, sy2 = source.bbox.normalized
        tx1, ty1, tx2, ty2 = target.bbox.normalized
        overlap_x = max(0.0, min(sx2, tx2) - max(sx1, tx1))
        overlap_y = max(0.0, min(sy2, ty2) - max(sy1, ty1))
        source_area = (sx2 - sx1) * (sy2 - sy1)
        return overlap_x * overlap_y / source_area if source_area > 0 else 0.0

    @staticmethod
    def _bbox_iou(source: Element, target: Element) -> float:
        if source.bbox is None or target.bbox is None:
            return 0.0
        sx1, sy1, sx2, sy2 = source.bbox.normalized
        tx1, ty1, tx2, ty2 = target.bbox.normalized
        overlap_x = max(0.0, min(sx2, tx2) - max(sx1, tx1))
        overlap_y = max(0.0, min(sy2, ty2) - max(sy1, ty1))
        intersection = overlap_x * overlap_y
        union = (
            (sx2 - sx1) * (sy2 - sy1)
            + (tx2 - tx1) * (ty2 - ty1)
            - intersection
        )
        return intersection / union if union > 0 else 0.0

