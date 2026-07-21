"""MinerU artifact adapter. MinerU-specific names are confined to this module."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from softdoc.ids import (
    bbox_id,
    document_id,
    element_id,
    page_id,
    provenance_id,
    section_id,
    stable_digest,
)
from softdoc.models import (
    BoundingBox,
    Document,
    Element,
    ElementType,
    Page,
    Provenance,
    Section,
)
from softdoc.relations import RelationBuilder
from softdoc.store import DocumentStore

logger = logging.getLogger(__name__)


_BLOCK_TYPE_MAP: dict[str, ElementType] = {
    "title": ElementType.HEADING,
    "heading": ElementType.HEADING,
    "paragraph": ElementType.PARAGRAPH,
    "text": ElementType.PARAGRAPH,
    "table": ElementType.TABLE,
    "image": ElementType.FIGURE,
    "figure": ElementType.FIGURE,
    "chart": ElementType.CHART,
    "caption": ElementType.CAPTION,
    "footnote": ElementType.FOOTNOTE,
    "page_footnote": ElementType.FOOTNOTE,
    "list": ElementType.LIST,
    "equation": ElementType.EQUATION,
    "equation_interline": ElementType.EQUATION,
}

_IGNORED_BLOCK_TYPES = {"page_header", "page_footer", "page_number", "page_aside_text"}
_KNOWN_BLOCK_KEYS = {
    "type",
    "bbox",
    "content",
    "page_idx",
    "page_index",
    "index",
    "reading_order",
    "column_index",
    "metadata",
    "text",
    "img_path",
    "image_path",
}
_KNOWN_CONTENT_KEYS = {
    "title_content",
    "paragraph_content",
    "list_type",
    "list_items",
    "image_source",
    "image_caption",
    "image_footnote",
    "chart_caption",
    "chart_footnote",
    "table_caption",
    "table_footnote",
    "caption_content",
    "footnote_content",
    "html",
    "table_type",
    "table_nest_level",
    "math_content",
    "math_type",
    "content",
    "text",
    "level",
    "style",
    "column_index",
    "column_count",
    "caption_bbox",
    "footnote_bbox",
    "image_path",
    "img_path",
    "target_element_id",
    "reference_label",
}


class MinerUAdapter:
    adapter_name = "mineru"

    def __init__(self) -> None:
        self.warnings: list[dict[str, Any]] = []

    def parse(self, input_path: Path, output_dir: Path) -> Document:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        self.warnings = []
        if not input_path.is_dir():
            raise NotADirectoryError(f"MinerU input must be a directory: {input_path}")
        layout_path = input_path / "layout.json"
        if not layout_path.exists():
            raise FileNotFoundError(f"Missing MinerU layout.json: {layout_path}")
        content_path = self._find_content_path(input_path)
        layout = self._load_json(layout_path)
        content_payload = self._load_json(content_path)
        layout_pages = self._layout_pages(layout)
        content_pages = self._content_pages(content_payload)

        declared_id = _first_string(layout, "document_id", "doc_id", "id")
        doc_id = document_id(input_path.name, declared_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "assets" / "pages").mkdir(parents=True, exist_ok=True)
        (output_dir / "assets" / "elements").mkdir(parents=True, exist_ok=True)

        page_indexes = sorted(set(layout_pages) | set(content_pages))
        pages: list[Page] = []
        elements: list[Element] = []
        for fallback_index, current_page_index in enumerate(page_indexes):
            page_payload = layout_pages.get(current_page_index, {})
            width, height = self._page_size(page_payload)
            page_number = int(page_payload.get("page_number") or current_page_index + 1)
            current_page_id = page_id(doc_id, current_page_index)
            page_image = self._copy_page_asset(
                input_path,
                output_dir,
                page_payload,
                current_page_index,
                current_page_id,
            )
            page_elements = self._parse_page_elements(
                input_path=input_path,
                output_dir=output_dir,
                doc_id=doc_id,
                current_page_id=current_page_id,
                page_index=current_page_index,
                page_number=page_number,
                page_width=width,
                page_height=height,
                blocks=content_pages.get(current_page_index, []),
                content_path=content_path,
            )
            elements.extend(page_elements)
            ordered_ids = [element.element_id for element in page_elements]
            page_provenance = self._provenance(
                source_path=layout_path.relative_to(input_path),
                source_locator=f"pdf_info[{fallback_index}]",
                raw_payload=page_payload,
            )
            pages.append(
                Page(
                    page_id=current_page_id,
                    document_id=doc_id,
                    page_index=current_page_index,
                    page_number=page_number,
                    width=width,
                    height=height,
                    element_ids=ordered_ids,
                    reading_order=ordered_ids,
                    image_path=page_image,
                    provenance=page_provenance,
                    metadata={"source_page_index": current_page_index},
                )
            )

        sections = self._build_sections(doc_id, pages, elements)
        doc_provenance = self._provenance(
            source_path=layout_path.relative_to(input_path),
            source_locator="document",
            raw_payload={
                "layout_metadata": {key: value for key, value in layout.items() if key not in {"pdf_info", "pages"}},
                "content_file": content_path.name,
            },
        )
        document = Document(
            document_id=doc_id,
            title=_first_string(layout, "title", "document_title"),
            source_path=Path(input_path.name),
            pages=pages,
            sections=sections,
            elements=elements,
            relations=[],
            provenance=doc_provenance,
            metadata={
                "adapter": self.adapter_name,
                "adapter_warnings": self.warnings,
                "summary_generation": "disabled",
                "keyword_generation": "disabled",
            },
        )
        RelationBuilder(document).build_all()
        DocumentStore(document).validate_references(raise_on_error=True)
        return document

    def _parse_page_elements(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        doc_id: str,
        current_page_id: str,
        page_index: int,
        page_number: int,
        page_width: float,
        page_height: float,
        blocks: list[dict[str, Any]],
        content_path: Path,
    ) -> list[Element]:
        elements: list[Element] = []
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                self._warn("invalid_block", f"Page {page_index} block {block_index} is not an object", block)
                continue
            self._record_unknown_fields(block, page_index, block_index)
            raw_type = str(block.get("type") or "").strip().lower()
            if raw_type in _IGNORED_BLOCK_TYPES:
                self._warn("ignored_block_type", f"Ignored MinerU auxiliary type: {raw_type}", block)
                continue
            element_type = _BLOCK_TYPE_MAP.get(raw_type)
            if element_type is None:
                self._warn("unsupported_block_type", f"Unsupported MinerU block type: {raw_type or '<missing>'}", block)
                continue
            source_index = int(block.get("index", block_index))
            main_id = element_id(doc_id, page_index, source_index, element_type.value)
            content = block.get("content") if isinstance(block.get("content"), dict) else {}
            text, html = _element_content(element_type, block, content)
            image_path = self._copy_element_asset(input_path, output_dir, block, content, main_id)
            bbox = self._bbox(
                owner_id=main_id,
                raw_bbox=block.get("bbox"),
                page_width=page_width,
                page_height=page_height,
                context=f"page={page_index} block={block_index}",
            )
            provenance = self._provenance(
                source_path=content_path.relative_to(input_path),
                source_locator=f"page[{page_index}].block[{block_index}]",
                raw_payload=block,
            )
            metadata = dict(block.get("metadata") or {})
            if isinstance(content.get("style"), dict):
                metadata["style"] = dict(content["style"])
            if content.get("column_count") is not None:
                metadata["column_count"] = content.get("column_count")
            main = Element(
                element_id=main_id,
                document_id=doc_id,
                page_id=current_page_id,
                page_number=page_number,
                element_type=element_type,
                reading_order=len(elements),
                bbox=bbox,
                column_index=_optional_int(block.get("column_index", content.get("column_index"))),
                heading_level=_heading_level(block, content) if element_type == ElementType.HEADING else None,
                text=text,
                html=html,
                image_path=image_path,
                reference_label=_optional_text(content.get("reference_label")),
                summary=None,
                keywords=[],
                provenance=provenance,
                metadata=metadata,
            )
            elements.append(main)
            if element_type in {ElementType.FIGURE, ElementType.CHART, ElementType.TABLE}:
                caption_text = _caption_text(element_type, content)
                if caption_text:
                    elements.append(
                        self._derived_function_element(
                            parent=main,
                            doc_id=doc_id,
                            page_index=page_index,
                            page_number=page_number,
                            source_index=source_index,
                            element_type=ElementType.CAPTION,
                            text=caption_text,
                            raw_bbox=content.get("caption_bbox"),
                            page_width=page_width,
                            page_height=page_height,
                            content_path=content_path,
                            input_path=input_path,
                            raw_payload={"derived_from": block, "caption": caption_text},
                            reading_order=len(elements),
                        )
                    )
                footnote_text = _footnote_text(element_type, content)
                if footnote_text:
                    elements.append(
                        self._derived_function_element(
                            parent=main,
                            doc_id=doc_id,
                            page_index=page_index,
                            page_number=page_number,
                            source_index=source_index,
                            element_type=ElementType.FOOTNOTE,
                            text=footnote_text,
                            raw_bbox=content.get("footnote_bbox"),
                            page_width=page_width,
                            page_height=page_height,
                            content_path=content_path,
                            input_path=input_path,
                            raw_payload={"derived_from": block, "footnote": footnote_text},
                            reading_order=len(elements),
                        )
                    )
        return elements

    def _derived_function_element(
        self,
        *,
        parent: Element,
        doc_id: str,
        page_index: int,
        page_number: int,
        source_index: int,
        element_type: ElementType,
        text: str,
        raw_bbox: Any,
        page_width: float,
        page_height: float,
        content_path: Path,
        input_path: Path,
        raw_payload: dict[str, Any],
        reading_order: int,
    ) -> Element:
        derived_id = element_id(doc_id, page_index, source_index, element_type.value, parent.element_type.value)
        return Element(
            element_id=derived_id,
            document_id=doc_id,
            page_id=parent.page_id,
            page_number=page_number,
            element_type=element_type,
            reading_order=reading_order,
            bbox=self._bbox(
                owner_id=derived_id,
                raw_bbox=raw_bbox,
                page_width=page_width,
                page_height=page_height,
                context=f"derived {element_type.value} for {parent.element_id}",
            ),
            column_index=parent.column_index,
            text=text,
            summary=None,
            keywords=[],
            provenance=self._provenance(
                source_path=content_path.relative_to(input_path),
                source_locator=f"derived:{parent.element_id}:{element_type.value}",
                raw_payload=raw_payload,
            ),
            metadata={
                "target_element_id": parent.element_id,
                "derived_from_element_id": parent.element_id,
            },
        )

    def _build_sections(self, doc_id: str, pages: list[Page], elements: list[Element]) -> list[Section]:
        page_by_id = {page.page_id: page for page in pages}
        ordered = sorted(elements, key=lambda item: (page_by_id[item.page_id].page_index, item.reading_order))
        stack: list[Section] = []
        sections: list[Section] = []
        for element in ordered:
            if element.element_type == ElementType.HEADING:
                level = element.heading_level or 1
                while stack and stack[-1].level >= level:
                    stack.pop()
                path = [section.title for section in stack] + [element.text or ""]
                current = Section(
                    section_id=section_id(doc_id, element.element_id),
                    document_id=doc_id,
                    title=element.text or "",
                    level=level,
                    heading_element_id=element.element_id,
                    parent_section_id=stack[-1].section_id if stack else None,
                    section_path=path,
                    page_ids=[element.page_id],
                    element_ids=[element.element_id],
                    provenance=element.provenance,
                    metadata={"created_by": "heading_stack"},
                )
                sections.append(current)
                stack.append(current)
                element.section_id = current.section_id
                element.section_path = path
                continue
            if not stack:
                continue
            current = stack[-1]
            element.section_id = current.section_id
            element.section_path = list(current.section_path)
            if element.element_id not in current.element_ids:
                current.element_ids.append(element.element_id)
            if element.page_id not in current.page_ids:
                current.page_ids.append(element.page_id)
        return sections

    def _layout_pages(self, layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
        raw_pages = layout.get("pdf_info", layout.get("pages", []))
        if not isinstance(raw_pages, list):
            raise ValueError("MinerU layout pages must be a list")
        result: dict[int, dict[str, Any]] = {}
        for fallback_index, page in enumerate(raw_pages):
            if not isinstance(page, dict):
                self._warn("invalid_page", f"Layout page {fallback_index} is not an object", page)
                continue
            index = int(page.get("page_idx", page.get("page_index", fallback_index)))
            result[index] = page
        return result

    def _content_pages(self, payload: Any) -> dict[int, list[dict[str, Any]]]:
        if isinstance(payload, dict):
            payload = payload.get("pages", payload.get("content", payload))
        if not isinstance(payload, list):
            raise ValueError("MinerU content_list_v2 must be a list or contain a pages list")
        if all(isinstance(item, list) for item in payload):
            return {index: blocks for index, blocks in enumerate(payload)}
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in payload:
            if not isinstance(item, dict):
                self._warn("invalid_content_item", "Content item is not an object", item)
                continue
            index = int(item.get("page_idx", item.get("page_index", 0)))
            grouped[index].append(item)
        return dict(grouped)

    def _page_size(self, page: dict[str, Any]) -> tuple[float, float]:
        size = page.get("page_size")
        if isinstance(size, dict):
            width, height = size.get("width"), size.get("height")
        elif isinstance(size, (list, tuple)) and len(size) >= 2:
            width, height = size[0], size[1]
        else:
            width, height = page.get("width"), page.get("height")
        if not width or not height or float(width) <= 0 or float(height) <= 0:
            self._warn("missing_page_size", "Missing page size; using normalized 1000x1000 canvas", page)
            return 1000.0, 1000.0
        return float(width), float(height)

    def _bbox(
        self,
        *,
        owner_id: str,
        raw_bbox: Any,
        page_width: float,
        page_height: float,
        context: str,
    ) -> BoundingBox | None:
        if raw_bbox in (None, [], ()):
            return None
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            self._warn("invalid_bbox", f"Invalid bbox shape at {context}", raw_bbox)
            return None
        values = tuple(float(value) for value in raw_bbox)
        coordinate_system = "page"
        if (
            (values[2] > page_width * 1.05 or values[3] > page_height * 1.05)
            and min(values) >= 0
            and max(values) <= 1000
        ):
            coordinate_system = "normalized_1000"
        try:
            return BoundingBox.from_raw(
                bbox_id=bbox_id(owner_id),
                raw=values,
                page_width=page_width,
                page_height=page_height,
                coordinate_system=coordinate_system,
            )
        except ValueError as exc:
            self._warn("invalid_bbox", f"{context}: {exc}", raw_bbox)
            return None

    def _copy_page_asset(
        self,
        input_path: Path,
        output_dir: Path,
        page: dict[str, Any],
        page_index: int,
        owner_id: str,
    ) -> Path | None:
        candidates: list[Path] = []
        declared = _first_string(page, "image_path", "page_image", "path")
        if declared:
            candidates.append(input_path / declared)
        for directory in (input_path / "pages", input_path / "images"):
            for name in (
                f"page_{page_index}.png",
                f"page_{page_index + 1}.png",
                f"page_{page_index:04d}.png",
                f"page_{page_index + 1:04d}.png",
            ):
                candidates.append(directory / name)
        source = next((candidate for candidate in candidates if candidate.is_file()), None)
        return self._copy_asset(source, output_dir / "assets" / "pages", owner_id, output_dir)

    def _copy_element_asset(
        self,
        input_path: Path,
        output_dir: Path,
        block: dict[str, Any],
        content: dict[str, Any],
        owner_id: str,
    ) -> Path | None:
        source_payload = content.get("image_source") if isinstance(content.get("image_source"), dict) else {}
        declared = (
            source_payload.get("path")
            or content.get("image_path")
            or content.get("img_path")
            or block.get("image_path")
            or block.get("img_path")
        )
        source = input_path / str(declared) if declared else None
        if source is not None and not source.is_file():
            self._warn("missing_asset", f"Referenced element asset does not exist: {declared}", {"owner_id": owner_id})
            source = None
        return self._copy_asset(source, output_dir / "assets" / "elements", owner_id, output_dir)

    def _copy_asset(self, source: Path | None, destination_dir: Path, owner_id: str, output_dir: Path) -> Path | None:
        if source is None:
            return None
        destination_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower() or ".bin"
        destination = destination_dir / f"{stable_digest(owner_id)}{suffix}"
        shutil.copy2(source, destination)
        return destination.relative_to(output_dir)

    def _record_unknown_fields(self, block: dict[str, Any], page_index: int, block_index: int) -> None:
        unknown_block = sorted(set(block) - _KNOWN_BLOCK_KEYS)
        content = block.get("content") if isinstance(block.get("content"), dict) else {}
        unknown_content = sorted(set(content) - _KNOWN_CONTENT_KEYS)
        if unknown_block or unknown_content:
            self._warn(
                "unrecognized_fields",
                f"Unrecognized MinerU fields at page {page_index}, block {block_index}",
                {"block_fields": unknown_block, "content_fields": unknown_content},
            )

    def _provenance(
        self,
        *,
        source_path: Path,
        source_locator: str,
        raw_payload: dict[str, Any],
    ) -> Provenance:
        return Provenance(
            provenance_id=provenance_id(self.adapter_name, source_path.as_posix(), source_locator),
            adapter=self.adapter_name,
            source_path=source_path,
            source_locator=source_locator,
            raw_payload=raw_payload,
        )

    def _warn(self, code: str, message: str, payload: Any) -> None:
        item = {"code": code, "message": message, "payload": payload}
        self.warnings.append(item)
        logger.warning("%s: %s", code, message)

    @staticmethod
    def _find_content_path(input_path: Path) -> Path:
        direct = input_path / "content_list_v2.json"
        if direct.exists():
            return direct
        candidates = sorted(input_path.glob("*_content_list_v2.json"))
        if not candidates:
            raise FileNotFoundError(f"Missing *_content_list_v2.json under {input_path}")
        return candidates[0]

    @staticmethod
    def _load_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def _first_string(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _optional_text(value: Any) -> str | None:
    text = _spans_text(value)
    return text or None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _heading_level(block: dict[str, Any], content: dict[str, Any]) -> int:
    return max(1, int(content.get("level", block.get("level", 1)) or 1))


def _element_content(
    element_type: ElementType,
    block: dict[str, Any],
    content: dict[str, Any],
) -> tuple[str | None, str | None]:
    if element_type == ElementType.HEADING:
        return _optional_text(content.get("title_content", block.get("text"))), None
    if element_type == ElementType.PARAGRAPH:
        return _optional_text(content.get("paragraph_content", content.get("text", block.get("text")))), None
    if element_type == ElementType.LIST:
        items = content.get("list_items", [])
        text = "\n".join(
            part
            for item in items
            if isinstance(item, dict)
            for part in [_spans_text(item.get("item_content"))]
            if part
        )
        return text or _optional_text(block.get("text")), None
    if element_type == ElementType.TABLE:
        html = _optional_text(content.get("html"))
        return _optional_text(content.get("content", block.get("text"))), html
    if element_type in {ElementType.FIGURE, ElementType.CHART}:
        return _optional_text(content.get("content", block.get("text"))), None
    if element_type == ElementType.EQUATION:
        return _optional_text(content.get("math_content", content.get("content", block.get("text")))), None
    if element_type == ElementType.CAPTION:
        return _optional_text(content.get("caption_content", content.get("text", block.get("text")))), None
    if element_type == ElementType.FOOTNOTE:
        return _optional_text(content.get("footnote_content", content.get("text", block.get("text")))), None
    return _optional_text(block.get("text")), None


def _spans_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return _spans_text(value.get("content", value.get("text", "")))
    if isinstance(value, list):
        parts = [_spans_text(item) for item in value]
        return "".join(part for part in parts if part).strip()
    return str(value).strip()


def _caption_text(element_type: ElementType, content: dict[str, Any]) -> str:
    key = {
        ElementType.FIGURE: "image_caption",
        ElementType.CHART: "chart_caption",
        ElementType.TABLE: "table_caption",
    }[element_type]
    return _spans_text(content.get(key))


def _footnote_text(element_type: ElementType, content: dict[str, Any]) -> str:
    key = {
        ElementType.FIGURE: "image_footnote",
        ElementType.CHART: "chart_footnote",
        ElementType.TABLE: "table_footnote",
    }[element_type]
    return _spans_text(content.get(key))
