"""MinerU artifact adapter. MinerU-specific names are confined to this module."""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from softdoc.assets import crop_page_bbox
from softdoc.ids import (
    bbox_id,
    document_id,
    element_id,
    page_id,
    provenance_id,
    stable_digest,
)
from softdoc.models import (
    BoundingBox,
    ContentAvailability,
    Document,
    Element,
    ElementParseStatus,
    ElementType,
    Page,
    Provenance,
)

logger = logging.getLogger(__name__)


_HTML_IMAGE_SOURCE = re.compile(
    r"(?P<prefix><img\b[^>]*?\bsrc\s*=\s*)(?P<quote>[\"'])(?P<src>.*?)(?P=quote)",
    re.IGNORECASE,
)


_BLOCK_TYPE_MAP: dict[str, ElementType] = {
    "title": ElementType.HEADING,
    "heading": ElementType.HEADING,
    "paragraph": ElementType.PARAGRAPH,
    "text": ElementType.PARAGRAPH,
    "table": ElementType.TABLE,
    "image": ElementType.FIGURE,
    "figure": ElementType.FIGURE,
    "chart": ElementType.CHART,
    "code": ElementType.CODE,
    "algorithm": ElementType.ALGORITHM,
    "caption": ElementType.CAPTION,
    "footnote": ElementType.FOOTNOTE,
    "page_footnote": ElementType.FOOTNOTE,
    "page_header": ElementType.PARAGRAPH,
    "page_footer": ElementType.PARAGRAPH,
    "header": ElementType.PARAGRAPH,
    "footer": ElementType.PARAGRAPH,
    "list": ElementType.LIST,
    # MinerU 3.x uses ``index`` for some visually grouped text lists.  It is
    # a layout block name, not necessarily a back-of-book index section.
    "index": ElementType.LIST,
    "equation": ElementType.EQUATION,
    "equation_interline": ElementType.EQUATION,
}

_IGNORED_BLOCK_TYPES = {
    "page_number",
    "page_aside_text",
}
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
    "sub_type",
    "img_path",
    "image_path",
}
_KNOWN_CONTENT_KEYS = {
    "title_content",
    "paragraph_content",
    "page_footnote_content",
    "page_footer_content",
    "page_header_content",
    "page_number_content",
    "list_type",
    "list_items",
    "image_source",
    "image_caption",
    "image_footnote",
    "chart_caption",
    "chart_footnote",
    "code_caption",
    "code_content",
    "code_footnote",
    "code_language",
    "algorithm_caption",
    "algorithm_content",
    "algorithm_footnote",
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
        self.backend: str = "pipeline"

    def parse(self, input_path: Path, output_dir: Path) -> Document:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        self.warnings = []
        if not input_path.is_dir():
            raise NotADirectoryError(f"MinerU input must be a directory: {input_path}")
        content_path = self._find_content_path(input_path)
        layout_path, content_bbox_coordinate_system = self._find_layout_path(input_path)
        layout = self._load_json(layout_path)
        content_payload = self._load_json(content_path)
        self.backend = self._detect_backend(input_path, content_payload)
        layout_pages = self._layout_pages(layout)
        content_pages = self._content_pages(content_payload)

        declared_id = _first_string(layout, "document_id", "doc_id", "id")
        source_name = self._artifact_stem(content_path) or input_path.name
        doc_id = document_id(source_name, declared_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "assets" / "pages").mkdir(parents=True, exist_ok=True)
        (output_dir / "assets" / "elements").mkdir(parents=True, exist_ok=True)

        page_indexes = sorted(set(layout_pages) | set(content_pages))
        rendered_page_images = self._render_page_assets(
            input_path=input_path,
            output_dir=output_dir,
            doc_id=doc_id,
            page_indexes=page_indexes,
        )
        pages: list[Page] = []
        elements: list[Element] = []
        for fallback_index, current_page_index in enumerate(page_indexes):
            page_payload = layout_pages.get(current_page_index, {})
            width, height = self._page_size(page_payload)
            page_number = int(page_payload.get("page_number") or current_page_index + 1)
            current_page_id = page_id(doc_id, current_page_index)
            page_image = rendered_page_images.get(current_page_index) or self._copy_page_asset(
                input_path=input_path,
                output_dir=output_dir,
                page=page_payload,
                page_index=current_page_index,
                owner_id=current_page_id,
            )
            blocks = content_pages.get(current_page_index, [])
            layout_blocks = self._align_layout_blocks(page_payload, blocks)
            page_elements = self._parse_page_elements(
                input_path=input_path,
                output_dir=output_dir,
                doc_id=doc_id,
                current_page_id=current_page_id,
                page_index=current_page_index,
                page_number=page_number,
                page_width=width,
                page_height=height,
                page_image=page_image,
                page_payload=page_payload,
                blocks=blocks,
                layout_blocks=layout_blocks,
                content_bbox_coordinate_system=content_bbox_coordinate_system,
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
                    metadata={
                        "source_page_index": current_page_index,
                        # Preserve MinerU's page-number blocks as parser
                        # signals without promoting them to content Elements
                        # or deciding whether they are trustworthy labels.
                        "parser_page_label_signals": self._page_label_signals(
                            doc_id=doc_id,
                            page_index=current_page_index,
                            page_width=width,
                            page_height=height,
                            blocks=blocks,
                            coordinate_system=content_bbox_coordinate_system,
                        ),
                    },
                )
            )

        source_pdf = self._find_source_pdf(input_path)
        doc_provenance = self._provenance(
            source_path=layout_path.relative_to(input_path),
            source_locator="document",
            raw_payload={
                "layout_metadata": {key: value for key, value in layout.items() if key not in {"pdf_info", "pages"}},
                "content_file": content_path.name,
                "layout_file": layout_path.name,
            },
            parser_version=_first_string(layout, "_version_name", "version"),
        )
        document = Document(
            document_id=doc_id,
            title=_first_string(layout, "title", "document_title"),
            source_path=Path(source_pdf.name if source_pdf else source_name),
            pages=pages,
            sections=[],
            elements=elements,
            relations=[],
            provenance=doc_provenance,
            metadata={
                "adapter": self.adapter_name,
                "parser_backend": self.backend,
                "parser_capabilities": {
                    "vlm_visual_description": self.backend == "hybrid",
                    "typed_lists": self.backend == "hybrid",
                    "visual_subtypes": self.backend == "hybrid",
                },
                "adapter_warnings": self.warnings,
                "summary_generation": "disabled",
                "keyword_generation": "disabled",
            },
        )
        return document

    @staticmethod
    def _detect_backend(input_path: Path, content_payload: Any) -> str:
        """Identify MinerU's parser family without leaking it into core models.

        The output directory name is the most reliable signal produced by the
        current MinerU CLI.  ``sub_type`` is a fallback for copied/renamed
        Hybrid artifacts.
        """

        if "hybrid" in input_path.name.casefold():
            return "hybrid"

        pending: list[Any] = [content_payload]
        inspected = 0
        while pending and inspected < 500:
            value = pending.pop()
            inspected += 1
            if isinstance(value, dict):
                if value.get("sub_type") is not None:
                    return "hybrid"
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value[:100])
        return "pipeline"

    def _page_label_signals(
        self,
        *,
        doc_id: str,
        page_index: int,
        page_width: float,
        page_height: float,
        blocks: list[dict[str, Any]],
        coordinate_system: str,
    ) -> list[dict[str, Any]]:
        """Faithfully retain parser page-number blocks for a later pass."""

        signals: list[dict[str, Any]] = []
        for block_index, block in enumerate(blocks):
            if str(block.get("type") or "").strip().casefold() != "page_number":
                continue
            content = block.get("content")
            if not isinstance(content, dict):
                content = {}
            text = _spans_text(
                content.get("page_number_content", block.get("text"))
            )
            owner_id = f"{doc_id}:page:{page_index:04d}:page-label:{block_index:04d}"
            bbox = self._bbox(
                owner_id=owner_id,
                raw_bbox=block.get("bbox"),
                page_width=page_width,
                page_height=page_height,
                context=f"page[{page_index}].block[{block_index}].page_number",
                coordinate_system=coordinate_system,
            )
            signals.append(
                {
                    "text": text,
                    "normalized_bbox": list(bbox.normalized) if bbox else None,
                    "source_locator": f"page[{page_index}].block[{block_index}]",
                    "raw_payload": block,
                }
            )
        return signals

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
        page_image: Path | None,
        page_payload: dict[str, Any],
        blocks: list[dict[str, Any]],
        layout_blocks: list[dict[str, Any] | None],
        content_bbox_coordinate_system: str,
        content_path: Path,
    ) -> list[Element]:
        elements: list[Element] = []
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                self._warn("invalid_block", f"Page {page_index} block {block_index} is not an object", block)
                continue
            layout_block = layout_blocks[block_index] if block_index < len(layout_blocks) else None
            try:
                converted = self._parse_block_elements(
                    input_path=input_path,
                    output_dir=output_dir,
                    doc_id=doc_id,
                    current_page_id=current_page_id,
                    page_index=page_index,
                    page_number=page_number,
                    page_width=page_width,
                    page_height=page_height,
                    page_image=page_image,
                    page_payload=page_payload,
                    block=block,
                    block_index=block_index,
                    layout_block=layout_block,
                    content_bbox_coordinate_system=content_bbox_coordinate_system,
                    content_path=content_path,
                    reading_order_start=len(elements),
                )
            except Exception as exc:
                self._warn(
                    "block_conversion_failed",
                    (
                        f"Failed to convert page {page_index} block {block_index}; "
                        f"continuing with later blocks: {type(exc).__name__}: {exc}"
                    ),
                    {
                        "page_index": page_index,
                        "block_index": block_index,
                        "source_locator": f"page[{page_index}].block[{block_index}]",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "raw_payload": block,
                        "layout_payload": layout_block,
                    },
                )
                continue
            elements.extend(converted)
        return elements

    def _parse_block_elements(
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
        page_image: Path | None,
        page_payload: dict[str, Any],
        block: dict[str, Any],
        block_index: int,
        layout_block: dict[str, Any] | None,
        content_bbox_coordinate_system: str,
        content_path: Path,
        reading_order_start: int,
    ) -> list[Element]:
        self._record_unknown_fields(block, page_index, block_index)
        raw_type = str(block.get("type") or "").strip().lower()
        if raw_type in _IGNORED_BLOCK_TYPES:
            self._warn("ignored_block_type", f"Ignored MinerU auxiliary type: {raw_type}", block)
            return []
        element_type = _BLOCK_TYPE_MAP.get(raw_type)
        if element_type is None:
            self._warn("unsupported_block_type", f"Unsupported MinerU block type: {raw_type or '<missing>'}", block)
            return []

        source_index = int(block.get("index", block_index))
        main_id = element_id(doc_id, page_index, source_index, element_type.value)
        content = block.get("content") if isinstance(block.get("content"), dict) else {}
        text, html = _element_content(element_type, block, content)
        html_recovery: dict[str, Any] | None = None
        if element_type == ElementType.TABLE and not (html or "").strip():
            recovered = self._recover_table_html_from_preproc(
                page_payload=page_payload,
                layout_block=layout_block,
            )
            if recovered is not None:
                html, html_recovery = recovered
                self._warn(
                    "table_html_recovered_from_preproc",
                    (
                        f"Recovered missing table HTML from MinerU preproc_blocks "
                        f"at page {page_index}, block {block_index}"
                    ),
                    {
                        "page_index": page_index,
                        "block_index": block_index,
                        "source_locator": f"page[{page_index}].block[{block_index}]",
                        **html_recovery,
                    },
                )
        if element_type == ElementType.HEADING and not (text or "").strip():
            self._warn(
                "empty_heading_skipped",
                f"Skipped empty heading at page {page_index}, block {block_index}",
                {
                    "page_index": page_index,
                    "block_index": block_index,
                    "source_locator": f"page[{page_index}].block[{block_index}]",
                    "raw_payload": block,
                    "layout_payload": layout_block,
                },
            )
            return []

        image_path = self._copy_element_asset(input_path, output_dir, block, content, main_id)
        embedded_html_assets: list[dict[str, Any]] = []
        if html:
            html, embedded_html_assets = self._copy_embedded_html_assets(
                input_path=input_path,
                output_dir=output_dir,
                html=html,
                owner_id=main_id,
            )
        layout_bbox = layout_block.get("bbox") if layout_block else None
        bbox = self._bbox(
            owner_id=main_id,
            raw_bbox=layout_bbox or block.get("bbox"),
            page_width=page_width,
            page_height=page_height,
            context=f"page={page_index} block={block_index}",
            coordinate_system=(
                "page" if layout_bbox is not None else content_bbox_coordinate_system
            ),
        )
        parse_status = ElementParseStatus.PARSED
        content_availability: ContentAvailability | None = None
        crop_image_path: Path | None = None
        if (
            element_type == ElementType.TABLE
            and not any(((text or "").strip(), (html or "").strip(), image_path))
        ):
            parse_status = ElementParseStatus.DEGRADED
            crop_image_path = self._crop_page_region(
                output_dir=output_dir,
                page_image=page_image,
                bbox=bbox,
                owner_id=main_id,
            )
            content_availability = (
                ContentAvailability.VISUAL_ONLY
                if crop_image_path is not None
                else ContentAvailability.UNAVAILABLE
            )
            self._warn(
                "degraded_table",
                (
                    f"Preserved empty table at page {page_index}, block {block_index} "
                    + (
                        "as a fallback page crop"
                        if crop_image_path is not None
                        else "without recoverable content"
                    )
                ),
                {
                    "page_index": page_index,
                    "block_index": block_index,
                    "source_locator": f"page[{page_index}].block[{block_index}]",
                    "content_availability": content_availability.value,
                    "fallback_crop": (
                        crop_image_path.as_posix()
                        if crop_image_path is not None
                        else None
                    ),
                    "raw_payload": block,
                    "layout_payload": layout_block,
                },
            )

        provenance_metadata = {"layout_payload": layout_block} if layout_block else {}
        if html_recovery is not None:
            provenance_metadata["html_recovery"] = html_recovery
        provenance = self._provenance(
            source_path=content_path.relative_to(input_path),
            source_locator=f"page[{page_index}].block[{block_index}]",
            raw_payload=block,
            metadata=provenance_metadata,
        )
        metadata = dict(block.get("metadata") or {})
        metadata["mineru_type"] = raw_type
        metadata["parser_backend"] = self.backend
        if html_recovery is not None:
            metadata["html_recovery"] = html_recovery
        if embedded_html_assets:
            metadata["embedded_html_assets"] = embedded_html_assets
        block_sub_type = block.get("sub_type")
        if block_sub_type is not None:
            metadata["mineru_sub_type"] = str(block_sub_type)
        if (
            self.backend == "hybrid"
            and element_type in {ElementType.FIGURE, ElementType.CHART}
            and bool((text or "").strip())
        ):
            # MinerU Hybrid's visual body is VLM-generated.  Preserve it for a
            # later visual read, but do not silently treat it like OCR/native
            # text in deterministic retrieval.
            metadata["text_source"] = "vlm_generated_visual_description"
            metadata["retrieval_text_status"] = "unverified"
            metadata["generated_visual_text"] = True
        if layout_block:
            if layout_block.get("score") is not None:
                metadata["parser_score"] = layout_block["score"]
            if layout_block.get("sub_type") is not None:
                metadata["mineru_sub_type"] = layout_block["sub_type"]
        if isinstance(content.get("style"), dict):
            metadata["style"] = dict(content["style"])
        if content.get("column_count") is not None:
            metadata["column_count"] = content.get("column_count")
        if content.get("code_language") is not None:
            metadata["code_language"] = content.get("code_language")
        if raw_type == "index":
            metadata["grouped_items"] = _grouped_text_items(
                content,
                layout_block,
                page_width,
                page_height,
            )
        main = Element(
            element_id=main_id,
            document_id=doc_id,
            page_id=current_page_id,
            page_number=page_number,
            element_type=element_type,
            reading_order=reading_order_start,
            bbox=bbox,
            column_index=_optional_int(block.get("column_index", content.get("column_index"))),
            heading_level=_heading_level(block, content) if element_type == ElementType.HEADING else None,
            text=text,
            html=html,
            image_path=image_path,
            crop_image_path=crop_image_path,
            parse_status=parse_status,
            content_availability=content_availability,
            reference_label=_optional_text(content.get("reference_label")),
            summary=None,
            keywords=[],
            provenance=provenance,
            metadata=metadata,
        )
        converted = [main]
        if element_type not in {
            ElementType.FIGURE,
            ElementType.CHART,
            ElementType.TABLE,
            ElementType.CODE,
            ElementType.ALGORITHM,
        }:
            return converted

        caption_items = _function_items(
            element_type,
            content,
            layout_block,
            "caption",
        )
        for item_index, (caption_text, caption_bbox) in enumerate(
            caption_items
        ):
            converted.append(
                self._derived_function_element(
                    parent=main,
                    doc_id=doc_id,
                    page_index=page_index,
                    page_number=page_number,
                    source_index=source_index,
                    element_type=ElementType.CAPTION,
                    text=caption_text,
                    raw_bbox=caption_bbox or content.get("caption_bbox"),
                    page_width=page_width,
                    page_height=page_height,
                    coordinate_system=(
                        "page"
                        if caption_bbox is not None
                        else content_bbox_coordinate_system
                    ),
                    content_path=content_path,
                    input_path=input_path,
                    raw_payload={"derived_from": block, "caption": caption_text},
                    reading_order=reading_order_start + len(converted),
                    role=(
                        None
                        if len(caption_items) == 1
                        else f"{element_type.value}-{item_index + 1}"
                    ),
                )
            )
        footnote_items = _function_items(
            element_type,
            content,
            layout_block,
            "footnote",
        )
        for item_index, (footnote_text, footnote_bbox) in enumerate(
            footnote_items
        ):
            converted.append(
                self._derived_function_element(
                    parent=main,
                    doc_id=doc_id,
                    page_index=page_index,
                    page_number=page_number,
                    source_index=source_index,
                    element_type=ElementType.FOOTNOTE,
                    text=footnote_text,
                    raw_bbox=footnote_bbox or content.get("footnote_bbox"),
                    page_width=page_width,
                    page_height=page_height,
                    coordinate_system=(
                        "page"
                        if footnote_bbox is not None
                        else content_bbox_coordinate_system
                    ),
                    content_path=content_path,
                    input_path=input_path,
                    raw_payload={"derived_from": block, "footnote": footnote_text},
                    reading_order=reading_order_start + len(converted),
                    role=(
                        None
                        if len(footnote_items) == 1
                        else f"{element_type.value}-{item_index + 1}"
                    ),
                )
            )
        return converted

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
        coordinate_system: str,
        content_path: Path,
        input_path: Path,
        raw_payload: dict[str, Any],
        reading_order: int,
        role: str | None = None,
    ) -> Element:
        derived_id = element_id(
            doc_id,
            page_index,
            source_index,
            element_type.value,
            role or parent.element_type.value,
        )
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
                coordinate_system=coordinate_system,
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
        coordinate_system: str | None = None,
    ) -> BoundingBox | None:
        if raw_bbox in (None, [], ()):
            return None
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            self._warn("invalid_bbox", f"Invalid bbox shape at {context}", raw_bbox)
            return None
        values = tuple(float(value) for value in raw_bbox)
        looks_normalized_1000 = (
            (
                values[2] > page_width * 1.05
                or values[3] > page_height * 1.05
            )
            and min(values) >= -2
            and max(values) <= 1002
        )
        if coordinate_system is None:
            coordinate_system = (
                "normalized_1000"
                if looks_normalized_1000
                else "page"
            )
        elif coordinate_system == "page" and looks_normalized_1000:
            self._warn(
                "bbox_coordinate_system_corrected",
                (
                    f"{context}: bbox exceeds the page canvas but fits "
                    "MinerU's normalized-1000 coordinate system"
                ),
                raw_bbox,
            )
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
            if coordinate_system == "normalized_1000":
                unclipped = tuple(value / 1000.0 for value in values)
            else:
                unclipped = (
                    values[0] / page_width,
                    values[1] / page_height,
                    values[2] / page_width,
                    values[3] / page_height,
                )
            if (
                min(unclipped) >= -0.002
                and max(unclipped) <= 1.002
            ):
                normalized = tuple(
                    min(1.0, max(0.0, value))
                    for value in unclipped
                )
                if (
                    normalized[0] < normalized[2]
                    and normalized[1] < normalized[3]
                ):
                    self._warn(
                        "bbox_boundary_clipped",
                        (
                            f"{context}: clipped a small page-boundary "
                            "rounding overflow while preserving raw coordinates"
                        ),
                        raw_bbox,
                    )
                    return BoundingBox(
                        bbox_id=bbox_id(owner_id),
                        raw=values,
                        normalized=normalized,
                        coordinate_system=coordinate_system,
                    )
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

    def _copy_embedded_html_assets(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        html: str,
        owner_id: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Copy local ``<img src>`` resources and rewrite their HTML paths.

        MinerU may place visual table cells in separate image files referenced
        only from the table HTML.  Copying just ``content.image_source`` keeps
        the outer table crop but leaves those cell images dangling.  Raw MinerU
        HTML remains available in provenance; the Element HTML is rewritten to
        portable paths inside the SoftDoc output.
        """

        records: list[dict[str, Any]] = []
        input_root = input_path.resolve()

        def replace(match: re.Match[str]) -> str:
            source_text = match.group("src").strip()
            record: dict[str, Any] = {
                "original_src": source_text,
                "stored_path": None,
                "available": False,
            }
            records.append(record)
            if not source_text:
                record["reason"] = "empty_src"
                return match.group(0)
            lowered = source_text.casefold()
            if lowered.startswith("data:"):
                record["available"] = True
                record["reason"] = "inline_data_uri"
                return match.group(0)
            if lowered.startswith(("http://", "https://")):
                record["reason"] = "external_url_not_copied"
                return match.group(0)

            # Forward slashes are valid on Windows and native on Linux;
            # normalize the uncommon backslash form for cross-platform runs.
            relative_source = Path(source_text.replace("\\", "/"))
            candidates = [input_root / relative_source]
            if len(relative_source.parts) == 1:
                candidates.append(input_root / "images" / relative_source.name)
            source = next((path for path in candidates if path.is_file()), None)
            if source is None:
                record["reason"] = "source_file_missing"
                self._warn(
                    "missing_embedded_html_asset",
                    f"Referenced HTML image asset does not exist: {source_text}",
                    {"owner_id": owner_id, "src": source_text},
                )
                return match.group(0)
            try:
                source.resolve().relative_to(input_root)
            except ValueError:
                record["reason"] = "source_outside_input_root"
                self._warn(
                    "unsafe_embedded_html_asset",
                    f"Ignored HTML image outside MinerU input: {source_text}",
                    {"owner_id": owner_id, "src": source_text},
                )
                return match.group(0)

            copied = self._copy_asset(
                source,
                output_dir / "assets" / "elements",
                f"{owner_id}:embedded-html:{len(records) - 1}:{source_text}",
                output_dir,
            )
            if copied is None:  # Defensive; ``source`` is a verified file.
                record["reason"] = "copy_failed"
                return match.group(0)
            portable = copied.as_posix()
            record["stored_path"] = portable
            record["available"] = True
            record["reason"] = "copied"
            return (
                f"{match.group('prefix')}{match.group('quote')}"
                f"{portable}{match.group('quote')}"
            )

        return _HTML_IMAGE_SOURCE.sub(replace, html), records

    def _copy_asset(self, source: Path | None, destination_dir: Path, owner_id: str, output_dir: Path) -> Path | None:
        if source is None:
            return None
        destination_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower() or ".bin"
        destination = destination_dir / f"{stable_digest(owner_id)}{suffix}"
        shutil.copy2(source, destination)
        return destination.relative_to(output_dir)

    def _crop_page_region(
        self,
        *,
        output_dir: Path,
        page_image: Path | None,
        bbox: BoundingBox | None,
        owner_id: str,
    ) -> Path | None:
        try:
            destination = crop_page_bbox(
                output_dir=output_dir,
                page_image=page_image,
                normalized_bbox=bbox.normalized if bbox else None,
                owner_id=owner_id,
            )
        except Exception as exc:  # Defensive: helper normally returns None.
            self._warn(
                "fallback_crop_failed",
                f"Failed to crop page image for {owner_id}: {type(exc).__name__}: {exc}",
                {
                    "owner_id": owner_id,
                    "page_image": page_image.as_posix(),
                    "bbox": bbox.model_dump(mode="json"),
                },
            )
            return None
        if destination is None and page_image is not None and bbox is not None:
            self._warn(
                "fallback_crop_failed",
                f"Failed to crop page image for {owner_id}",
                {
                    "owner_id": owner_id,
                    "page_image": page_image.as_posix(),
                    "bbox": bbox.model_dump(mode="json"),
                },
            )
        return destination

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
        parser_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Provenance:
        return Provenance(
            provenance_id=provenance_id(self.adapter_name, source_path.as_posix(), source_locator),
            adapter=self.adapter_name,
            source_path=source_path,
            source_locator=source_locator,
            parser_version=parser_version,
            raw_payload=raw_payload,
            metadata=metadata or {},
        )

    def _warn(self, code: str, message: str, payload: Any) -> None:
        item = {"code": code, "message": message, "payload": payload}
        self.warnings.append(item)
        logger.warning("%s: %s", code, message)

    @staticmethod
    def _find_layout_path(input_path: Path) -> tuple[Path, str]:
        legacy_layout = input_path / "layout.json"
        if legacy_layout.exists():
            return legacy_layout, "page"
        direct_middle = input_path / "middle.json"
        if direct_middle.exists():
            return direct_middle, "normalized_1000"
        candidates = sorted(input_path.glob("*_middle.json"))
        if not candidates:
            raise FileNotFoundError(
                f"Missing MinerU layout.json or *_middle.json under {input_path}"
            )
        return candidates[0], "normalized_1000"

    @staticmethod
    def _artifact_stem(content_path: Path) -> str | None:
        suffix = "_content_list_v2.json"
        if content_path.name.endswith(suffix):
            stem = content_path.name[: -len(suffix)]
            return stem or None
        return None

    @staticmethod
    def _find_source_pdf(input_path: Path) -> Path | None:
        preferred = sorted(input_path.glob("*_origin.pdf"))
        if preferred:
            return preferred[0]
        candidates = [
            path
            for path in sorted(input_path.glob("*.pdf"))
            if not path.stem.endswith(("_layout", "_span"))
        ]
        return candidates[0] if candidates else None

    def _render_page_assets(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        doc_id: str,
        page_indexes: list[int],
    ) -> dict[int, Path]:
        source_pdf = self._find_source_pdf(input_path)
        if source_pdf is None:
            return {}
        try:
            import pypdfium2
        except ImportError:
            self._warn(
                "page_rendering_unavailable",
                "pypdfium2 is unavailable; page images were not rendered",
                {"source_pdf": source_pdf.name},
            )
            return {}

        rendered: dict[int, Path] = {}
        document = None
        try:
            document = pypdfium2.PdfDocument(source_pdf)
            for current_page_index in page_indexes:
                if current_page_index < 0 or current_page_index >= len(document):
                    self._warn(
                        "missing_pdf_page",
                        f"Source PDF has no page index {current_page_index}",
                        {"source_pdf": source_pdf.name},
                    )
                    continue
                owner_id = page_id(doc_id, current_page_index)
                destination = (
                    output_dir
                    / "assets"
                    / "pages"
                    / f"{stable_digest(owner_id)}.png"
                )
                if not destination.exists():
                    page = document[current_page_index]
                    bitmap = page.render(scale=1.5)
                    image = bitmap.to_pil()
                    image.save(destination)
                    image.close()
                    bitmap.close()
                    page.close()
                rendered[current_page_index] = destination.relative_to(output_dir)
        except Exception as exc:
            self._warn(
                "page_rendering_failed",
                f"Failed to render page images from {source_pdf.name}: {exc}",
                {"source_pdf": source_pdf.name},
            )
        finally:
            if document is not None:
                document.close()
        return rendered

    @staticmethod
    def _align_layout_blocks(
        page_payload: dict[str, Any],
        content_blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any] | None]:
        layout_blocks = [
            block
            for key in ("para_blocks", "discarded_blocks")
            for block in page_payload.get(key, [])
            if isinstance(block, dict)
        ]
        if not layout_blocks:
            return [None] * len(content_blocks)
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for layout_block in layout_blocks:
            by_type[_normalized_layout_type(layout_block.get("type"))].append(layout_block)
        aligned: list[dict[str, Any] | None] = []
        for content_block in content_blocks:
            raw_type = content_block.get("type") if isinstance(content_block, dict) else None
            candidates = by_type[_normalized_layout_type(raw_type)]
            aligned.append(candidates.pop(0) if candidates else None)
        return aligned

    @staticmethod
    def _recover_table_html_from_preproc(
        *,
        page_payload: dict[str, Any],
        layout_block: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Recover table HTML omitted from content_list_v2.

        Some MinerU pipeline outputs retain an empty table in
        ``*_content_list_v2.json`` while the same table, with complete HTML,
        remains in ``*_middle.json`` under ``preproc_blocks``.  Recovery is
        intentionally conservative: the content-list table must already have
        an aligned layout block, and exactly one preproc table must overlap its
        page-coordinate bbox with IoU >= 0.95.
        """

        reference_bbox = layout_block.get("bbox") if layout_block else None
        if not _valid_bbox_values(reference_bbox):
            return None

        matches: list[tuple[float, int, dict[str, Any], str]] = []
        for candidate_index, candidate in enumerate(
            page_payload.get("preproc_blocks", [])
        ):
            if not isinstance(candidate, dict):
                continue
            if _normalized_layout_type(candidate.get("type")) != "table":
                continue
            candidate_bbox = candidate.get("bbox")
            if not _valid_bbox_values(candidate_bbox):
                continue
            candidate_html = _first_nonempty_html(candidate)
            if candidate_html is None:
                continue
            overlap = _bbox_iou(reference_bbox, candidate_bbox)
            if overlap >= 0.95:
                matches.append(
                    (overlap, candidate_index, candidate, candidate_html)
                )

        if len(matches) != 1:
            return None
        overlap, candidate_index, candidate, candidate_html = matches[0]
        return candidate_html, {
            "source": "page_preproc_blocks",
            "candidate_index": candidate_index,
            "content_bbox": list(reference_bbox),
            "preproc_bbox": list(candidate["bbox"]),
            "bbox_iou": round(overlap, 6),
        }

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


def _normalized_layout_type(value: Any) -> str:
    raw_type = str(value or "").strip().lower()
    return {
        "heading": "title",
        "text": "paragraph",
        "abstract": "paragraph",
        "code": "code_like",
        "algorithm": "code_like",
        "footer": "page_footer",
        "header": "page_header",
        "equation_interline": "equation",
    }.get(raw_type, raw_type)


def _nested_function_bbox(
    layout_block: dict[str, Any] | None,
    role: str,
) -> Any:
    boxes = _nested_function_bboxes(layout_block, role)
    return boxes[0] if boxes else None


def _nested_function_bboxes(
    layout_block: dict[str, Any] | None,
    role: str,
) -> list[Any]:
    if not layout_block:
        return []
    boxes: list[Any] = []
    for child in layout_block.get("blocks", []):
        if not isinstance(child, dict):
            continue
        child_type = str(child.get("type") or "").strip().lower()
        if child_type == role or child_type.endswith(f"_{role}"):
            if child.get("bbox") is not None:
                boxes.append(child["bbox"])
    return boxes


def _element_content(
    element_type: ElementType,
    block: dict[str, Any],
    content: dict[str, Any],
) -> tuple[str | None, str | None]:
    if element_type == ElementType.HEADING:
        return _optional_text(content.get("title_content", block.get("text"))), None
    if element_type == ElementType.PARAGRAPH:
        paragraph_value = content.get("paragraph_content")
        if paragraph_value is None:
            paragraph_value = content.get("page_header_content")
        if paragraph_value is None:
            paragraph_value = content.get("page_footer_content")
        if paragraph_value is None:
            paragraph_value = content.get("text", block.get("text"))
        return _optional_text(paragraph_value), None
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
    if element_type == ElementType.CODE:
        return _optional_text(content.get("code_content", content.get("content", block.get("text")))), None
    if element_type == ElementType.ALGORITHM:
        return _optional_text(
            content.get("algorithm_content", content.get("content", block.get("text")))
        ), None
    if element_type == ElementType.EQUATION:
        return _optional_text(content.get("math_content", content.get("content", block.get("text")))), None
    if element_type == ElementType.CAPTION:
        return _optional_text(content.get("caption_content", content.get("text", block.get("text")))), None
    if element_type == ElementType.FOOTNOTE:
        return _optional_text(
            content.get(
                "footnote_content",
                content.get("page_footnote_content", content.get("text", block.get("text"))),
            )
        ), None
    return _optional_text(block.get("text")), None


def _first_nonempty_html(payload: Any) -> str | None:
    if isinstance(payload, dict):
        direct = payload.get("html")
        if isinstance(direct, str) and direct.strip():
            return direct
        for value in payload.values():
            recovered = _first_nonempty_html(value)
            if recovered is not None:
                return recovered
    elif isinstance(payload, list):
        for value in payload:
            recovered = _first_nonempty_html(value)
            if recovered is not None:
                return recovered
    return None


def _valid_bbox_values(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return x1 < x2 and y1 < y2


def _bbox_iou(first: Any, second: Any) -> float:
    if not _valid_bbox_values(first) or not _valid_bbox_values(second):
        return 0.0
    ax1, ay1, ax2, ay2 = (float(item) for item in first)
    bx1, by1, bx2, by2 = (float(item) for item in second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


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
        ElementType.CODE: "code_caption",
        ElementType.ALGORITHM: "algorithm_caption",
    }[element_type]
    return _spans_text(content.get(key))


def _footnote_text(element_type: ElementType, content: dict[str, Any]) -> str:
    key = {
        ElementType.FIGURE: "image_footnote",
        ElementType.CHART: "chart_footnote",
        ElementType.TABLE: "table_footnote",
        ElementType.CODE: "code_footnote",
        ElementType.ALGORITHM: "algorithm_footnote",
    }[element_type]
    return _spans_text(content.get(key))


def _function_items(
    element_type: ElementType,
    content: dict[str, Any],
    layout_block: dict[str, Any] | None,
    role: str,
) -> list[tuple[str, Any]]:
    """Return distinct functional text blocks when MinerU exposes them.

    MinerU may place several spatially separate labels in one caption array.
    Joining that array destroys both the text boundary and its geometry.  We
    split only when the layout payload independently exposes multiple matching
    blocks; otherwise the original multi-span text remains one element.
    """

    key_by_role = {
        "caption": {
            ElementType.FIGURE: "image_caption",
            ElementType.CHART: "chart_caption",
            ElementType.TABLE: "table_caption",
            ElementType.CODE: "code_caption",
            ElementType.ALGORITHM: "algorithm_caption",
        },
        "footnote": {
            ElementType.FIGURE: "image_footnote",
            ElementType.CHART: "chart_footnote",
            ElementType.TABLE: "table_footnote",
            ElementType.CODE: "code_footnote",
            ElementType.ALGORITHM: "algorithm_footnote",
        },
    }
    key = key_by_role[role][element_type]
    value = content.get(key)
    boxes = _nested_function_bboxes(layout_block, role)
    if isinstance(value, list) and len(boxes) > 1:
        texts = [
            text
            for item in value
            for text in [_spans_text(item)]
            if text
        ]
        if len(texts) == len(boxes):
            return list(zip(texts, boxes, strict=True))
    text = _spans_text(value)
    if not text:
        return []
    fallback_box = boxes[0] if boxes else content.get(f"{role}_bbox")
    return [(text, fallback_box)]


def _grouped_text_items(
    content: dict[str, Any],
    layout_block: dict[str, Any] | None,
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    values = [
        text
        for item in content.get("list_items", [])
        if isinstance(item, dict)
        for text in [_spans_text(item.get("item_content"))]
        if text
    ]
    lines = (
        [
            line
            for line in layout_block.get("lines", [])
            if isinstance(line, dict)
        ]
        if layout_block
        else []
    )
    result: list[dict[str, Any]] = []
    for index, text in enumerate(values):
        raw_bbox = lines[index].get("bbox") if index < len(lines) else None
        normalized_bbox = None
        if (
            isinstance(raw_bbox, list)
            and len(raw_bbox) == 4
            and page_width > 0
            and page_height > 0
        ):
            normalized_bbox = [
                float(raw_bbox[0]) / page_width,
                float(raw_bbox[1]) / page_height,
                float(raw_bbox[2]) / page_width,
                float(raw_bbox[3]) / page_height,
            ]
        result.append(
            {
                "text": text,
                "raw_bbox": raw_bbox,
                "normalized_bbox": normalized_bbox,
            }
        )
    return result
