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
    stable_digest,
)
from softdoc.hierarchy import HeadingHierarchyBuilder
from softdoc.models import (
    BoundingBox,
    Document,
    Element,
    ElementType,
    Page,
    Provenance,
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
    "code": ElementType.CODE,
    "algorithm": ElementType.ALGORITHM,
    "caption": ElementType.CAPTION,
    "footnote": ElementType.FOOTNOTE,
    "page_footnote": ElementType.FOOTNOTE,
    "list": ElementType.LIST,
    "equation": ElementType.EQUATION,
    "equation_interline": ElementType.EQUATION,
}

_IGNORED_BLOCK_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
    "page_aside_text",
    "header",
    "footer",
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
                    metadata={"source_page_index": current_page_index},
                )
            )

        hierarchy = HeadingHierarchyBuilder().build(doc_id, pages, elements)
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
            title=(
                _first_string(layout, "title", "document_title")
                or hierarchy.document_title
            ),
            source_path=Path(source_pdf.name if source_pdf else source_name),
            pages=pages,
            sections=hierarchy.sections,
            elements=elements,
            relations=[],
            provenance=doc_provenance,
            metadata={
                "adapter": self.adapter_name,
                "adapter_warnings": self.warnings,
                "summary_generation": "disabled",
                "keyword_generation": "disabled",
                "heading_hierarchy": {
                    "created_by": "deterministic_rule",
                },
                "heading_decisions": [
                    decision.model_dump(mode="json")
                    for decision in hierarchy.decisions
                ],
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
        layout_blocks: list[dict[str, Any] | None],
        content_bbox_coordinate_system: str,
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
            layout_block = layout_blocks[block_index] if block_index < len(layout_blocks) else None
            text, html = _element_content(element_type, block, content)
            image_path = self._copy_element_asset(input_path, output_dir, block, content, main_id)
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
            provenance = self._provenance(
                source_path=content_path.relative_to(input_path),
                source_locator=f"page[{page_index}].block[{block_index}]",
                raw_payload=block,
                metadata={"layout_payload": layout_block} if layout_block else {},
            )
            metadata = dict(block.get("metadata") or {})
            metadata["mineru_type"] = raw_type
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
            if element_type in {
                ElementType.FIGURE,
                ElementType.CHART,
                ElementType.TABLE,
                ElementType.CODE,
                ElementType.ALGORITHM,
            }:
                caption_text = _caption_text(element_type, content)
                if caption_text:
                    caption_bbox = _nested_function_bbox(layout_block, "caption")
                    elements.append(
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
                            reading_order=len(elements),
                        )
                    )
                footnote_text = _footnote_text(element_type, content)
                if footnote_text:
                    footnote_bbox = _nested_function_bbox(layout_block, "footnote")
                    elements.append(
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
        coordinate_system: str,
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
        if coordinate_system is None:
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
    if not layout_block:
        return None
    for child in layout_block.get("blocks", []):
        if not isinstance(child, dict):
            continue
        child_type = str(child.get("type") or "").strip().lower()
        if child_type == role or child_type.endswith(f"_{role}"):
            return child.get("bbox")
    return None


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
