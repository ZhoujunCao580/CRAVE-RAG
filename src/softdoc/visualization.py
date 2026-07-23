"""Human-inspection overlays for parsed pages."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from softdoc.models import Document, Element, ElementType, RelationType


_COLORS: dict[ElementType, tuple[int, int, int]] = {
    ElementType.HEADING: (180, 30, 30),
    ElementType.PARAGRAPH: (30, 100, 210),
    ElementType.TABLE: (135, 45, 180),
    ElementType.FIGURE: (0, 140, 90),
    ElementType.CHART: (0, 125, 125),
    ElementType.CODE: (70, 70, 70),
    ElementType.ALGORITHM: (90, 80, 150),
    ElementType.CAPTION: (230, 120, 0),
    ElementType.FOOTNOTE: (110, 90, 60),
    ElementType.LIST: (40, 120, 180),
    ElementType.EQUATION: (190, 50, 140),
}

_VISIBLE_RELATIONS = {
    RelationType.CAPTION_OF,
    RelationType.FOOTNOTE_OF,
    RelationType.REFERS_TO,
}


def render_page_overlays(document: Document, output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    overlay_dir = output_dir / "debug" / "page_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    elements = {element.element_id: element for element in document.elements}
    rendered: list[Path] = []
    for page in document.pages:
        canvas = _page_canvas(page.image_path, output_dir, page.width, page.height)
        draw = ImageDraw.Draw(canvas)
        page_elements = [elements[element_id] for element_id in page.reading_order if element_id in elements]
        for element in page_elements:
            if element.bbox is None:
                continue
            rectangle = _pixel_box(element, canvas.width, canvas.height)
            color = _COLORS[element.element_type]
            draw.rectangle(rectangle, outline=color, width=max(2, canvas.width // 500))
            label = _element_label(element)
            label_y = max(0, rectangle[1] - 12)
            draw.rectangle((rectangle[0], label_y, min(canvas.width, rectangle[0] + 8 * len(label)), rectangle[1]), fill=(255, 255, 255))
            draw.text((rectangle[0] + 2, label_y), label, fill=color)
        for relation in document.relations:
            if relation.relation_type not in _VISIBLE_RELATIONS:
                continue
            source = elements.get(relation.source_id)
            target = elements.get(relation.target_id)
            if not source or not target or source.page_id != page.page_id or target.page_id != page.page_id:
                continue
            if not source.bbox or not target.bbox:
                continue
            start = _pixel_center(source, canvas.width, canvas.height)
            end = _pixel_center(target, canvas.width, canvas.height)
            draw.line((start, end), fill=(220, 0, 0), width=max(2, canvas.width // 600))
            middle = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            draw.text(middle, relation.relation_type.value, fill=(220, 0, 0))
        destination = overlay_dir / f"page_{page.page_number:04d}.png"
        canvas.save(destination)
        rendered.append(destination)
    return rendered


def _page_canvas(image_path: Path | None, output_dir: Path, width: float, height: float) -> Image.Image:
    if image_path:
        candidate = image_path if image_path.is_absolute() else output_dir / image_path
        if candidate.is_file():
            return Image.open(candidate).convert("RGB")
    canvas_width = max(400, min(int(round(width)), 1600))
    aspect = height / width
    canvas_height = max(400, min(int(round(canvas_width * aspect)), 2400))
    return Image.new("RGB", (canvas_width, canvas_height), "white")


def _pixel_box(element: Element, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = element.bbox.normalized
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)


def _pixel_center(element: Element, width: int, height: int) -> tuple[int, int]:
    x1, y1, x2, y2 = _pixel_box(element, width, height)
    return (x1 + x2) // 2, (y1 + y2) // 2


def _element_label(element: Element) -> str:
    element_type = element.element_type.value
    if element.element_type == ElementType.HEADING:
        hierarchy = element.metadata.get("heading_hierarchy", {})
        if hierarchy.get("action") == "document_title":
            element_type = "TITLE"
        else:
            element_type = f"H{element.heading_level or '?'}"
    return f"{element.reading_order} {element_type} {element.element_id}"
