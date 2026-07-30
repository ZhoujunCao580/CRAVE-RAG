"""Human-inspection overlays for parsed pages."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from softdoc.models import (
    Document,
    Element,
    ElementType,
    Relation,
    RelationType,
)


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
            if (
                element.bbox is None
                or element.metadata.get("excluded_from_relations") is True
            ):
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


def render_cross_page_relation_overlays(
    document: Document,
    output_dir: Path,
) -> list[Path]:
    """Render adjacent page pairs with their cross-page relations.

    A relation whose endpoints live on different pages cannot be inspected on
    either single-page overlay.  This view keeps both original page images,
    highlights the endpoint boxes, and draws the relation across the gutter.
    """

    output_dir = Path(output_dir)
    overlay_dir = output_dir / "debug" / "cross_page_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    elements = {element.element_id: element for element in document.elements}
    pages = {page.page_id: page for page in document.pages}
    grouped: dict[tuple[str, str], list[Relation]] = {}
    for relation in document.relations:
        source = elements.get(relation.source_id)
        target = elements.get(relation.target_id)
        if source is None or target is None or source.page_id == target.page_id:
            continue
        if source.page_id not in pages or target.page_id not in pages:
            continue
        pair = (source.page_id, target.page_id)
        grouped.setdefault(pair, []).append(relation)

    rendered: list[Path] = []
    for (source_page_id, target_page_id), relations in grouped.items():
        source_page = pages[source_page_id]
        target_page = pages[target_page_id]
        left = _page_canvas(
            source_page.image_path,
            output_dir,
            source_page.width,
            source_page.height,
        )
        right = _page_canvas(
            target_page.image_path,
            output_dir,
            target_page.width,
            target_page.height,
        )
        left, right = _match_image_heights(left, right)
        gutter = max(140, (left.width + right.width) // 14)
        banner_height = 44
        canvas = Image.new(
            "RGB",
            (
                left.width + gutter + right.width,
                banner_height + max(left.height, right.height),
            ),
            "white",
        )
        canvas.paste(left, (0, banner_height))
        canvas.paste(right, (left.width + gutter, banner_height))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 12),
            (
                f"Page {source_page.page_number} -> "
                f"Page {target_page.page_number}"
            ),
            fill=(20, 20, 20),
        )
        for relation_index, relation in enumerate(relations):
            source = elements[relation.source_id]
            target = elements[relation.target_id]
            if source.bbox is None or target.bbox is None:
                continue
            source_box = _pixel_box(source, left.width, left.height)
            target_box = _pixel_box(target, right.width, right.height)
            source_box = tuple(
                value + (banner_height if index % 2 else 0)
                for index, value in enumerate(source_box)
            )
            target_box = (
                target_box[0] + left.width + gutter,
                target_box[1] + banner_height,
                target_box[2] + left.width + gutter,
                target_box[3] + banner_height,
            )
            color = (
                (215, 40, 40)
                if relation.status.value == "confirmed"
                else (215, 125, 0)
            )
            width = max(3, canvas.width // 700)
            draw.rectangle(source_box, outline=color, width=width)
            draw.rectangle(target_box, outline=color, width=width)
            start = (
                source_box[2],
                (source_box[1] + source_box[3]) // 2,
            )
            end = (
                target_box[0],
                (target_box[1] + target_box[3]) // 2,
            )
            offset = (relation_index - (len(relations) - 1) / 2) * 10
            middle_x = left.width + gutter // 2
            draw.line(
                (
                    start,
                    (middle_x, int((start[1] + end[1]) / 2 + offset)),
                    end,
                ),
                fill=color,
                width=width,
            )
            label = (
                f"{relation.relation_type.value} "
                f"{relation.status.value} {relation.confidence:.2f}"
            )
            draw.text(
                (
                    max(left.width + 4, middle_x - 60),
                    int((start[1] + end[1]) / 2 + offset - 12),
                ),
                label,
                fill=color,
            )
        destination = overlay_dir / (
            f"pages_{source_page.page_number:04d}_"
            f"{target_page.page_number:04d}.png"
        )
        canvas.save(destination)
        left.close()
        right.close()
        canvas.close()
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
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    banner_height = max(30, canvas_height // 25)
    draw.rectangle(
        (0, 0, canvas_width, banner_height),
        fill=(170, 0, 0),
    )
    draw.text(
        (10, max(2, banner_height // 4)),
        "PAGE IMAGE UNAVAILABLE - BBOX-ONLY OVERLAY",
        fill=(255, 255, 255),
    )
    return canvas


def _match_image_heights(
    left: Image.Image,
    right: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    target_height = min(max(left.height, right.height), 1800)

    def resized(image: Image.Image) -> Image.Image:
        if image.height == target_height:
            return image.copy()
        target_width = max(
            1,
            int(round(image.width * target_height / image.height)),
        )
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    matched_left = resized(left)
    matched_right = resized(right)
    left.close()
    right.close()
    return matched_left, matched_right


def _pixel_box(element: Element, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = element.bbox.normalized
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)


def _pixel_center(element: Element, width: int, height: int) -> tuple[int, int]:
    x1, y1, x2, y2 = _pixel_box(element, width, height)
    return (x1 + x2) // 2, (y1 + y2) // 2


def _element_label(element: Element) -> str:
    element_type = element.element_type.value
    repeated_region = element.metadata.get("repeated_region")
    if repeated_region == "page_header":
        element_type = "HEADER"
    elif repeated_region == "page_footer":
        element_type = "FOOTER"
    if element.element_type == ElementType.HEADING:
        hierarchy = element.metadata.get("heading_hierarchy", {})
        if hierarchy.get("action") == "document_title":
            element_type = "TITLE"
        elif hierarchy.get("action") == "excluded_repeated_region":
            element_type = "HEADER" if repeated_region == "page_header" else "FOOTER"
        else:
            raw_level = element.metadata.get("parser_heading_level")
            raw_label = f" raw={raw_level}" if raw_level is not None else " raw=?"
            element_type = f"H{element.heading_level or '?'}{raw_label}"
    return f"{element.reading_order} {element_type} {element.element_id}"
