"""Deterministic visual-asset recovery for parser-neutral SoftDocs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from softdoc.ids import stable_digest
from softdoc.models import ContentAvailability, Document, Element, ElementType


VISUAL_ELEMENT_TYPES = frozenset(
    {
        ElementType.TABLE,
        ElementType.FIGURE,
        ElementType.CHART,
        ElementType.EQUATION,
    }
)


@dataclass(frozen=True)
class VisualAssetRecoveryResult:
    recovered_element_ids: tuple[str, ...]
    unavailable_element_ids: tuple[str, ...]


def recover_visual_element_assets(
    document: Document,
    output_dir: Path,
) -> VisualAssetRecoveryResult:
    """Ensure every visual or otherwise empty Element has a recoverable asset.

    A parser-provided element image remains preferred.  If it is absent or its
    file is missing, the Element bbox is cropped from the retained page image.
    Textual Elements with usable text/HTML deliberately do not receive duplicate
    crops.  Empty parser blocks do: otherwise their bbox is preserved but the
    content itself is not directly recoverable from the Element.
    """

    output_dir = Path(output_dir)
    pages = {page.page_id: page for page in document.pages}
    recovered: list[str] = []
    unavailable: list[str] = []
    for element in document.elements:
        if not _requires_visual_asset(element):
            continue
        if _element_has_valid_visual_asset(element, output_dir):
            continue
        page = pages[element.page_id]
        destination = crop_page_bbox(
            output_dir=output_dir,
            page_image=page.image_path,
            normalized_bbox=(element.bbox.normalized if element.bbox else None),
            owner_id=element.element_id,
        )
        recovery = {
            "source": "page_image_bbox",
            "page_id": page.page_id,
            "page_image_path": (
                page.image_path.as_posix() if page.image_path else None
            ),
            "bbox_id": element.bbox.bbox_id if element.bbox else None,
        }
        if destination is None:
            recovery["status"] = "unavailable"
            unavailable.append(element.element_id)
        else:
            recovery["status"] = "recovered"
            recovery["crop_image_path"] = destination.as_posix()
            element.crop_image_path = destination
            element.content_availability = (
                ContentAvailability.MIXED
                if bool((element.text or "").strip() or (element.html or "").strip())
                else ContentAvailability.VISUAL_ONLY
            )
            recovered.append(element.element_id)
        element.metadata["visual_asset_recovery"] = recovery
    return VisualAssetRecoveryResult(
        recovered_element_ids=tuple(recovered),
        unavailable_element_ids=tuple(unavailable),
    )


def crop_page_bbox(
    *,
    output_dir: Path,
    page_image: Path | None,
    normalized_bbox: tuple[float, float, float, float] | None,
    owner_id: str,
) -> Path | None:
    """Crop a normalized bbox from a retained page image."""

    if page_image is None or normalized_bbox is None:
        return None
    source = page_image if page_image.is_absolute() else output_dir / page_image
    if not source.is_file():
        return None
    destination = (
        output_dir
        / "assets"
        / "elements"
        / f"{stable_digest(f'{owner_id}:fallback-crop')}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as image:
            width, height = image.size
            x1, y1, x2, y2 = normalized_bbox
            left = max(0, min(width - 1, int(round(x1 * width))))
            top = max(0, min(height - 1, int(round(y1 * height))))
            right = max(left + 1, min(width, int(round(x2 * width))))
            bottom = max(top + 1, min(height, int(round(y2 * height))))
            image.crop((left, top, right, bottom)).save(destination)
    except Exception:
        return None
    return destination.relative_to(output_dir)


def _element_has_valid_visual_asset(
    element: Element,
    output_dir: Path,
) -> bool:
    for value in (element.image_path, element.crop_image_path):
        if value is None:
            continue
        path = value if value.is_absolute() else output_dir / value
        if path.is_file():
            return True
    return False


def _requires_visual_asset(element: Element) -> bool:
    if element.element_type in VISUAL_ELEMENT_TYPES:
        return True
    has_textual_content = bool(
        (element.text or "").strip() or (element.html or "").strip()
    )
    return not has_textual_content
