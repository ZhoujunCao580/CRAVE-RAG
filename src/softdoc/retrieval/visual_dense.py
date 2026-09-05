"""Visual-element inventory, runtime search, and fusion experiment helpers.

The index is question-independent and bound to real SoftDoc Elements. Runtime
search emits compact visual candidate metadata; Controller-facing projection
still happens through the ordinary SearchSession CandidatePreview contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
from PIL import Image

from softdoc.ids import stable_digest
from softdoc.models import (
    ContentAvailability,
    Document,
    Element,
    ElementType,
    RelationStatus,
    RelationType,
)
from softdoc.retrieval.models import (
    SubQuestionInput,
    VisualElementCandidate,
    VisualSearchResult,
)
from softdoc.serialization import load_document
from softdoc.visual_retrieval import visual_retrieval_descriptor


VISUAL_INDEX_SCHEMA_VERSION = "softdoc-visual-multivector-v0.1"
VISUAL_ELEMENT_TYPES = frozenset({ElementType.FIGURE, ElementType.CHART})


class MultiVectorSearchModel(Protocol):
    def encode_query(self, sentences: list[str], **kwargs: Any) -> Any: ...

    def similarity(self, queries: list[Any], documents: list[Any], **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class VisualAssetRecord:
    """One retrievable SoftDoc Element backed by a decodable image."""

    visual_asset_id: str
    document_id: str
    element_id: str
    element_type: str
    page_id: str
    page_number: int
    softdoc_relpath: str
    asset_relpath: str
    image_sha256: str
    width: int
    height: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "visual_asset_id": self.visual_asset_id,
            "document_id": self.document_id,
            "element_id": self.element_id,
            "element_type": self.element_type,
            "page_id": self.page_id,
            "page_number": self.page_number,
            "softdoc_relpath": self.softdoc_relpath,
            "asset_relpath": self.asset_relpath,
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class SkippedVisualAsset:
    document_id: str
    element_id: str
    reason: str
    asset_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "element_id": self.element_id,
            "reason": self.reason,
            "asset_path": self.asset_path,
        }


class VisualDenseIndex:
    """Query one completed visual embedding index for a single SoftDoc."""

    def __init__(
        self,
        document: Document,
        index_dir: Path,
        *,
        model_name: str | None = None,
        device: str = "cuda",
        similarity_chunk_elements: int = 16_000_000,
        model: MultiVectorSearchModel | None = None,
    ) -> None:
        self.document = document
        self.index_dir = Path(index_dir).resolve()
        self.device = device
        self.similarity_chunk_elements = similarity_chunk_elements
        if similarity_chunk_elements < 1:
            raise ValueError("Visual similarity chunk size must be positive")

        state = _read_json(self.index_dir / "state.json")
        config = _read_json(self.index_dir / "config.json")
        if (
            state.get("state") != "completed"
            or int(state.get("pending_image_count", -1)) != 0
        ):
            raise RuntimeError("Visual embedding index is not complete")
        if state.get("schema_version") != VISUAL_INDEX_SCHEMA_VERSION:
            raise RuntimeError("Visual embedding index schema is incompatible")
        if state.get("inventory_fingerprint") != config.get("inventory_fingerprint"):
            raise RuntimeError("Visual embedding index fingerprint is inconsistent")
        configured_model = str(config.get("model") or "").strip()
        if not configured_model:
            raise RuntimeError("Visual embedding index has no model binding")
        if model_name is not None and model_name != configured_model:
            raise RuntimeError("Requested visual model does not match the index")
        self.model_name = configured_model
        self.index_fingerprint = str(config["inventory_fingerprint"])

        elements_by_id = {item.element_id: item for item in document.elements}
        raw_assets = [
            row
            for row in _read_jsonl(self.index_dir / "assets.jsonl")
            if row.get("document_id") == document.document_id
        ]
        asset_ids = [str(row.get("element_id")) for row in raw_assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise RuntimeError("Visual index contains duplicate Element mappings")
        self._assets: list[dict[str, Any]] = []
        for row in raw_assets:
            element_id = str(row.get("element_id"))
            element = elements_by_id.get(element_id)
            if element is None:
                raise RuntimeError(
                    f"Visual index refers to a missing Element: {element_id}"
                )
            if (
                row.get("page_id") != element.page_id
                or int(row.get("page_number", -1)) != element.page_number
                or row.get("element_type") != element.element_type.value
            ):
                raise RuntimeError(
                    f"Visual index metadata drifted for Element: {element_id}"
                )
            self._assets.append(row)
        required_hashes = {str(row["image_sha256"]) for row in self._assets}
        self._vectors = load_visual_vectors(
            self.index_dir / "shards", required_hashes
        )
        self._elements_by_id = elements_by_id
        self._captions_by_target = _confirmed_captions(document)
        if model is None:
            from sentence_transformers import MultiVectorEncoder

            model = MultiVectorEncoder(self.model_name, device=device)
        self._model = model

    def search(self, subquestion: SubQuestionInput) -> VisualSearchResult:
        if not self._assets:
            return VisualSearchResult(
                subquestion_id=subquestion.subquestion_id,
                document_id=self.document.document_id,
                index_fingerprint=self.index_fingerprint,
                model_name=self.model_name,
                total_visual_assets=0,
                total_candidates=0,
                candidates=[],
            )
        encoded = self._model.encode_query(
            [subquestion.text], batch_size=1, show_progress_bar=False
        )
        query = _one_query_embedding(encoded)
        if hasattr(query, "half"):
            query = query.half()
        unique_hashes = sorted({str(row["image_sha256"]) for row in self._assets})
        scores = self._model.similarity(
            [query],
            [self._vectors[digest] for digest in unique_hashes],
            device=self.device,
            chunk_elements=self.similarity_chunk_elements,
        )
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        matrix = np.asarray(scores)
        if matrix.ndim == 1:
            row_scores = matrix
        elif matrix.ndim == 2 and matrix.shape[0] == 1:
            row_scores = matrix[0]
        else:
            raise RuntimeError(
                f"Unexpected visual similarity matrix shape: {matrix.shape}"
            )
        if len(row_scores) != len(unique_hashes):
            raise RuntimeError("Visual similarity count does not match image count")
        score_by_hash = dict(zip(unique_hashes, row_scores.tolist(), strict=True))
        ranked_assets = sorted(
            self._assets,
            key=lambda row: (
                -float(score_by_hash[str(row["image_sha256"])]),
                int(row["page_number"]),
                str(row["element_id"]),
            ),
        )
        candidates = []
        for rank, row in enumerate(ranked_assets, start=1):
            element = self._elements_by_id[str(row["element_id"])]
            candidates.append(
                VisualElementCandidate(
                    element_id=element.element_id,
                    visual_asset_id=str(row["visual_asset_id"]),
                    visual_score=float(score_by_hash[str(row["image_sha256"])]),
                    visual_rank=rank,
                    page_id=element.page_id,
                    page_number=element.page_number,
                    section_id=element.section_id,
                    section_path=list(element.section_path or []),
                    display_label=element.reference_label or None,
                    preview_text=_visual_preview_text(
                        element,
                        self._captions_by_target.get(element.element_id, []),
                    ),
                    element_type=element.element_type,
                    content_availability=(
                        element.content_availability
                        or ContentAvailability.UNAVAILABLE
                    ),
                )
            )
        return VisualSearchResult(
            subquestion_id=subquestion.subquestion_id,
            document_id=self.document.document_id,
            index_fingerprint=self.index_fingerprint,
            model_name=self.model_name,
            total_visual_assets=len(self._assets),
            total_candidates=len(candidates),
            candidates=candidates,
        )


def collect_visual_asset_inventory(
    softdocs_root: Path,
) -> tuple[list[VisualAssetRecord], list[SkippedVisualAsset]]:
    """Collect all question-independent visual retrieval candidates.

    Figures, charts, and tables are considered because structured table text
    can be incomplete or lose visual grouping even when HTML is present.  The
    collector still requires a real, decodable image before adding a record.
    Identical image bytes are embedded once later, while every Element mapping
    remains in the inventory.
    """

    root = softdocs_root.resolve()
    records: list[VisualAssetRecord] = []
    skipped: list[SkippedVisualAsset] = []
    for document_json in sorted(root.glob("*/document.json"), key=lambda p: p.as_posix()):
        softdoc_dir = document_json.parent
        document = load_document(softdoc_dir)
        softdoc_relpath = softdoc_dir.relative_to(root).as_posix()
        for element in sorted(
            document.elements,
            key=lambda item: (item.page_number, item.reading_order, item.element_id),
        ):
            if not is_visual_retrieval_candidate(element):
                continue
            raw_asset = element.visual_asset_path
            if raw_asset is None:
                skipped.append(
                    SkippedVisualAsset(
                        document_id=document.document_id,
                        element_id=element.element_id,
                        reason="missing_visual_asset_reference",
                    )
                )
                continue
            asset_path = resolve_softdoc_asset(softdoc_dir, raw_asset)
            if not asset_path.is_file():
                skipped.append(
                    SkippedVisualAsset(
                        document_id=document.document_id,
                        element_id=element.element_id,
                        reason="missing_visual_asset_file",
                        asset_path=str(asset_path),
                    )
                )
                continue
            try:
                with Image.open(asset_path) as image:
                    image.verify()
                with Image.open(asset_path) as image:
                    width, height = image.size
                if width < 2 or height < 2:
                    raise ValueError("image dimensions are smaller than 2x2")
            except Exception as exc:
                skipped.append(
                    SkippedVisualAsset(
                        document_id=document.document_id,
                        element_id=element.element_id,
                        reason=f"undecodable_visual_asset:{type(exc).__name__}",
                        asset_path=str(asset_path),
                    )
                )
                continue
            image_sha256 = sha256_file(asset_path)
            try:
                asset_relpath = asset_path.relative_to(softdoc_dir).as_posix()
            except ValueError:
                asset_relpath = str(asset_path)
            records.append(
                VisualAssetRecord(
                    visual_asset_id="visual-asset:"
                    + stable_digest(document.document_id, element.element_id, image_sha256),
                    document_id=document.document_id,
                    element_id=element.element_id,
                    element_type=element.element_type.value,
                    page_id=element.page_id,
                    page_number=element.page_number,
                    softdoc_relpath=softdoc_relpath,
                    asset_relpath=asset_relpath,
                    image_sha256=image_sha256,
                    width=width,
                    height=height,
                )
            )
    return records, skipped


def is_visual_retrieval_candidate(element: Element) -> bool:
    if element.element_type in VISUAL_ELEMENT_TYPES:
        return True
    # HTML is not proof that the table is fully searchable. OCR may concatenate
    # labels or omit spatial grouping that remains visible in the image.
    # Asset existence/decodability is checked by collect_visual_asset_inventory.
    return element.element_type == ElementType.TABLE


def resolve_softdoc_asset(softdoc_dir: Path, raw_path: Path | str) -> Path:
    """Resolve Windows-authored SoftDoc paths on Windows or Linux."""

    value = str(raw_path).replace("\\", "/")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return softdoc_dir / Path(PurePosixPath(value))


def unique_images(
    records: Sequence[VisualAssetRecord],
    softdocs_root: Path,
) -> list[tuple[str, Path]]:
    """Return deterministic content-deduplicated image inputs."""

    root = softdocs_root.resolve()
    first_by_hash: dict[str, Path] = {}
    for record in records:
        path = resolve_softdoc_asset(
            root / Path(PurePosixPath(record.softdoc_relpath)),
            record.asset_relpath,
        )
        first_by_hash.setdefault(record.image_sha256, path)
    return sorted(first_by_hash.items())


def fixed_text_visual_quota_ranking(
    text_ranking: Sequence[Sequence[Any]],
    visual_ranking: Sequence[Sequence[Any]],
    *,
    batch_size: int = 5,
    text_quota: int = 3,
    visual_quota: int = 2,
    excluded_ids: Iterable[str] = (),
) -> list[list[Any]]:
    """Build deterministic 3-text + 2-visual batches with unique backfill."""

    if batch_size < 1 or text_quota < 0 or visual_quota < 0:
        raise ValueError("Batch size and quotas must be non-negative")
    if text_quota + visual_quota != batch_size:
        raise ValueError("Text and visual quotas must sum to batch_size")
    streams = (list(text_ranking), list(visual_ranking))
    quotas = (text_quota, visual_quota)
    cursors = [0, 0]
    result: list[list[Any]] = []
    seen = set(excluded_ids)

    def take(stream_index: int, count: int) -> int:
        added = 0
        while cursors[stream_index] < len(streams[stream_index]) and added < count:
            raw = streams[stream_index][cursors[stream_index]]
            cursors[stream_index] += 1
            entry = list(raw)
            element_id = str(entry[0])
            if element_id in seen:
                continue
            seen.add(element_id)
            result.append(entry)
            added += 1
        return added

    while True:
        batch_start = len(result)
        for stream_index, quota in enumerate(quotas):
            take(stream_index, quota)
        while len(result) - batch_start < batch_size:
            before = len(result)
            for stream_index in range(len(streams)):
                take(stream_index, 1)
                if len(result) - batch_start >= batch_size:
                    break
            if len(result) == before:
                break
        if len(result) == batch_start:
            break
    return result


def weighted_rrf_three_route_ranking(
    bm25_ranking: Sequence[Sequence[Any]],
    dense_ranking: Sequence[Sequence[Any]],
    visual_ranking: Sequence[Sequence[Any]],
    *,
    rrf_k: int = 20,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.25,
    visual_weight: float = 1.0,
    excluded_ids: Iterable[str] = (),
) -> list[list[Any]]:
    """Fuse BM25, text Dense, and visual Dense without mixing raw scores."""

    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    weights = (bm25_weight, dense_weight, visual_weight)
    if any(weight <= 0 for weight in weights):
        raise ValueError("RRF weights must be positive")
    excluded = set(excluded_ids)
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    page_number: dict[str, int] = {}
    for ranking, weight in zip(
        (bm25_ranking, dense_ranking, visual_ranking), weights, strict=True
    ):
        for rank, raw in enumerate(ranking, start=1):
            entry = list(raw)
            element_id = str(entry[0])
            if element_id in excluded:
                continue
            scores[element_id] = scores.get(element_id, 0.0) + weight / (rrf_k + rank)
            best_rank[element_id] = min(best_rank.get(element_id, rank), rank)
            page_number[element_id] = int(entry[1])
    ordered = sorted(
        scores,
        key=lambda element_id: (
            -scores[element_id],
            best_rank[element_id],
            page_number[element_id],
            element_id,
        ),
    )
    return [[element_id, page_number[element_id]] for element_id in ordered]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_fingerprint(records: Sequence[VisualAssetRecord]) -> str:
    payload = [
        (
            record.document_id,
            record.element_id,
            record.page_id,
            record.image_sha256,
        )
        for record in records
    ]
    return stable_digest(VISUAL_INDEX_SCHEMA_VERSION, payload, length=32)


def load_visual_vectors(
    shards_dir: Path,
    required_hashes: set[str],
) -> dict[str, np.ndarray]:
    """Load only the image embeddings needed by one document."""

    result: dict[str, np.ndarray] = {}
    for path in sorted(Path(shards_dir).glob("shard-*.npz")):
        with np.load(path) as shard:
            for digest in shard.files:
                if digest in required_hashes:
                    result[digest] = shard[digest]
        if required_hashes.issubset(result):
            break
    missing = required_hashes - result.keys()
    if missing:
        raise RuntimeError(f"Missing {len(missing)} visual embeddings")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing visual index file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Visual index JSON must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing visual index file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Visual index row must be an object: {path}:{line_number}"
                )
            rows.append(payload)
    return rows


def _one_query_embedding(value: Any) -> Any:
    if hasattr(value, "detach"):
        if len(value) != 1:
            raise RuntimeError("Visual encoder did not return exactly one query")
        return value[0]
    values = list(value)
    if len(values) != 1:
        raise RuntimeError("Visual encoder did not return exactly one query")
    return values[0]


def _confirmed_captions(document: Document) -> dict[str, list[str]]:
    elements_by_id = {item.element_id: item for item in document.elements}
    result: dict[str, list[str]] = {}
    for relation in document.relations:
        if (
            relation.relation_type != RelationType.CAPTION_OF
            or relation.status != RelationStatus.CONFIRMED
        ):
            continue
        caption = elements_by_id.get(relation.source_id)
        target = elements_by_id.get(relation.target_id)
        text = (caption.text or "").strip() if caption is not None else ""
        if target is None or caption is None or not text:
            continue
        result.setdefault(target.element_id, []).append(" ".join(text.split()))
    return result


def _visual_preview_text(element: Element, captions: Sequence[str]) -> str:
    descriptor = visual_retrieval_descriptor(element)
    if descriptor is not None:
        # The descriptor already received the trusted label/caption context when
        # it was generated.  Keep the Controller-facing preview compact instead
        # of repeating that context and the retrieval-only keyword list.
        return " ".join(descriptor.search_summary.split())

    parts: list[str] = []
    if element.reference_label:
        parts.append(" ".join(element.reference_label.split()))
    parts.extend(" ".join(text.split()) for text in captions if text.strip())
    if element.element_type == ElementType.TABLE and element.html:
        # A visually retrieved table still needs a lightweight semantic cue for
        # the Controller. Reuse parser text as a preview, while the Reader must
        # inspect the original image before producing Evidence.
        from softdoc.retrieval.units import html_to_text

        table_text = html_to_text(element.html)
        if table_text:
            parts.append(table_text)
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(part)
    if unique:
        return " | ".join(unique)
    return f"{element.element_type.value} candidate matched from visual content"
