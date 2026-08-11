"""Build BM25 and multilingual-E5 indexes for one or more SoftDoc roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from softdoc.retrieval import (
    BM25Index,
    DenseConfig,
    DenseIndex,
    FileEmbeddingCache,
    HuggingFaceE5Encoder,
    SearchUnitBuilder,
)
from softdoc.serialization import load_document


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOFTDOC_ROOT = (
    ROOT / "data" / "processed" / "representative_28" / "softdoc"
)
DEFAULT_MODEL_DIR = (
    ROOT
    / "data"
    / "cache"
    / "huggingface"
    / "intfloat--multilingual-e5-small"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "data" / "processed" / "representative_28" / "retrieval"
)
MODEL_NAME = "intfloat/multilingual-e5-small"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--softdoc-root",
        type=Path,
        action="append",
        dest="softdoc_roots",
        help=(
            "Directory containing serialized SoftDocs. Repeat this option to "
            "build one combined corpus from multiple roots."
        ),
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    softdoc_roots = [
        path.resolve()
        for path in (args.softdoc_roots or [DEFAULT_SOFTDOC_ROOT])
    ]
    model_dir = args.model_dir.resolve()
    output_root = args.output_root.resolve()
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"E5 model is missing from {model_dir}")
    document_dirs = _discover_document_dirs(softdoc_roots)

    output_root.mkdir(parents=True, exist_ok=True)
    revision = _downloaded_revision(model_dir) or "main"
    encoder_started = time.perf_counter()
    encoder = HuggingFaceE5Encoder(
        model_name=MODEL_NAME,
        model_path=model_dir,
        model_revision=revision,
        tokenizer_revision=revision,
        device=args.device,
        local_files_only=True,
    )
    encoder_seconds = time.perf_counter() - encoder_started
    cache = FileEmbeddingCache(output_root / "embedding_cache")
    config = DenseConfig(batch_size=args.batch_size)

    rows: list[dict[str, object]] = []
    run_started = time.perf_counter()
    for document_dir in document_dirs:
        started = time.perf_counter()
        stage_started = time.perf_counter()
        document = load_document(document_dir)
        load_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        build_result = SearchUnitBuilder().build(document)
        search_unit_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        BM25Index(build_result)
        bm25_build_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        index = DenseIndex(
            build_result,
            encoder,
            config=config,
            cache=cache,
        )
        dense_index_seconds = time.perf_counter() - stage_started
        document_output = output_root / "documents" / document_dir.name
        document_output.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            document_output / "dense_segments.jsonl",
            [segment.model_dump(mode="json") for segment in index.segments],
        )
        split_units = {
            segment.search_unit_id
            for segment in index.segments
            if segment.segment_count > 1
        }
        token_counts = [segment.model_token_count for segment in index.segments]
        elapsed = time.perf_counter() - started
        row: dict[str, object] = {
            "document": document_dir.name,
            "document_id": document.document_id,
            "title": document.title,
            "pages": len(document.pages),
            "elements": len(document.elements),
            "search_units": len(build_result.units),
            "skipped_elements": len(build_result.skipped_elements),
            "dense_segments": len(index.segments),
            "split_search_units": len(split_units),
            "additional_segments": len(index.segments) - len(build_result.units),
            "max_model_tokens": max(token_counts, default=0),
            "p50_model_tokens": _percentile(token_counts, 0.50),
            "p95_model_tokens": _percentile(token_counts, 0.95),
            "cache_hits": index.cache_hits,
            "cache_misses": index.cache_misses,
            "cache_writes": index.cache_writes,
            "load_seconds": round(load_seconds, 6),
            "search_unit_seconds": round(search_unit_seconds, 6),
            "bm25_build_seconds": round(bm25_build_seconds, 6),
            "dense_index_seconds": round(dense_index_seconds, 6),
            "elapsed_seconds": round(elapsed, 3),
        }
        rows.append(row)
        (document_output / "dense_manifest.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            f"{document_dir.name}: units={len(build_result.units)} "
            f"segments={len(index.segments)} split={len(split_units)} "
            f"cache={index.cache_hits}/{index.cache_misses} "
            f"seconds={elapsed:.2f}",
            flush=True,
        )

    total_seconds = time.perf_counter() - run_started
    summary = {
        "model": encoder.fingerprint.model_dump(mode="json"),
        "device": encoder.device,
        "model_directory": str(model_dir),
        "model_disk_bytes": sum(
            path.stat().st_size for path in model_dir.rglob("*") if path.is_file()
        ),
        "encoder_load_seconds": round(encoder_seconds, 3),
        "softdoc_roots": [str(path) for path in softdoc_roots],
        "documents": len(rows),
        "total_pages": sum(int(row["pages"]) for row in rows),
        "total_elements": sum(int(row["elements"]) for row in rows),
        "total_search_units": sum(int(row["search_units"]) for row in rows),
        "total_dense_segments": sum(int(row["dense_segments"]) for row in rows),
        "total_split_search_units": sum(
            int(row["split_search_units"]) for row in rows
        ),
        "total_additional_segments": sum(
            int(row["additional_segments"]) for row in rows
        ),
        "total_cache_hits": sum(int(row["cache_hits"]) for row in rows),
        "total_cache_misses": sum(int(row["cache_misses"]) for row in rows),
        "run_kind": (
            "cache_reuse"
            if sum(int(row["cache_misses"]) for row in rows) == 0
            else "index_build_or_update"
        ),
        "total_seconds": round(total_seconds, 3),
        "average_seconds_per_document": round(total_seconds / len(rows), 3)
        if rows
        else 0.0,
        "stage_totals_seconds": {
            "softdoc_load": round(sum(float(row["load_seconds"]) for row in rows), 3),
            "search_unit_build": round(
                sum(float(row["search_unit_seconds"]) for row in rows), 3
            ),
            "bm25_build": round(
                sum(float(row["bm25_build_seconds"]) for row in rows), 3
            ),
            "dense_index": round(
                sum(float(row["dense_index_seconds"]) for row in rows), 3
            ),
        },
        "documents_detail": rows,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "run_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _discover_document_dirs(roots: list[Path]) -> list[Path]:
    discovered: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"SoftDoc root does not exist: {root}")
        for path in sorted(root.iterdir()):
            if not path.is_dir() or not (path / "document.json").is_file():
                continue
            if path.name in discovered:
                raise RuntimeError(
                    f"Duplicate SoftDoc directory name {path.name!r}: "
                    f"{discovered[path.name]} and {path}"
                )
            discovered[path.name] = path
    if not discovered:
        raise RuntimeError("No serialized SoftDocs were found")
    return [discovered[name] for name in sorted(discovered)]


def _downloaded_revision(model_dir: Path) -> str | None:
    metadata = (
        model_dir
        / ".cache"
        / "huggingface"
        / "download"
        / "config.json.metadata"
    )
    if not metadata.is_file():
        return None
    first_line = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
    return first_line or None


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Representative Dense Retrieval Index Run",
        "",
        f"- Documents: {summary['documents']}",
        f"- Pages: {summary['total_pages']}",
        f"- Elements: {summary['total_elements']}",
        f"- SearchUnits: {summary['total_search_units']}",
        f"- DenseSegments: {summary['total_dense_segments']}",
        f"- Split SearchUnits: {summary['total_split_search_units']}",
        f"- Additional segments: {summary['total_additional_segments']}",
        f"- Device: {summary['device']}",
        f"- Run kind: {summary['run_kind']}",
        f"- Total seconds: {summary['total_seconds']}",
        f"- Average seconds/document: {summary['average_seconds_per_document']}",
        "",
        "| Document | Units | Segments | Split units | Max tokens | Seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["documents_detail"]:  # type: ignore[union-attr]
        lines.append(
            f"| {row['document']} | {row['search_units']} | "
            f"{row['dense_segments']} | {row['split_search_units']} | "
            f"{row['max_model_tokens']} | {row['elapsed_seconds']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
