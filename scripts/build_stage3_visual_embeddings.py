"""Build a resumable, question-independent visual Element embedding index."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

from softdoc.retrieval.visual_dense import (
    VISUAL_INDEX_SCHEMA_VERSION,
    collect_visual_asset_inventory,
    inventory_fingerprint,
    unique_images,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOFTDOCS = ROOT / "data" / "processed" / "mmlongbench_doc" / "softdocs"
DEFAULT_OUTPUT = ROOT / ".runlogs" / "stage3_visual_retrieval" / "colsmol-500m"
DEFAULT_MODEL = "vidore/colSmol-500M"
BUILD_VERSION = "stage3-visual-embedding-build-v0.1"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--softdocs-root", type=Path, default=DEFAULT_SOFTDOCS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch-size and shard-size must be positive")

    softdocs_root = args.softdocs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(exist_ok=True)

    records, skipped = collect_visual_asset_inventory(softdocs_root)
    images = unique_images(records, softdocs_root)
    fingerprint = inventory_fingerprint(records)
    config = {
        "schema_version": VISUAL_INDEX_SCHEMA_VERSION,
        "build_version": BUILD_VERSION,
        "model": args.model,
        "selection": {
            "figures_and_charts": "all elements with a decodable visual asset",
            "tables": "only visual-only tables without text or HTML",
            "question_or_gold_conditioning": False,
            "content_hash_deduplication": True,
        },
        "embedding_storage_dtype": "float16",
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "softdocs_root": str(softdocs_root),
        "document_count": len({record.document_id for record in records}),
        "element_count": len(records),
        "unique_image_count": len(images),
        "skipped_count": len(skipped),
        "inventory_fingerprint": fingerprint,
    }
    _validate_or_write_config(output_dir / "config.json", config)
    _write_jsonl_atomic(output_dir / "assets.jsonl", [item.as_dict() for item in records])
    _write_jsonl_atomic(output_dir / "skipped_assets.jsonl", [item.as_dict() for item in skipped])

    completed = _completed_hashes(output_dir / "shards")
    pending = [(digest, path) for digest, path in images if digest not in completed]
    state = {
        "schema_version": VISUAL_INDEX_SCHEMA_VERSION,
        "build_version": BUILD_VERSION,
        "state": "inventory_ready" if args.inventory_only else "running",
        "updated_at": _now(),
        "inventory_fingerprint": fingerprint,
        "element_count": len(records),
        "unique_image_count": len(images),
        "completed_image_count": len(completed),
        "pending_image_count": len(pending),
        "skipped_count": len(skipped),
    }
    _write_json(output_dir / "state.json", state)
    print(
        f"inventory documents={config['document_count']} elements={len(records)} "
        f"unique_images={len(images)} skipped={len(skipped)} pending={len(pending)}",
        flush=True,
    )
    if args.inventory_only:
        return 0
    if not pending:
        state.update({"state": "completed", "completed_at": _now()})
        _write_json(output_dir / "state.json", state)
        print(output_dir, flush=True)
        return 0

    started = time.perf_counter()
    try:
        from sentence_transformers import MultiVectorEncoder

        model = MultiVectorEncoder(args.model, device=args.device)
        shard_index = _next_shard_index(output_dir / "shards")
        for start in range(0, len(pending), args.shard_size):
            shard_items = pending[start : start + args.shard_size]
            shard_arrays: dict[str, np.ndarray] = {}
            for batch_start in range(0, len(shard_items), args.batch_size):
                batch = shard_items[batch_start : batch_start + args.batch_size]
                encoded = model.encode_document(
                    [str(path) for _, path in batch],
                    batch_size=args.batch_size,
                    show_progress_bar=False,
                )
                vectors = _embedding_list(encoded, expected=len(batch))
                for (digest, _), vector in zip(batch, vectors, strict=True):
                    shard_arrays[digest] = vector.astype(np.float16, copy=False)
            shard_name = f"shard-{shard_index:05d}"
            _write_npz_atomic(output_dir / "shards" / f"{shard_name}.npz", shard_arrays)
            _write_json(
                output_dir / "shards" / f"{shard_name}.json",
                {
                    "schema_version": VISUAL_INDEX_SCHEMA_VERSION,
                    "model": args.model,
                    "image_hashes": list(shard_arrays),
                    "embedding_shapes": {
                        key: list(value.shape) for key, value in shard_arrays.items()
                    },
                    "storage_dtype": "float16",
                    "created_at": _now(),
                },
            )
            completed.update(shard_arrays)
            state.update(
                {
                    "updated_at": _now(),
                    "completed_image_count": len(completed),
                    "pending_image_count": len(images) - len(completed),
                    "last_completed_shard": shard_name,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            _write_json(output_dir / "state.json", state)
            print(
                f"{shard_name}: completed={len(completed)}/{len(images)}",
                flush=True,
            )
            shard_index += 1
        state.update(
            {
                "state": "completed",
                "completed_at": _now(),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        _write_json(output_dir / "state.json", state)
        print(output_dir, flush=True)
        return 0
    except Exception as exc:
        state.update(
            {
                "state": "failed",
                "updated_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(output_dir / "state.json", state)
        raise


def _embedding_list(value: Any, *, expected: int) -> list[np.ndarray]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray) and value.dtype != object:
        if expected == 1 and value.ndim == 2:
            result = [value]
        elif value.ndim >= 3 and len(value) == expected:
            result = [value[index] for index in range(expected)]
        else:
            raise ValueError(f"Unexpected embedding array shape: {value.shape}")
    else:
        result = []
        for item in list(value):
            if hasattr(item, "detach"):
                # NumPy cannot represent torch.bfloat16 directly.  Convert on
                # the CPU before the stable float16 shard-storage cast below.
                item = item.detach().float().cpu().numpy()
            result.append(np.asarray(item))
    if len(result) != expected:
        raise ValueError(f"Expected {expected} embeddings, received {len(result)}")
    if any(item.ndim != 2 or item.shape[0] < 1 or item.shape[1] < 1 for item in result):
        raise ValueError("Visual embeddings must be non-empty two-dimensional arrays")
    return result


def _completed_hashes(shards_dir: Path) -> set[str]:
    completed: set[str] = set()
    for metadata_path in sorted(shards_dir.glob("shard-*.json")):
        npz_path = metadata_path.with_suffix(".npz")
        if not npz_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != VISUAL_INDEX_SCHEMA_VERSION:
            raise RuntimeError(f"Incompatible shard: {metadata_path}")
        completed.update(str(value) for value in metadata.get("image_hashes", []))
    return completed


def _next_shard_index(shards_dir: Path) -> int:
    indexes = [int(path.stem.split("-")[-1]) for path in shards_dir.glob("shard-*.json")]
    return max(indexes, default=-1) + 1


def _validate_or_write_config(path: Path, config: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_keys = (
            "schema_version",
            "build_version",
            "model",
            "selection",
            "inventory_fingerprint",
        )
        if any(existing.get(key) != config.get(key) for key in stable_keys):
            raise RuntimeError("Existing visual index configuration does not match")
        return
    _write_json(path, config)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
