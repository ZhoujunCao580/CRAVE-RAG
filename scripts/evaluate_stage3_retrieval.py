"""Run the resumable stage-3B default retrieval comparison.

The two policies share the same Exact, BM25, and Dense results.  Only candidate
fusion differs: frozen weighted RRF versus deterministic fixed 3+2 quota.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from softdoc.external_data import ExternalDatasetManifest
from softdoc.ids import stable_digest
from softdoc.retrieval import (
    BM25Index,
    CandidateMergePolicy,
    DenseConfig,
    DenseIndex,
    ExactAnchorLookup,
    FileEmbeddingCache,
    HuggingFaceE5Encoder,
    SearchSessionBuilder,
    SearchSessionConfig,
    SearchUnitBuilder,
    SubQuestionInput,
)
from softdoc.serialization import load_document


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REACHABILITY_ROOT = ROOT / ".runlogs" / "stage3_softdoc_audit" / "runs"
DEFAULT_OUTPUT_ROOT = ROOT / ".runlogs" / "stage3_retrieval" / "runs"
DEFAULT_CACHE = ROOT / ".runlogs" / "stage3_retrieval" / "embedding_cache"
DEFAULT_MODEL_DIR = (
    ROOT / "data" / "cache" / "huggingface" / "intfloat--multilingual-e5-small"
)
MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EVALUATION_VERSION = "stage3-retrieval-default-comparison-v0.2"
K_VALUES = (1, 3, 5, 10, 20, 50)
POLICY_NAMES = ("weighted_rrf", "fixed_quota_3_2")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reachability-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dense-batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    reachability_run = (
        args.reachability_run.resolve()
        if args.reachability_run
        else _latest_completed_reachability_run(DEFAULT_REACHABILITY_ROOT)
    )
    reachability_summary = reachability_run / "gold_evidence_reachability_summary.json"
    reachability_rows = reachability_run / "gold_evidence_reachability.jsonl"
    external_manifest_path = reachability_run / "external_dataset_manifest.json"
    for path in (reachability_summary, reachability_rows, external_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_dir = args.model_dir.resolve()
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"E5 model is missing from {model_dir}")

    eligible_question_ids = _eligible_question_ids(reachability_rows)
    resolved_gold_pages = _resolved_gold_pages_by_question(reachability_rows)
    manifest = ExternalDatasetManifest.model_validate_json(
        external_manifest_path.read_text(encoding="utf-8")
    )
    eligible_questions = [
        item for item in manifest.questions if item.question_id in eligible_question_ids
    ]
    if len(eligible_questions) != len(eligible_question_ids):
        raise RuntimeError("Reachability and external manifest question IDs disagree")
    by_document: dict[str, list[Any]] = defaultdict(list)
    for question in eligible_questions:
        by_document[question.document_id].append(question)

    rrf_config = SearchSessionConfig(
        merge_policy=CandidateMergePolicy.WEIGHTED_RRF,
        batch_size=5,
        rrf_k=20,
        bm25_weight=1.0,
        dense_weight=1.25,
    )
    quota_config = SearchSessionConfig(
        merge_policy=CandidateMergePolicy.FIXED_QUOTA,
        batch_size=5,
        bm25_quota=3,
        dense_quota=2,
    )
    config_payload = {
        "evaluation_version": EVALUATION_VERSION,
        "reachability_run": str(reachability_run),
        "reachability_summary_sha256": _sha256(reachability_summary),
        "reachability_rows_sha256": _sha256(reachability_rows),
        "eligible_question_count": len(eligible_questions),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_dir": str(model_dir),
        "device_request": args.device,
        "dense_config": DenseConfig(batch_size=args.dense_batch_size).model_dump(
            mode="json"
        ),
        "weighted_rrf": rrf_config.model_dump(mode="json"),
        "fixed_quota_3_2": quota_config.model_dump(mode="json"),
        "winner_rule": [
            "higher exact_plus_candidates mean_gold_page_recall@5",
            "higher exact_plus_candidates complete_gold_page_hit_rate@5",
            "higher exact_plus_candidates hit_rate@5",
            "higher exact_plus_candidates mrr",
            "lower mean fusion latency",
        ],
        "exact_anchor_is_shared_upstream": True,
        "bm25_and_dense_results_are_shared": True,
        "visual_descriptions_are_searchunit_metadata": True,
        "fallback_enabled": False,
    }
    config_fingerprint = stable_digest(config_payload, length=32)

    if args.resume_run is not None:
        run_dir = args.resume_run.resolve()
        state_path = run_dir / "run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError("Resume run configuration fingerprint mismatch")
        if state.get("state") == "completed":
            print(run_dir)
            return 0
    else:
        run_id = args.run_id or (
            "default-compare-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
            raise ValueError("Invalid run-id")
        run_dir = args.output_root.resolve() / run_id
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
        (run_dir / "documents").mkdir(parents=True)
        state = {
            "evaluation_version": EVALUATION_VERSION,
            "run_id": run_id,
            "state": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": config_fingerprint,
            "completed_documents": [],
        }
        _write_json(run_dir / "config.json", config_payload)
        _write_json(run_dir / "run_state.json", state)

    encoder = HuggingFaceE5Encoder(
        model_name=MODEL_NAME,
        model_path=model_dir,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        device=args.device,
        local_files_only=True,
    )
    cache = FileEmbeddingCache(args.embedding_cache.resolve())
    exact_lookup = ExactAnchorLookup()
    dense_config = DenseConfig(batch_size=args.dense_batch_size)
    completed = set(state.get("completed_documents", []))
    document_manifests: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        for document_order, external_document in enumerate(manifest.documents):
            questions = by_document.get(external_document.document_id, [])
            if not questions:
                continue
            document_key = f"{document_order:03d}-{stable_digest(external_document.document_id)}"
            output_path = run_dir / "documents" / f"{document_key}.jsonl"
            manifest_path = run_dir / "documents" / f"{document_key}.json"
            if external_document.document_id in completed:
                document_manifests.append(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
                continue

            document_started = time.perf_counter()
            softdoc_dir = _resolve_manifest_path(manifest, external_document.softdoc_dir)
            document = load_document(softdoc_dir)
            build_started = time.perf_counter()
            search_units = SearchUnitBuilder().build(document)
            search_unit_ms = (time.perf_counter() - build_started) * 1000.0
            bm25_started = time.perf_counter()
            bm25_index = BM25Index(search_units)
            bm25_index_ms = (time.perf_counter() - bm25_started) * 1000.0
            dense_started = time.perf_counter()
            dense_index = DenseIndex(
                search_units,
                encoder,
                config=dense_config,
                cache=cache,
            )
            dense_index_ms = (time.perf_counter() - dense_started) * 1000.0

            rows: list[dict[str, Any]] = []
            for question in questions:
                subquestion = SubQuestionInput(
                    subquestion_id=question.question_id,
                    text=question.question,
                )
                timing: dict[str, float] = {}
                stage_started = time.perf_counter()
                exact = exact_lookup.lookup(subquestion, document)
                timing["exact"] = _elapsed_ms(stage_started)
                stage_started = time.perf_counter()
                bm25 = bm25_index.search(subquestion)
                timing["bm25"] = _elapsed_ms(stage_started)
                stage_started = time.perf_counter()
                dense = dense_index.search(subquestion)
                timing["dense"] = _elapsed_ms(stage_started)

                stage_started = time.perf_counter()
                rrf_session = SearchSessionBuilder(rrf_config).create(
                    subquestion=subquestion,
                    search_units=search_units,
                    exact=exact,
                    bm25=bm25,
                    dense=dense,
                )
                timing["weighted_rrf_fusion"] = _elapsed_ms(stage_started)
                stage_started = time.perf_counter()
                quota_session = SearchSessionBuilder(quota_config).create(
                    subquestion=subquestion,
                    search_units=search_units,
                    exact=exact,
                    bm25=bm25,
                    dense=dense,
                )
                timing["fixed_quota_fusion"] = _elapsed_ms(stage_started)

                exact_entries = _exact_entries(exact)
                bm25_entries = [
                    [item.element_id, item.page_number] for item in bm25.candidates
                ]
                dense_entries = [
                    [item.element_id, item.page_number] for item in dense.candidates
                ]
                gold_pages = resolved_gold_pages[question.question_id]
                rows.append(
                    {
                        "question_id": question.question_id,
                        "case_id": question.case_id,
                        "question_index": question.metadata.get("source_index"),
                        "document_id": question.document_id,
                        "softdoc_document_id": document.document_id,
                        "question": question.question,
                        "gold_pages": gold_pages,
                        "timing_ms": {key: round(value, 6) for key, value in timing.items()},
                        "source_overlap_top_50": len(
                            {item[0] for item in bm25_entries[:50]}
                            & {item[0] for item in dense_entries[:50]}
                        ),
                        "exact_entries": exact_entries,
                        "source_rankings": {
                            "bm25": bm25_entries,
                            "dense": dense_entries,
                        },
                        "policies": {
                            "weighted_rrf": _policy_result(
                                rrf_session, exact_entries, gold_pages
                            ),
                            "fixed_quota_3_2": _policy_result(
                                quota_session, exact_entries, gold_pages
                            ),
                        },
                    }
                )

            _write_jsonl_atomic(output_path, rows)
            document_manifest = {
                "document_id": external_document.document_id,
                "softdoc_document_id": document.document_id,
                "softdoc_dir": str(softdoc_dir),
                "questions": len(rows),
                "pages": len(document.pages),
                "elements": len(document.elements),
                "search_units": len(search_units.units),
                "dense_segments": len(dense_index.segments),
                "dense_cache_hits": dense_index.cache_hits,
                "dense_cache_misses": dense_index.cache_misses,
                "dense_cache_writes": dense_index.cache_writes,
                "search_unit_ms": round(search_unit_ms, 3),
                "bm25_index_ms": round(bm25_index_ms, 3),
                "dense_index_ms": round(dense_index_ms, 3),
                "elapsed_seconds": round(time.perf_counter() - document_started, 3),
                "result_file": output_path.name,
            }
            _write_json(manifest_path, document_manifest)
            document_manifests.append(document_manifest)
            completed.add(external_document.document_id)
            state["completed_documents"] = sorted(completed)
            state["last_completed_document"] = external_document.document_id
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(run_dir / "run_state.json", state)
            print(
                f"{external_document.document_id}: questions={len(rows)} "
                f"units={len(search_units.units)} "
                f"dense_cache={dense_index.cache_hits}/{dense_index.cache_misses}",
                flush=True,
            )

        all_rows = _load_document_rows(run_dir / "documents")
        if len(all_rows) != len(eligible_questions):
            raise RuntimeError(
                f"Expected {len(eligible_questions)} results, found {len(all_rows)}"
            )
        summary = _build_summary(
            all_rows,
            document_manifests,
            device=encoder.device,
            elapsed_seconds=time.perf_counter() - started,
            config_fingerprint=config_fingerprint,
        )
        _write_jsonl_atomic(run_dir / "retrieval_results.jsonl", all_rows)
        _write_json(run_dir / "retrieval_comparison_summary.json", summary)
        _write_json(run_dir / "winner.json", summary["winner"])
        (run_dir / "retrieval_comparison.md").write_text(
            _summary_markdown(summary), encoding="utf-8"
        )
        state.update(
            {
                "state": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "result_rows": len(all_rows),
                "winner": summary["winner"]["policy"],
            }
        )
        _write_json(run_dir / "run_state.json", state)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(run_dir, flush=True)
        return 0
    except Exception as exc:
        state.update(
            {
                "state": "failed",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(run_dir / "run_state.json", state)
        raise


def _policy_result(session: Any, exact_entries: list[list[Any]], gold_pages: list[int]) -> dict[str, Any]:
    candidates = [
        [candidate.element_id, candidate.page_number]
        for candidate in session.candidate_catalog
    ]
    return {
        "candidate_ranking": candidates,
        "candidate_count": len(candidates),
        "duplicate_candidate_count": len(candidates)
        - len({item[0] for item in candidates}),
        "metrics": _ranking_metrics(
            candidate_entries=candidates,
            exact_entries=exact_entries,
            gold_pages=gold_pages,
            batch_size=session.config.batch_size,
        ),
    }


def _ranking_metrics(
    *,
    candidate_entries: list[list[Any]],
    exact_entries: list[list[Any]],
    gold_pages: list[int],
    batch_size: int,
) -> dict[str, Any]:
    gold = set(gold_pages)
    exact_gold = {int(item[1]) for item in exact_entries} & gold
    candidate_pages = [int(item[1]) for item in candidate_entries]
    first_rank_by_gold: dict[int, int] = {}
    for rank, page in enumerate(candidate_pages, start=1):
        if page in gold and page not in first_rank_by_gold:
            first_rank_by_gold[page] = rank
    if exact_gold:
        first_gold_rank: int | None = 0
    else:
        first_gold_rank = min(first_rank_by_gold.values(), default=None)
    remaining = gold - exact_gold
    all_gold_rank = (
        0
        if not remaining
        else max(first_rank_by_gold[page] for page in remaining)
        if remaining.issubset(first_rank_by_gold)
        else None
    )
    return {
        "exact_gold_pages": sorted(exact_gold),
        "exact_gold_page_recall": len(exact_gold) / len(gold),
        "first_gold_candidate_rank": first_gold_rank,
        "all_gold_candidate_rank": all_gold_rank,
        "batches_until_first_gold": _batches(first_gold_rank, batch_size),
        "batches_until_all_gold": _batches(all_gold_rank, batch_size),
        "hit_rate": {
            str(k): bool(exact_gold | (set(candidate_pages[:k]) & gold))
            for k in K_VALUES
        },
        "gold_page_recall": {
            str(k): len(exact_gold | (set(candidate_pages[:k]) & gold)) / len(gold)
            for k in K_VALUES
        },
        "complete_gold_page_hit": {
            str(k): gold.issubset(exact_gold | set(candidate_pages[:k]))
            for k in K_VALUES
        },
    }


def _build_summary(
    rows: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    device: str,
    elapsed_seconds: float,
    config_fingerprint: str,
) -> dict[str, Any]:
    policy_metrics = {
        policy: _aggregate_policy(rows, policy) for policy in POLICY_NAMES
    }
    winner = _choose_winner(policy_metrics)
    dense_query_ms = [float(row["timing_ms"]["dense"]) for row in rows]
    exact_ms = [float(row["timing_ms"]["exact"]) for row in rows]
    bm25_ms = [float(row["timing_ms"]["bm25"]) for row in rows]
    overlap = [int(row["source_overlap_top_50"]) for row in rows]
    return {
        "evaluation_version": EVALUATION_VERSION,
        "config_fingerprint": config_fingerprint,
        "scope": "3A all-Gold-pages-reachable questions only",
        "questions": len(rows),
        "documents": len(documents),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "fairness": {
            "exact_shared": True,
            "bm25_shared": True,
            "dense_shared": True,
            "candidate_batch_size": 5,
            "fallback_enabled": False,
            "visual_channel": "SearchUnit search-only metadata in BM25 and Dense",
        },
        "shared_retrieval_cost": {
            "device": device,
            "exact_latency_ms": _distribution(exact_ms),
            "bm25_query_latency_ms": _distribution(bm25_ms),
            "dense_query_latency_ms": _distribution(dense_query_ms),
            "dense_query_count": len(rows),
            "dense_segments": sum(int(item["dense_segments"]) for item in documents),
            "dense_passage_cache_hits": sum(
                int(item["dense_cache_hits"]) for item in documents
            ),
            "dense_passage_cache_misses": sum(
                int(item["dense_cache_misses"]) for item in documents
            ),
            "source_overlap_top_50": _distribution(overlap),
        },
        "policies": policy_metrics,
        "winner": winner,
        "documents_detail": documents,
    }


def _aggregate_policy(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    metrics = [row["policies"][policy]["metrics"] for row in rows]
    fusion_key = (
        "weighted_rrf_fusion" if policy == "weighted_rrf" else "fixed_quota_fusion"
    )
    first_ranks = [item["first_gold_candidate_rank"] for item in metrics]
    all_ranks = [item["all_gold_candidate_rank"] for item in metrics]
    return {
        "hit_rate": {
            str(k): sum(bool(item["hit_rate"][str(k)]) for item in metrics)
            / len(metrics)
            for k in K_VALUES
        },
        "mean_gold_page_recall": {
            str(k): sum(float(item["gold_page_recall"][str(k)]) for item in metrics)
            / len(metrics)
            for k in K_VALUES
        },
        "complete_gold_page_hit_rate": {
            str(k): sum(
                bool(item["complete_gold_page_hit"][str(k)]) for item in metrics
            )
            / len(metrics)
            for k in K_VALUES
        },
        "mrr": sum(
            1.0 if rank == 0 else 1.0 / rank if rank is not None else 0.0
            for rank in first_ranks
        )
        / len(metrics),
        "questions_with_no_retrievable_gold_page": sum(
            rank is None for rank in first_ranks
        ),
        "questions_with_incomplete_gold_retrieval": sum(
            rank is None for rank in all_ranks
        ),
        "previews_until_first_gold": _distribution(
            [rank for rank in first_ranks if rank is not None]
        ),
        "previews_until_all_gold": _distribution(
            [rank for rank in all_ranks if rank is not None]
        ),
        "batches_until_first_gold": _distribution(
            [
                item["batches_until_first_gold"]
                for item in metrics
                if item["batches_until_first_gold"] is not None
            ]
        ),
        "batches_until_all_gold": _distribution(
            [
                item["batches_until_all_gold"]
                for item in metrics
                if item["batches_until_all_gold"] is not None
            ]
        ),
        "fusion_latency_ms": _distribution(
            [float(row["timing_ms"][fusion_key]) for row in rows]
        ),
        "merged_duplicate_candidates": sum(
            int(row["policies"][policy]["duplicate_candidate_count"])
            for row in rows
        ),
    }


def _choose_winner(policies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def key(name: str) -> tuple[float, ...]:
        item = policies[name]
        return (
            float(item["mean_gold_page_recall"]["5"]),
            float(item["complete_gold_page_hit_rate"]["5"]),
            float(item["hit_rate"]["5"]),
            float(item["mrr"]),
            -float(item["fusion_latency_ms"]["mean"]),
        )

    ordered = sorted(POLICY_NAMES, key=lambda name: (key(name), name), reverse=True)
    return {
        "policy": ordered[0],
        "runner_up": ordered[1],
        "selection_key": list(key(ordered[0])),
        "rule": "lexicographic predeclared rule in config.json",
    }


def _exact_entries(result: Any) -> list[list[Any]]:
    entries: list[list[Any]] = []
    seen: set[str] = set()
    for match in result.exact_anchor_matches:
        if match.target_id in seen:
            continue
        seen.add(match.target_id)
        entries.append([match.target_id, match.page_number])
    return entries


def _eligible_question_ids(path: Path) -> set[str]:
    eligible: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("all_gold_pages_reachable") is True:
            eligible.add(str(row["question_id"]))
    return eligible


def _resolved_gold_pages_by_question(path: Path) -> dict[str, list[int]]:
    """Read the physical-page Gold mapping frozen by stage 3A.

    MMLongBench-Doc mixes physical PDF positions, printed page labels, and a
    handful of malformed values.  Stage 3B must therefore score against the
    audited physical-page mapping rather than re-reading raw annotations from
    the external manifest.
    """

    pages: dict[str, set[int]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("all_gold_pages_reachable") is not True:
            continue
        gold_page = row.get("gold_page_id")
        if not isinstance(gold_page, int):
            raise ValueError(
                f"Reachable row lacks an integer physical Gold page: {row}"
            )
        pages[str(row["question_id"])].add(gold_page)
    return {
        question_id: sorted(question_pages)
        for question_id, question_pages in pages.items()
    }


def _latest_completed_reachability_run(root: Path) -> Path:
    candidates = []
    for path in root.glob("reachability-*"):
        state_path = path / "run_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("state") == "completed":
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No completed stage-3A run found")
    return max(candidates, key=lambda path: path.name)


def _resolve_manifest_path(manifest: ExternalDatasetManifest, path: Path) -> Path:
    if path.is_absolute():
        return path
    root = manifest.path_root
    if not root.is_absolute():
        root = ROOT / root
    return root / path


def _batches(rank: int | None, batch_size: int) -> int | None:
    if rank is None:
        return None
    return 0 if rank == 0 else (rank + batch_size - 1) // batch_size


def _distribution(values: list[int] | list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(values: list[float], fraction: float) -> float:
    return values[round((len(values) - 1) * fraction)]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _load_document_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    return sorted(rows, key=lambda row: int(row["question_index"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 3B default retrieval comparison",
        "",
        f"- Questions: {summary['questions']}",
        f"- Documents: {summary['documents']}",
        f"- Device: {summary['shared_retrieval_cost']['device']}",
        f"- Winner: `{summary['winner']['policy']}`",
        "",
        "| Policy | Recall@5 | Complete@5 | Hit@5 | MRR | Mean batches first | Mean fusion ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICY_NAMES:
        item = summary["policies"][policy]
        lines.append(
            f"| {policy} | {item['mean_gold_page_recall']['5']:.6f} | "
            f"{item['complete_gold_page_hit_rate']['5']:.6f} | "
            f"{item['hit_rate']['5']:.6f} | {item['mrr']:.6f} | "
            f"{item['batches_until_first_gold']['mean']:.4f} | "
            f"{item['fusion_latency_ms']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Exact Anchor, BM25, Dense, visual SearchUnit metadata, and Dense cost "
            "are shared. Only candidate fusion differs.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
