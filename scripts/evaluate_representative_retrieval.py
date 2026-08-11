"""Compare Exact, BM25, Dense, baseline interleave, and weighted RRF."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from softdoc.retrieval import (
    BM25Index,
    DenseConfig,
    DenseIndex,
    ExactAnchorLookup,
    FileEmbeddingCache,
    HuggingFaceE5Encoder,
    SearchSessionBuilder,
    SearchUnitBuilder,
    SubQuestionInput,
)
from softdoc.serialization import load_document

from build_representative_dense_index import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOFTDOC_ROOT,
    MODEL_NAME,
    _discover_document_dirs,
    _downloaded_revision,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "data" / "raw" / "mmlongbench_doc" / "questions.json"
K_VALUES = (1, 3, 5, 10, 20, 50)
PREVIEW_BATCH_SIZE = 5


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--softdoc-root",
        type=Path,
        action="append",
        dest="softdoc_roots",
        help="SoftDoc root; repeat to evaluate a combined corpus.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Optional existing retrieval root whose embedding_cache should be reused.",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    softdoc_roots = [
        path.resolve()
        for path in (args.softdoc_roots or [DEFAULT_SOFTDOC_ROOT])
    ]
    model_dir = args.model_dir.resolve()
    output_root = args.output_root.resolve()
    questions = json.loads(args.questions.resolve().read_text(encoding="utf-8-sig"))
    document_dirs = _discover_document_dirs(softdoc_roots)
    documents = {f"{path.name}.pdf": path for path in document_dirs}
    selected = [row for row in questions if str(row["doc_id"]) in documents]
    if not selected:
        raise RuntimeError("No benchmark questions matched the discovered SoftDocs")

    revision = _downloaded_revision(model_dir) or "main"
    encoder = HuggingFaceE5Encoder(
        model_name=MODEL_NAME,
        model_path=model_dir,
        model_revision=revision,
        tokenizer_revision=revision,
        device=args.device,
        local_files_only=True,
    )
    cache_root = (
        args.cache_root.resolve() if args.cache_root is not None else output_root
    )
    cache = FileEmbeddingCache(cache_root / "embedding_cache")
    exact_lookup = ExactAnchorLookup()
    by_document: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for question_index, question in enumerate(questions):
        doc_id = str(question["doc_id"])
        if doc_id in documents:
            by_document[doc_id].append((question_index, question))

    result_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for doc_id, path in documents.items():
        document = load_document(path)
        elements = {element.element_id: element for element in document.elements}
        build_result = SearchUnitBuilder().build(document)
        bm25_index = BM25Index(build_result)
        dense_index = DenseIndex(
            build_result,
            encoder,
            config=DenseConfig(batch_size=args.batch_size),
            cache=cache,
        )
        for question_index, source in by_document[doc_id]:
            question = SubQuestionInput(
                subquestion_id=f"mmlongbench-{question_index:04d}",
                text=str(source["question"]),
            )
            gold_pages = _evidence_pages(source.get("evidence_pages"))
            query_started = time.perf_counter()
            stage_started = time.perf_counter()
            exact = exact_lookup.lookup(question, document)
            exact_ms = (time.perf_counter() - stage_started) * 1000.0
            stage_started = time.perf_counter()
            bm25 = bm25_index.search(question)
            bm25_ms = (time.perf_counter() - stage_started) * 1000.0
            stage_started = time.perf_counter()
            dense = dense_index.search(question)
            dense_ms = (time.perf_counter() - stage_started) * 1000.0
            total_query_ms = (time.perf_counter() - query_started) * 1000.0

            exact_entries = _exact_entries(exact)
            bm25_entries = [
                _candidate_entry(candidate, "bm25") for candidate in bm25.candidates
            ]
            dense_entries = [
                _candidate_entry(candidate, "dense") for candidate in dense.candidates
            ]
            dual_entries = _round_robin(bm25_entries, dense_entries)
            exact_first_entries = _deduplicate_entries([*exact_entries, *dual_entries])
            search_session = SearchSessionBuilder().create(
                subquestion=question,
                search_units=build_result,
                exact=exact,
                bm25=bm25,
                dense=dense,
            )
            scope_audit = next(
                item.data
                for item in search_session.retrieval_trace
                if item.code == "search_metadata_scope_audit"
            )
            rrf_entries = [
                {
                    "candidate_id": candidate.element_id,
                    "element_id": candidate.element_id,
                    "page_number": candidate.page_number,
                    "source": "weighted_rrf",
                    "source_rank": rank,
                    "target_type": candidate.element_type.value,
                    "bm25_rank": candidate.bm25_rank,
                    "dense_rank": candidate.dense_rank,
                    "rrf_score": candidate.rrf_score,
                }
                for rank, candidate in enumerate(
                    search_session.candidate_catalog, start=1
                )
            ]
            exact_first_rrf_entries = _deduplicate_entries(
                [*exact_entries, *rrf_entries]
            )
            rrf_variants = {
                "rrf_equal_k60": _weighted_rrf_entries(
                    bm25_entries, dense_entries, k=60, bm25_weight=1.0,
                    dense_weight=1.0,
                ),
                "rrf_dense_1_25_k60": _weighted_rrf_entries(
                    bm25_entries, dense_entries, k=60, bm25_weight=1.0,
                    dense_weight=1.25,
                ),
                "rrf_dense_1_5_k60": _weighted_rrf_entries(
                    bm25_entries, dense_entries, k=60, bm25_weight=1.0,
                    dense_weight=1.5,
                ),
                "rrf_equal_k20": _weighted_rrf_entries(
                    bm25_entries, dense_entries, k=20, bm25_weight=1.0,
                    dense_weight=1.0,
                ),
                "rrf_dense_1_25_k20": rrf_entries,
            }
            gold = set(gold_pages)
            bm25_top_50_ids = {
                str(entry["candidate_id"]) for entry in bm25_entries[:50]
            }
            dense_top_50_ids = {
                str(entry["candidate_id"]) for entry in dense_entries[:50]
            }

            result_row = {
                    "question_index": question_index,
                    "doc_id": doc_id,
                    "doc_type": source.get("doc_type"),
                    "question": source["question"],
                    "answer": source.get("answer"),
                    "answer_format": source.get("answer_format"),
                    "evidence_sources": source.get("evidence_sources"),
                    "gold_pages": gold_pages,
                    "has_gold_pages": bool(gold_pages),
                    "timing_ms": {
                        "exact": round(exact_ms, 6),
                        "bm25": round(bm25_ms, 6),
                        "dense": round(dense_ms, 6),
                        "total": round(total_query_ms, 6),
                    },
                    "duplicate_candidates_top_50": len(
                        bm25_top_50_ids & dense_top_50_ids
                    ),
                    "search_metadata_scope_audit": scope_audit,
                    "exact": {
                        "anchor_count": len(exact.anchor_resolutions),
                        "statuses": [
                            resolution.status.value
                            for resolution in exact.anchor_resolutions
                        ],
                        "entries": exact_entries,
                        "gold_page_hit": _hit(exact_entries, gold, len(exact_entries)),
                    },
                    "bm25": _retriever_result(bm25_entries, gold, elements),
                    "dense": _retriever_result(dense_entries, gold, elements),
                    "dual_round_robin": _entry_result(dual_entries, gold),
                    "exact_first_dual": _entry_result(exact_first_entries, gold),
                    "weighted_rrf": _entry_result(rrf_entries, gold),
                    "exact_first_weighted_rrf": _entry_result(
                        exact_first_rrf_entries, gold
                    ),
                }
            for name, entries in rrf_variants.items():
                result_row[name] = _entry_result(entries, gold)
                result_row[f"exact_first_{name}"] = _entry_result(
                    _deduplicate_entries([*exact_entries, *entries]), gold
                )
            result_rows.append(result_row)
        print(
            f"{doc_id}: questions={len(by_document[doc_id])} "
            f"units={len(build_result.units)} "
            f"dense_cache={dense_index.cache_hits}/{dense_index.cache_misses}",
            flush=True,
        )

    summary = _summary(result_rows, time.perf_counter() - started, encoder.device)
    evaluation_dir = output_root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(evaluation_dir / "retrieval_results.jsonl", result_rows)
    (evaluation_dir / "retrieval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (evaluation_dir / "retrieval_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _exact_entries(result: Any) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for resolution in result.anchor_resolutions:
        for match in resolution.matches:
            if match.target_id in seen:
                continue
            seen.add(match.target_id)
            entries.append(
                {
                    "candidate_id": match.target_id,
                    "element_id": (
                        match.target_id if match.target_type.value != "page" else None
                    ),
                    "page_number": match.page_number,
                    "source": "exact",
                    "source_rank": len(entries) + 1,
                    "target_type": match.target_type.value,
                    "resolution_method": match.resolution_method,
                    "resolution_status": resolution.status.value,
                }
            )
    return entries


def _candidate_entry(candidate: Any, source: str) -> dict[str, object]:
    rank = candidate.bm25_rank if source == "bm25" else candidate.dense_rank
    return {
        "candidate_id": candidate.element_id,
        "element_id": candidate.element_id,
        "page_number": candidate.page_number,
        "source": source,
        "source_rank": rank,
        "target_type": candidate.element_type.value,
    }


def _round_robin(
    bm25: list[dict[str, object]],
    dense: list[dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    limit = max(len(bm25), len(dense))
    for index in range(limit):
        for collection in (bm25, dense):
            if index >= len(collection):
                continue
            entry = collection[index]
            candidate_id = str(entry["candidate_id"])
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            result.append(entry)
    return result


def _weighted_rrf_entries(
    bm25: list[dict[str, object]],
    dense: list[dict[str, object]],
    *,
    k: int,
    bm25_weight: float,
    dense_weight: float,
) -> list[dict[str, object]]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    best_entry: dict[str, dict[str, object]] = {}
    for entries, weight in ((bm25, bm25_weight), (dense, dense_weight)):
        for fallback_rank, entry in enumerate(entries, start=1):
            candidate_id = str(entry["candidate_id"])
            rank = int(entry.get("source_rank") or fallback_rank)
            scores[candidate_id] = scores.get(candidate_id, 0.0) + weight / (k + rank)
            if rank < best_rank.get(candidate_id, 1_000_000):
                best_rank[candidate_id] = rank
                best_entry[candidate_id] = entry
    return [
        {
            **best_entry[candidate_id],
            "source": "weighted_rrf",
            "source_rank": rank,
            "rrf_score": scores[candidate_id],
        }
        for rank, candidate_id in enumerate(
            sorted(
                scores,
                key=lambda item: (
                    -scores[item],
                    best_rank[item],
                    int(best_entry[item]["page_number"]),
                    item,
                ),
            ),
            start=1,
        )
    ]


def _deduplicate_entries(
    entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        candidate_id = str(entry["candidate_id"])
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        result.append(entry)
    return result


def _retriever_result(
    entries: list[dict[str, object]],
    gold: set[int],
    elements: dict[str, Any],
) -> dict[str, object]:
    result = _entry_result(entries, gold)
    result["candidate_count"] = len(entries)
    result["top_candidates"] = [
        {
            **entry,
            "text_preview": _preview(
                elements[str(entry["element_id"])].text
                if entry.get("element_id") in elements
                else None
            ),
        }
        for entry in entries[:50]
    ]
    return result


def _entry_result(
    entries: list[dict[str, object]],
    gold: set[int],
) -> dict[str, object]:
    pages = [int(entry["page_number"]) for entry in entries]
    first_rank = next(
        (
            rank
            for rank, page_number in enumerate(pages, start=1)
            if page_number in gold
        ),
        None,
    )
    return {
        "first_gold_page_rank": first_rank,
        "previews_until_first_gold": first_rank,
        "batches_until_first_gold": (
            (first_rank + PREVIEW_BATCH_SIZE - 1) // PREVIEW_BATCH_SIZE
            if first_rank is not None
            else None
        ),
        "hits": {str(k): _hit(entries, gold, k) for k in K_VALUES},
        "nearby_hits_radius_1": {
            str(k): _nearby_hit(entries, gold, k, radius=1) for k in K_VALUES
        },
        "gold_page_recall": {
            str(k): _gold_page_recall(entries, gold, k) for k in K_VALUES
        },
        "complete_gold_page_hits": {
            str(k): _complete_gold_page_hit(entries, gold, k) for k in K_VALUES
        },
    }


def _hit(entries: list[dict[str, object]], gold: set[int], k: int) -> bool:
    return bool(gold) and any(
        int(entry["page_number"]) in gold for entry in entries[:k]
    )


def _nearby_hit(
    entries: list[dict[str, object]],
    gold: set[int],
    k: int,
    *,
    radius: int,
) -> bool:
    if not gold:
        return False
    return any(
        any(abs(int(entry["page_number"]) - page) <= radius for page in gold)
        for entry in entries[:k]
    )


def _gold_page_recall(
    entries: list[dict[str, object]], gold: set[int], k: int
) -> float:
    if not gold:
        return 0.0
    retrieved_pages = {int(entry["page_number"]) for entry in entries[:k]}
    return len(retrieved_pages & gold) / len(gold)


def _complete_gold_page_hit(
    entries: list[dict[str, object]], gold: set[int], k: int
) -> bool:
    if not gold:
        return False
    retrieved_pages = {int(entry["page_number"]) for entry in entries[:k]}
    return gold.issubset(retrieved_pages)


def _summary(
    rows: list[dict[str, object]],
    elapsed_seconds: float,
    device: str,
) -> dict[str, object]:
    evaluable = [row for row in rows if row["has_gold_pages"]]
    anchored = [row for row in rows if row["exact"]["anchor_count"]]
    anchored_evaluable = [row for row in anchored if row["has_gold_pages"]]
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for status in row["exact"]["statuses"]:
            status_counts[str(status)] += 1

    retrievers = {
        name: _metrics(evaluable, name)
        for name in (
            "bm25",
            "dense",
            "dual_round_robin",
            "exact_first_dual",
            "weighted_rrf",
            "exact_first_weighted_rrf",
            "exact_first_rrf_equal_k60",
            "exact_first_rrf_dense_1_25_k60",
            "exact_first_rrf_dense_1_5_k60",
            "exact_first_rrf_equal_k20",
            "exact_first_rrf_dense_1_25_k20",
        )
    }
    complementarity: dict[str, dict[str, object]] = {}
    for k in K_VALUES:
        bm25_hits = [bool(row["bm25"]["hits"][str(k)]) for row in evaluable]
        dense_hits = [bool(row["dense"]["hits"][str(k)]) for row in evaluable]
        both = sum(left and right for left, right in zip(bm25_hits, dense_hits))
        bm25_only = sum(left and not right for left, right in zip(bm25_hits, dense_hits))
        dense_only = sum(right and not left for left, right in zip(bm25_hits, dense_hits))
        neither = len(evaluable) - both - bm25_only - dense_only
        complementarity[str(k)] = {
            "both": both,
            "bm25_only": bm25_only,
            "dense_only": dense_only,
            "neither": neither,
            "oracle_union_hit_rate": (both + bm25_only + dense_only) / len(evaluable),
        }

    return {
        "scope": (
            "Entry-retrieval evaluation on the one-based evidence page numbers "
            "in the local MMLongBench questions.json, compared with SoftDoc "
            "Page.page_number. Gold Element IDs are unavailable; no answer "
            "generation is measured."
        ),
        "combination_policy": (
            "The current online policy keeps Exact handles separate, then uses "
            "weighted RRF over Element-level BM25 and Dense ranks with k=20, "
            "BM25 weight=1.0 and Dense weight=1.25. The previous round-robin "
            "policy remains in the report as a baseline."
        ),
        "complementarity_definition": (
            "Oracle union at K means BM25 top-K OR Dense top-K succeeds; it may "
            "inspect up to 2K candidates and is not an online ranking."
        ),
        "device": device,
        "documents": len({str(row["doc_id"]) for row in rows}),
        "questions": len(rows),
        "questions_with_gold_pages": len(evaluable),
        "questions_without_gold_pages": len(rows) - len(evaluable),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "query_latency_ms": _timing_metrics(rows),
        "search_metadata_scope_audit": _metadata_scope_summary(rows),
        "candidate_inspection": _candidate_inspection_metrics(evaluable),
        "exact": {
            "questions_with_anchor": len(anchored),
            "questions_with_anchor_and_gold": len(anchored_evaluable),
            "resolution_status_counts": dict(sorted(status_counts.items())),
            "gold_page_hits": sum(
                bool(row["exact"]["gold_page_hit"])
                for row in anchored_evaluable
            ),
            "gold_page_hit_rate": (
                sum(
                    bool(row["exact"]["gold_page_hit"])
                    for row in anchored_evaluable
                )
                / len(anchored_evaluable)
                if anchored_evaluable
                else None
            ),
        },
        "retrievers": retrievers,
        "bm25_dense_complementarity": complementarity,
    }


def _metrics(rows: list[dict[str, object]], name: str) -> dict[str, object]:
    reciprocal = [
        1.0 / int(row[name]["first_gold_page_rank"])
        if row[name]["first_gold_page_rank"] is not None
        else 0.0
        for row in rows
    ]
    return {
        "hit_rate": {
            str(k): sum(bool(row[name]["hits"][str(k)]) for row in rows) / len(rows)
            for k in K_VALUES
        },
        "mrr": sum(reciprocal) / len(rows),
        "nearby_hit_rate_radius_1": {
            str(k): sum(
                bool(row[name]["nearby_hits_radius_1"][str(k)]) for row in rows
            )
            / len(rows)
            for k in K_VALUES
        },
        "mean_gold_page_recall": {
            str(k): sum(
                float(row[name]["gold_page_recall"][str(k)]) for row in rows
            )
            / len(rows)
            for k in K_VALUES
        },
        "complete_gold_page_hit_rate": {
            str(k): sum(
                bool(row[name]["complete_gold_page_hits"][str(k)])
                for row in rows
            )
            / len(rows)
            for k in K_VALUES
        },
    }


def _timing_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        stage: _distribution(
            [float(row["timing_ms"][stage]) for row in rows]
        )
        for stage in ("exact", "bm25", "dense", "total")
    }


def _metadata_scope_summary(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {
        "definition": (
            "Metadata-only means the matched range is outside SearchUnit.content_text "
            "and lies only in section_path or label context. This is diagnostic and "
            "does not alter ranking."
        ),
        "questions": len(rows),
    }
    for source in ("bm25", "dense"):
        evaluated = sum(
            int(row["search_metadata_scope_audit"][f"{source}_evaluated"])
            for row in rows
        )
        metadata_only = sum(
            int(row["search_metadata_scope_audit"][f"{source}_metadata_only"])
            for row in rows
        )
        result[source] = {
            "top_n": 5,
            "evaluated_candidates": evaluated,
            "metadata_only_candidates": metadata_only,
            "metadata_only_rate": metadata_only / evaluated if evaluated else None,
            "questions_with_metadata_only_candidate": sum(
                int(row["search_metadata_scope_audit"][f"{source}_metadata_only"])
                > 0
                for row in rows
            ),
        }
    return result


def _candidate_inspection_metrics(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    name = "exact_first_weighted_rrf"
    ranks = [
        int(row[name]["first_gold_page_rank"])
        for row in rows
        if row[name]["first_gold_page_rank"] is not None
    ]
    batches = [
        int(row[name]["batches_until_first_gold"])
        for row in rows
        if row[name]["batches_until_first_gold"] is not None
    ]
    duplicates = [int(row["duplicate_candidates_top_50"]) for row in rows]
    source_rank_distributions: dict[str, dict[str, object]] = {}
    for retriever in (
        "bm25",
        "dense",
        "dual_round_robin",
        "exact_first_dual",
        "weighted_rrf",
        name,
    ):
        values = [
            int(row[retriever]["first_gold_page_rank"])
            for row in rows
            if row[retriever]["first_gold_page_rank"] is not None
        ]
        source_rank_distributions[retriever] = {
            "retrievable_questions": len(values),
            "unretrievable_questions": len(rows) - len(values),
            "first_gold_page_rank": _distribution(values),
        }
    return {
        "policy": "exact_first_then_weighted_rrf",
        "preview_batch_size": PREVIEW_BATCH_SIZE,
        "questions_with_retrievable_gold_page": len(ranks),
        "questions_without_retrievable_gold_page": len(rows) - len(ranks),
        "previews_until_first_gold": _distribution(ranks),
        "batches_until_first_gold": _distribution(batches),
        "duplicate_candidates_in_bm25_dense_top_50": _distribution(duplicates),
        "first_gold_page_by_source": source_rank_distributions,
    }


def _distribution(values: list[float] | list[int]) -> dict[str, object]:
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
    index = round((len(values) - 1) * fraction)
    return values[index]


def _evidence_pages(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if not isinstance(value, str):
        return []
    return [int(item) for item in ast.literal_eval(value)]


def _preview(value: str | None, limit: int = 320) -> str:
    return re.sub(r"\s+", " ", value or "").strip()[:limit]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _summary_markdown(summary: dict[str, object]) -> str:
    exact = summary["exact"]
    retrievers = summary["retrievers"]
    lines = [
        f"# Representative-{summary['documents']} Retrieval Evaluation",
        "",
        str(summary["scope"]),
        "",
        f"- Questions: {summary['questions']}",
        f"- Questions with Gold pages: {summary['questions_with_gold_pages']}",
        f"- Questions without Gold pages: {summary['questions_without_gold_pages']}",
        f"- Device: {summary['device']}",
        f"- Elapsed seconds: {summary['elapsed_seconds']}",
        f"- Mean query latency: "
        f"{summary['query_latency_ms']['total']['mean']:.3f} ms",
        f"- P95 query latency: "
        f"{summary['query_latency_ms']['total']['p95']:.3f} ms",
        "",
        "## Exact",
        "",
        f"- Questions with supported Anchor: {exact['questions_with_anchor']}",
        f"- Anchored questions with Gold pages: "
        f"{exact['questions_with_anchor_and_gold']}",
        f"- Gold-page hits: {exact['gold_page_hits']}",
        f"- Gold-page hit rate: {exact['gold_page_hit_rate']}",
        f"- Resolution statuses: {exact['resolution_status_counts']}",
        "",
        "## Ranked entry retrieval",
        "",
        "| K | BM25 | Dense | Dual baseline | Exact+dual | Weighted RRF | Exact+RRF |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in K_VALUES:
        lines.append(
            f"| {k} | {retrievers['bm25']['hit_rate'][str(k)]:.4f} | "
            f"{retrievers['dense']['hit_rate'][str(k)]:.4f} | "
            f"{retrievers['dual_round_robin']['hit_rate'][str(k)]:.4f} | "
            f"{retrievers['exact_first_dual']['hit_rate'][str(k)]:.4f} | "
            f"{retrievers['weighted_rrf']['hit_rate'][str(k)]:.4f} | "
            f"{retrievers['exact_first_weighted_rrf']['hit_rate'][str(k)]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Full-ranking RRF policy comparison (Exact-first)",
            "",
            "| Policy | Hit@1 | Hit@5 | Hit@20 | Hit@50 | MRR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "exact_first_rrf_equal_k60",
        "exact_first_rrf_dense_1_25_k60",
        "exact_first_rrf_dense_1_5_k60",
        "exact_first_rrf_equal_k20",
        "exact_first_rrf_dense_1_25_k20",
    ):
        item = retrievers[name]
        label = name.removeprefix("exact_first_")
        lines.append(
            f"| {label} | {item['hit_rate']['1']:.4f} | "
            f"{item['hit_rate']['5']:.4f} | {item['hit_rate']['20']:.4f} | "
            f"{item['hit_rate']['50']:.4f} | {item['mrr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## BM25 and Dense complementarity",
            "",
            "| K | Both | BM25 only | Dense only | Neither | Oracle union |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k in K_VALUES:
        item = summary["bm25_dense_complementarity"][str(k)]
        lines.append(
            f"| {k} | {item['both']} | {item['bm25_only']} | "
            f"{item['dense_only']} | {item['neither']} | "
            f"{item['oracle_union_hit_rate']:.4f} |"
        )
    lines.extend(["", str(summary["combination_policy"]), ""])
    lines.extend([str(summary["complementarity_definition"]), ""])
    scope_audit = summary["search_metadata_scope_audit"]
    lines.extend(
        [
            "## Search metadata scope audit",
            "",
            str(scope_audit["definition"]),
            "",
            "| Source | Top-N | Evaluated | Metadata-only | Rate | Questions affected |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for source in ("bm25", "dense"):
        item = scope_audit[source]
        lines.append(
            f"| {source} | {item['top_n']} | {item['evaluated_candidates']} | "
            f"{item['metadata_only_candidates']} | "
            f"{item['metadata_only_rate']:.4f} | "
            f"{item['questions_with_metadata_only_candidate']} |"
        )
    lines.append("")
    inspection = summary["candidate_inspection"]
    lines.extend(
        [
            "## Candidate inspection cost (offline simulation)",
            "",
            f"- Preview batch size: {inspection['preview_batch_size']}",
            f"- Mean previews until first Gold page: "
            f"{inspection['previews_until_first_gold']['mean']}",
            f"- Mean batches until first Gold page: "
            f"{inspection['batches_until_first_gold']['mean']}",
            f"- Questions with no retrievable Gold page: "
            f"{inspection['questions_without_retrievable_gold_page']}",
            "",
            "`nearby_hit_rate_radius_1` is an optimistic navigation proxy only. "
            "It does not claim that an Agent found evidence or answered correctly.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
