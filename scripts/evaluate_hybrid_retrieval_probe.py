"""Paired retrieval probe over the Pipeline/Hybrid difficult-page packet."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from softdoc.retrieval import (
    BM25Index,
    DenseIndex,
    FileEmbeddingCache,
    HuggingFaceE5Encoder,
    SearchSessionBuilder,
    SearchUnitBuilder,
    SubQuestionInput,
)
from softdoc.serialization import load_document

from build_representative_dense_index import (
    DEFAULT_MODEL_DIR,
    MODEL_NAME,
    _downloaded_revision,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--hybrid-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    question_rows = _strict_questions(manifest, questions)
    revision = _downloaded_revision(args.model_dir) or "main"
    encoder = HuggingFaceE5Encoder(
        model_name=MODEL_NAME,
        model_path=args.model_dir,
        model_revision=revision,
        tokenizer_revision=revision,
        device=args.device,
        local_files_only=True,
    )
    cache = FileEmbeddingCache(args.output.parent / "hybrid_probe_embedding_cache")
    results: list[dict[str, Any]] = []
    corpus: dict[str, dict[str, Any]] = {}
    for backend, root in (
        ("pipeline", args.pipeline_root),
        ("hybrid", args.hybrid_root),
    ):
        for challenge in manifest:
            challenge_id = challenge["challenge_id"]
            document_dir = root / challenge_id
            document = load_document(document_dir)
            search_units = SearchUnitBuilder().build(document)
            bm25_index = BM25Index(search_units)
            dense_index = DenseIndex(search_units, encoder, cache=cache)
            corpus[f"{backend}:{challenge_id}"] = {
                "pages": len(document.pages),
                "elements": len(document.elements),
                "search_units": len(search_units.units),
                "skipped": {
                    reason: sum(
                        item.reason == reason
                        for item in search_units.skipped_elements
                    )
                    for reason in sorted(
                        {item.reason for item in search_units.skipped_elements}
                    )
                },
            }
            for item in question_rows.get(challenge_id, []):
                question = SubQuestionInput(
                    subquestion_id=f"probe:{item['question_index']}",
                    text=item["question"],
                )
                bm25 = bm25_index.search(question)
                dense = dense_index.search(question)
                session = SearchSessionBuilder().create(
                    subquestion=question,
                    search_units=search_units,
                    bm25=bm25,
                    dense=dense,
                )
                gold = set(item["challenge_gold_pages"])
                results.append(
                    {
                        "backend": backend,
                        "challenge_id": challenge_id,
                        **item,
                        "bm25_first_gold_rank": _first_rank(
                            bm25.candidates, gold
                        ),
                        "dense_first_gold_rank": _first_rank(
                            dense.candidates, gold
                        ),
                        "rrf_first_gold_rank": _first_rank(
                            session.candidate_catalog, gold
                        ),
                        "rrf_top_5": [
                            {
                                "element_id": candidate.element_id,
                                "page_number": candidate.page_number,
                                "element_type": candidate.element_type.value,
                                "bm25_rank": candidate.bm25_rank,
                                "dense_rank": candidate.dense_rank,
                                "rrf_score": candidate.rrf_score,
                            }
                            for candidate in session.candidate_catalog[:5]
                        ],
                    }
                )
    report = {
        "scope": (
            "Only benchmark questions whose complete Gold page set is contained "
            "inside a difficult-page mini-document are included. This paired "
            "probe measures direction on three questions, not full-corpus accuracy."
        ),
        "strict_question_count": sum(len(items) for items in question_rows.values()),
        "corpus": corpus,
        "results": results,
        "summary": _summary(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


def _strict_questions(
    manifest: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_source = {}
    for item in manifest:
        source_name = Path(item["source_pdf"]).name
        if source_name.endswith("_origin.pdf"):
            source_name = source_name.removesuffix("_origin.pdf") + ".pdf"
        by_source[source_name] = item
    result: dict[str, list[dict[str, Any]]] = {}
    for question_index, question in enumerate(questions):
        challenge = by_source.get(str(question["doc_id"]))
        if challenge is None:
            continue
        source_pages = [int(page) for page in challenge["source_page_numbers"]]
        gold_pages = _pages(question.get("evidence_pages"))
        if not gold_pages or not set(gold_pages).issubset(set(source_pages)):
            continue
        page_map = {
            source: challenge_page
            for source, challenge_page in zip(
                source_pages, challenge["challenge_page_numbers"]
            )
        }
        result.setdefault(challenge["challenge_id"], []).append(
            {
                "question_index": question_index,
                "question": str(question["question"]),
                "source_gold_pages": gold_pages,
                "challenge_gold_pages": [page_map[page] for page in gold_pages],
            }
        )
    return result


def _pages(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return [int(item) for item in parsed]
    return []


def _first_rank(candidates: list[Any], gold_pages: set[int]) -> int | None:
    return next(
        (
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if candidate.page_number in gold_pages
        ),
        None,
    )


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for backend in ("pipeline", "hybrid"):
        rows = [row for row in results if row["backend"] == backend]
        summary[backend] = {
            "questions": len(rows),
            "bm25_top1": sum(row["bm25_first_gold_rank"] == 1 for row in rows),
            "dense_top1": sum(row["dense_first_gold_rank"] == 1 for row in rows),
            "rrf_top1": sum(row["rrf_first_gold_rank"] == 1 for row in rows),
            "rrf_mean_rank": (
                sum(row["rrf_first_gold_rank"] for row in rows) / len(rows)
                if rows
                else None
            ),
        }
    return summary


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pipeline vs Hybrid retrieval probe",
        "",
        str(report["scope"]),
        "",
        "| Backend | Questions | BM25 Top-1 | Dense Top-1 | RRF Top-1 | RRF mean rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for backend in ("pipeline", "hybrid"):
        item = report["summary"][backend]
        lines.append(
            f"| {backend} | {item['questions']} | {item['bm25_top1']} | "
            f"{item['dense_top1']} | {item['rrf_top1']} | {item['rrf_mean_rank']} |"
        )
    lines.extend(["", "## Questions", ""])
    for row in report["results"]:
        lines.append(
            f"- {row['backend']} / {row['challenge_id']}: "
            f"RRF rank={row['rrf_first_gold_rank']} — {row['question']}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
