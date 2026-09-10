"""Build the document-level split used by the CRAVE-RAG SFT programme.

The split deliberately separates two notions of development data:

* ``train`` and ``diagnostic_dev`` are drawn from documents already touched by
  the 177-case failure analysis;
* ``clean_dev`` and ``test`` are drawn from the remaining documents.

No document may occur in more than one split.  The generated case manifests do
not contain Gold answers or Gold evidence.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "data" / "raw" / "mmlongbench_doc" / "questions.json"
DEFAULT_HARD_CASES = (
    ROOT / "configs" / "evaluation" / "next_round_unresolved_177_v0_1.jsonl"
)
DEFAULT_OUTPUT = ROOT / "configs" / "training" / "mmlongbench_sft_split_v0_1"
SCHEMA_VERSION = "mmlongbench-sft-document-split-v0.1"


_COVERAGE_RE = re.compile(
    r"\b(how many|number of|count|among all|across (?:all|the)|entire document|"
    r"whole document|list all|each of the|every (?:figure|table|chart|page))\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(highest|lowest|largest|smallest|most|least|greater|larger|smaller|"
    r"maximum|minimum|argmax|argmin|compare|compared)\b",
    re.IGNORECASE,
)
_DERIVED_RE = re.compile(
    r"\b(percentage|percent|ratio|growth|margin|return on|increase|decrease|"
    r"difference|average|change)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Question:
    case_id: str
    doc_id: str
    doc_type: str
    text: str
    features: frozenset[str]


def _parse_literal_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return [value]
    return parsed if isinstance(parsed, list) else [parsed]


def _normalise_doc_id(value: str) -> str:
    return Path(value).stem


def _question_features(row: dict[str, Any]) -> frozenset[str]:
    text = str(row.get("question", ""))
    sources = {str(item).strip().lower() for item in _parse_literal_list(row.get("evidence_sources"))}
    pages = {str(item).strip() for item in _parse_literal_list(row.get("evidence_pages"))}
    answer = str(row.get("answer", "")).strip().lower().replace("_", " ")
    features: set[str] = set()
    if any(item in sources for item in {"figure", "chart"}):
        features.add("visual")
    if "table" in sources:
        features.add("table")
    if "text" in sources:
        features.add("text")
    if len(sources) > 1:
        features.add("mixed_evidence")
    if len(pages) > 1:
        features.add("multi_page")
    if _COVERAGE_RE.search(text):
        features.add("coverage_or_enumeration")
    if _COMPARISON_RE.search(text):
        features.add("comparison_or_extremum")
    if _DERIVED_RE.search(text):
        features.add("derived_metric")
    if answer in {"not answerable", "notanswerable"}:
        features.add("not_answerable")
    structural = {
        "visual",
        "table",
        "mixed_evidence",
        "multi_page",
        "coverage_or_enumeration",
        "comparison_or_extremum",
        "derived_metric",
    }
    if len(features & structural) >= 2:
        features.add("structurally_hard")
    return frozenset(features)


def load_questions(path: Path) -> list[Question]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("questions JSON must contain a list")
    result: list[Question] = []
    for index, row in enumerate(rows):
        result.append(
            Question(
                case_id=f"Q{index}",
                doc_id=_normalise_doc_id(str(row["doc_id"])),
                doc_type=str(row["doc_type"]),
                text=str(row["question"]),
                features=_question_features(row),
            )
        )
    return result


def load_case_ids(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.add(str(json.loads(line)["case_id"]))
    return result


def _counter_for_docs(
    docs: Iterable[str], questions_by_doc: dict[str, list[Question]], hard_ids: set[str]
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for doc_id in docs:
        rows = questions_by_doc[doc_id]
        counter["documents"] += 1
        counter["questions"] += len(rows)
        counter["diagnostic_hard_cases"] += sum(row.case_id in hard_ids for row in rows)
        counter[f"doc_type::{rows[0].doc_type}"] += 1
        for row in rows:
            for feature in row.features:
                counter[f"feature::{feature}"] += 1
    return counter


def _balanced_subset(
    population: Sequence[str],
    size: int,
    questions_by_doc: dict[str, list[Question]],
    hard_ids: set[str],
    rng: random.Random,
    *,
    trials: int = 50_000,
) -> set[str]:
    if size < 0 or size > len(population):
        raise ValueError("invalid subset size")
    if size == 0:
        return set()
    total = _counter_for_docs(population, questions_by_doc, hard_ids)
    keys = [
        key
        for key in total
        if key.startswith("doc_type::")
        or key.startswith("feature::")
        or key in {"questions", "diagnostic_hard_cases"}
    ]
    target_fraction = size / len(population)

    def score(sample: Sequence[str]) -> float:
        observed = _counter_for_docs(sample, questions_by_doc, hard_ids)
        value = 0.0
        for key in keys:
            target = total[key] * target_fraction
            scale = max(1.0, target)
            weight = 3.0 if key.startswith("doc_type::") else 1.0
            if key == "diagnostic_hard_cases":
                weight = 4.0
            value += weight * ((observed[key] - target) / scale) ** 2
        return value

    best = list(population[:size])
    best_score = score(best)
    population_list = list(population)
    for _ in range(trials):
        candidate = rng.sample(population_list, size)
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best, best_score = candidate, candidate_score
    return set(best)


def _challenge_docs(
    population: Sequence[str],
    size: int,
    questions_by_doc: dict[str, list[Question]],
) -> set[str]:
    """Select hard documents while retaining broad document-type coverage."""

    def score(doc_id: str) -> tuple[float, int, str]:
        rows = questions_by_doc[doc_id]
        hard = sum("structurally_hard" in row.features for row in rows)
        visual = sum("visual" in row.features for row in rows)
        multi_page = sum("multi_page" in row.features for row in rows)
        coverage = sum("coverage_or_enumeration" in row.features for row in rows)
        weighted = (hard * 3 + visual + multi_page * 2 + coverage * 2) / max(1, len(rows))
        return weighted, len(rows), doc_id

    ranked = sorted(population, key=score, reverse=True)
    selected: list[str] = []
    used_types: Counter[str] = Counter()
    while ranked and len(selected) < size:
        best_index = max(
            range(len(ranked)),
            key=lambda index: (
                score(ranked[index])[0] + (0.75 if used_types[questions_by_doc[ranked[index]][0].doc_type] == 0 else 0),
                score(ranked[index])[1],
                ranked[index],
            ),
        )
        doc_id = ranked.pop(best_index)
        selected.append(doc_id)
        used_types[questions_by_doc[doc_id][0].doc_type] += 1
    return set(selected)


def _split_summary(
    docs: set[str], questions_by_doc: dict[str, list[Question]], hard_ids: set[str]
) -> dict[str, Any]:
    counts = _counter_for_docs(sorted(docs), questions_by_doc, hard_ids)
    return {
        "document_count": counts["documents"],
        "question_count": counts["questions"],
        "diagnostic_hard_case_count": counts["diagnostic_hard_cases"],
        "document_types": {
            key.removeprefix("doc_type::"): value
            for key, value in sorted(counts.items())
            if key.startswith("doc_type::")
        },
        "question_features": {
            key.removeprefix("feature::"): value
            for key, value in sorted(counts.items())
            if key.startswith("feature::")
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _write_cases(
    path: Path,
    docs: set[str],
    questions_by_doc: dict[str, list[Question]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for doc_id in sorted(docs):
            for row in questions_by_doc[doc_id]:
                payload = {
                    "case_id": row.case_id,
                    "question_id": f"mmlongbench-doc:{row.case_id}",
                    "document_id": doc_id,
                    "document_dir": f"/workspace/data/mmlongbench-doc/softdocs/{doc_id}",
                    "question": row.text,
                }
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_split(
    questions: list[Question],
    hard_ids: set[str],
    *,
    seed: int,
    train_docs: int,
    diagnostic_dev_docs: int,
    clean_dev_docs: int,
    test_core_docs: int,
    test_challenge_docs: int,
) -> tuple[dict[str, set[str]], dict[str, list[Question]]]:
    questions_by_doc: dict[str, list[Question]] = defaultdict(list)
    by_case_id: dict[str, Question] = {}
    for row in questions:
        questions_by_doc[row.doc_id].append(row)
        by_case_id[row.case_id] = row
    missing = hard_ids - by_case_id.keys()
    if missing:
        raise ValueError(f"unknown hard case IDs: {sorted(missing)}")

    touched_docs = {by_case_id[case_id].doc_id for case_id in hard_ids}
    if train_docs + diagnostic_dev_docs != len(touched_docs):
        raise ValueError(
            "train_docs + diagnostic_dev_docs must equal the number of documents touched by hard cases "
            f"({len(touched_docs)})"
        )
    rng = random.Random(seed)
    diagnostic = _balanced_subset(
        sorted(touched_docs), diagnostic_dev_docs, questions_by_doc, hard_ids, rng
    )
    train = touched_docs - diagnostic

    untouched = set(questions_by_doc) - touched_docs
    challenge = _challenge_docs(sorted(untouched), test_challenge_docs, questions_by_doc)
    remaining = untouched - challenge
    clean_dev = _balanced_subset(
        sorted(remaining), clean_dev_docs, questions_by_doc, hard_ids, rng
    )
    remaining -= clean_dev
    core = _balanced_subset(
        sorted(remaining), test_core_docs, questions_by_doc, hard_ids, rng
    )
    remaining -= core

    split = {
        "train": train,
        "diagnostic_dev": diagnostic,
        "clean_dev": clean_dev,
        "test_core": core,
        "test_challenge": challenge,
        "reserve": remaining,
    }
    flattened = [doc_id for docs in split.values() for doc_id in docs]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(questions_by_doc):
        raise AssertionError("document overlap or omission detected")
    return split, dict(questions_by_doc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--hard-cases", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--train-docs", type=int, default=42)
    parser.add_argument("--diagnostic-dev-docs", type=int, default=12)
    parser.add_argument("--clean-dev-docs", type=int, default=8)
    parser.add_argument("--test-core-docs", type=int, default=10)
    parser.add_argument("--test-challenge-docs", type=int, default=6)
    args = parser.parse_args(argv)

    questions_path = args.questions.resolve()
    hard_path = args.hard_cases.resolve()
    questions = load_questions(questions_path)
    hard_ids = load_case_ids(hard_path)
    split, questions_by_doc = build_split(
        questions,
        hard_ids,
        seed=args.seed,
        train_docs=args.train_docs,
        diagnostic_dev_docs=args.diagnostic_dev_docs,
        clean_dev_docs=args.clean_dev_docs,
        test_core_docs=args.test_core_docs,
        test_challenge_docs=args.test_challenge_docs,
    )

    source_fingerprint = sha256(questions_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "source": {
            "questions": _portable_path(args.questions),
            "questions_sha256": source_fingerprint,
            "diagnostic_cases": _portable_path(args.hard_cases),
            "diagnostic_case_count": len(hard_ids),
        },
        "policy": {
            "unit": "document",
            "document_overlap_allowed": False,
            "gold_answers_in_case_manifests": False,
            "train_and_diagnostic_dev_source": "documents touched by the audited 177-case diagnostic set",
            "clean_dev_and_test_source": "remaining documents",
            "test_slices": {
                "core": "document-type and question-feature balanced",
                "challenge": "structural difficulty enriched without using model outcomes",
            },
            "reporting_boundary": "Internal development split only; not a clean external benchmark holdout.",
        },
        "splits": {
            name: {
                "documents": sorted(docs),
                "summary": _split_summary(docs, questions_by_doc, hard_ids),
            }
            for name, docs in split.items()
        },
    }
    output_dir = args.output_dir.resolve()
    _write_json(output_dir / "split_manifest.json", manifest)
    for name, docs in split.items():
        _write_cases(output_dir / f"{name}.jsonl", docs, questions_by_doc)
    print(json.dumps({name: manifest["splits"][name]["summary"] for name in split}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
