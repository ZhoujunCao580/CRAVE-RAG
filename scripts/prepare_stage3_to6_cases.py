"""Build the deduplicated frozen case manifest for server stages 3 through 6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGE3 = {
    "3B_root_finalization": [24, 76, 380, 611],
    "3C_target_switch_recheck": [611, 375, 838],
    "3D_invisible_id_repair": [581, 746, 781, 786, 798, 880],
    "3E_checker_structural_repair": [771, 165],
    "3F_json_truncation": [80, 859, 327, 642, 468, 364],
    "3G_terminal_fallback": [877, 1004],
}

STAGE4 = {
    "4A_planner": [7, 1067, 1073, 565, 564],
    "4B_structural_coverage": [705, 874, 906],
    "4C_semantic_count": [
        913, 32, 89, 320, 422, 452, 580, 646, 721, 739, 774, 802, 879
    ],
    "4D_non_count_coverage": [99, 394, 564, 599, 644, 709, 710, 724, 804],
}

STAGE5 = [
    7, 10, 11, 24, 80, 114, 165, 171, 327, 372, 375, 393, 476, 516,
    535, 565, 579, 611, 642, 705, 771, 786, 787, 857, 859, 874, 913,
]

P07 = [76, 79, 101, 169, 378, 421, 543, 620, 711, 732, 747, 748, 775, 844, 857, 954]
CHECKER_REJECTED = [775, 366, 375, 378, 1059, 1070, 1071, 1074, 163, 837, 844, 338, 396, 859]
READ_NO_GAIN = [114, 476, 516, 393, 11]
P09_P10 = [80, 859, 327, 642, 468, 364, 771, 165]
STAGE1 = [17, 20, 412, 418, 423, 431, 448, 737, 787, 791, 793, 818, 953, 426, 705, 535, 658, 1086, 24, 716]
STAGE2 = [16, 171, 201, 389, 391, 474, 539, 603, 604, 708, 790, 801, 803, 844, 856, 1086, 320, 323, 328, 364, 372, 380, 468, 479, 535, 547, 548, 658, 716, 88, 579, 114, 378, 775]
RELATION = [444, 837, 838]


def _qid(number: int) -> str:
    return f"Q{number}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-cases", type=Path, required=True)
    parser.add_argument("--old-batch-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = {}
    for line in args.all_cases.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cases[row["case_id"]] = row
    old_manifest = json.loads(args.old_batch_manifest.read_text(encoding="utf-8"))
    old_program_failures = [
        row["case_id"] for row in old_manifest["cases"] if row["status"] == "failed"
    ]
    count_cases = [_qid(value) for value in STAGE4["4C_semantic_count"]]
    stage6_groups = {
        "old_program_failures": old_program_failures,
        "P07_provenance": [_qid(value) for value in P07],
        "checker_rejected": [_qid(value) for value in CHECKER_REJECTED],
        "read_no_state_gain": [_qid(value) for value in READ_NO_GAIN],
        "count_and_P09_P10": list(dict.fromkeys(count_cases + [_qid(v) for v in P09_P10])),
        "stage1_stage2_targeted": [
            _qid(value)
            for value in dict.fromkeys(STAGE1 + STAGE2 + RELATION)
        ],
    }
    groups = {
        "stage3": {key: [_qid(v) for v in values] for key, values in STAGE3.items()},
        "stage4": {key: [_qid(v) for v in values] for key, values in STAGE4.items()},
        "stage5": [_qid(value) for value in STAGE5],
        "stage6": stage6_groups,
    }
    requested = set()
    for values in groups["stage3"].values():
        requested.update(values)
    for values in groups["stage4"].values():
        requested.update(values)
    requested.update(groups["stage5"])
    for values in groups["stage6"].values():
        requested.update(values)
    missing = sorted(requested.difference(cases), key=lambda value: int(value[1:]))
    selected = sorted(requested.intersection(cases), key=lambda value: int(value[1:]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stage_groups.json").write_text(
        json.dumps(
            {
                "groups": groups,
                "requested_unique": len(requested),
                "selected_unique": len(selected),
                "missing": missing,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "stage3_to6_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case_id in selected:
            handle.write(json.dumps(cases[case_id], ensure_ascii=False) + "\n")
    print(json.dumps({"selected": len(selected), "missing": missing}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
