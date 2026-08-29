# Planner suite inventory

- Canonical Prompt: `src/softdoc/prompts/planner_v0_20.txt`
- Registry version: `planner-v0.20`
- Compact source: `scripts/evaluate_planner_mock.py` (6 synthetic cases; inputs
  and expected constraints are inline)
- Historical v0.18 runs and proposal reports were removed after materializing
  the Prompt-independent case and Gold files below.

The 49 cases are a Prompt-independent development set. Their original v0.18
run is historical and is not a result for the current Prompt.

## Materialized suite

- Model-visible questions:
  [`model_inputs/planner_cases_v1.jsonl`](model_inputs/planner_cases_v1.jsonl)
- Reviewer-only Gold:
  [`review_only/planner_gold_v1.jsonl`](review_only/planner_gold_v1.jsonl)
- Machine-readable inventory and current usage status:
  [`suite_manifest.json`](suite_manifest.json)

The model-visible JSONL contains exactly `case_id` and `question`. It does not
contain split names, expected shapes, labels, scoring rules, document metadata,
or adjudications. Those remain under `review_only/`.

The 49-case source contains 8 contract regressions, 14 synthetic boundaries,
6 real supplemental regressions, 14 real dev cases, and 7 former holdout cases.
All 49 were included in an earlier run, so the former holdout must not be
reported as unseen in future Prompt selection. The current status is
authoritative in `suite_manifest.json` and this README.

Still needed before formal testing: approve the remaining dev labels and select
a new unseen real holdout separately.
