# Answerer suite inventory

- Canonical Prompt: `src/softdoc/prompts/answerer_v0_8.txt`
- Registry version: `answerer-v0.8`
- Current source and runner: `scripts/evaluate_answerer_mock.py`
- Frozen suite: 24 English synthetic evidence-package cases
  - 14 existing contract regressions;
  - 9 controlled boundaries;
  - 1 defensive incomplete-packet case that is excluded from normal
    ready-only performance.
- Model-visible inputs: `model_inputs/answerer_cases_v1.jsonl`
- Reviewer-only Gold: `review_only/answerer_gold_v1.jsonl`
- Manifest: `suite_manifest.json`

The legacy Python runner keeps each `AnswerInput` beside its scoring
conditions, while the materialized suite separates those concerns. A Codex generation
conversation may read only the model-visible file, the Prompt, and the output
Schema. It must not read `review_only/`.

All current questions are in English. Case `08_language` checks that the
Answerer follows the language requested by the root question. The canonical
Prompt now also requires the shortest self-contained answer and gives generic
yes/no, numeric, unit-bearing, and list-format examples.

The added cases cover percentage points versus relative percentage change,
unit conversion, rounding, exclusion, yes/no comparison, temporal scope,
explicit causality versus correlation, incomplete multi-part evidence, and a
long noisy Evidence package.

Still needed before formal testing:

- retain synthetic cases for calculation, conflict, insufficiency, irrelevant
  evidence, condition mismatch, empty graph, dependency synthesis, unsupported
  causality, ranking, and graph-as-non-evidence;
- add frozen real `AnswerInput` packets built from Checker-accepted Evidence;
- add a second integration lane using complete end-to-end trajectories;
- reserve an unseen real holdout.

Answerer does not directly consume a Controller action. Controller affects what
is read; Checker decides what enters EvidenceMemory; the Answerer receives the
resulting question graph and Evidence package.
