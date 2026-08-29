# Controller suite inventory

- Canonical Prompt: `src/softdoc/prompts/controller_policy_v0_7.txt`
- Registry version: `controller-policy-v0.7`
- Current source and runner: `scripts/evaluate_controller_mock.py`
- Materialized suite: 28 Controller decisions
  - 15 existing contract regressions;
  - 5 controlled boundaries for Relation relevance, repetition, targeted
    rereading, decomposed current-gap focus, and STOP with no justified route;
  - 8 Controller steps extracted from historical real diagnostics.
- Model-visible inputs: `model_inputs/controller_cases_v1.jsonl`
- Reviewer-only Gold:
  `review_only/controller_gold_v1.jsonl`
- Manifest: `suite_manifest.json`
- Historical diagnostic source:
  `tests/fixtures/controller_gold_5_diagnostics_v0.json`

The 28 cases are a Prompt-independent development set. Historical v0.6 runs are
not results for the current Prompt.

The legacy source runner keeps each synthetic `ControllerInput` beside its expected
action, but fresh generation conversations must use the split model-visible
file in this directory. They must not read `review_only/`, the inline source
runner, or the historical diagnostic source.

The historical real steps were previously exposed to models and are regression
material, not an unseen holdout or a current-v0.5 performance estimate. The
Q14 Exact Anchor step remains outside this Controller suite because it is an
Environment `AUTO_READ_EXACT` decision with no `ControllerInput`.

K20 covers STOP while budget remains but all distinct searches and visible
routes are exhausted. There is deliberately no zero-budget case: budget
exhaustion terminates in the Environment rather than requesting another
Controller decision.

Still needed before formal testing:

- add real ControllerInput snapshots after Checker updates EvidenceMemory;
- reserve real trajectories from unseen documents for holdout.

Controller evaluation should use fixed snapshots before live-loop testing. A
live loop alone changes the input whenever an upstream Prompt changes and makes
failures difficult to attribute.
