# Checker suite inventory

- Canonical Prompt: `src/softdoc/prompts/checker_v2_1.txt`
- Registry version: `checker-v2.1`

The model-facing v2.1 output omits `used_for_evidence`. The runtime derives
that audit field from the atomically resulting Evidence `observation_ids`.
It also supports a state-only Root-finalization call after all planned
SubQuestions are satisfied.
- Current source and runner: `scripts/evaluate_checker_mock.py`
- Materializer: `scripts/materialize_checker_suite.py`
- Frozen synthetic suite: 27 state-transition cases
  - 20 existing contract regressions;
  - 7 controlled boundaries for cross-target Observations, Root evaluation
    after plan completion (including no-new-read finalization), Evidence
    removal, remaining-gap quality, and limitation-only reading.
- Model-visible inputs: `model_inputs/checker_cases_v1.jsonl`
- Reviewer-only Gold: `review_only/checker_gold_v1.jsonl`
- Manifest: `suite_manifest.json`

Regenerate the split files without calling a model:

```powershell
python -m scripts.materialize_checker_suite
```

The source runner still contains each `EvidenceCheckInput` beside its expected
state change. Fresh generation conversations must use only the materialized
model-visible file, the canonical Prompt, and the output Schema. They must not
read `review_only/` or the source runner.

Gold keeps used flags, add/replace/remove counts, target/root status,
remaining-gap requirements, and next-target labels separate from model input.
Evidence and gap wording should be reviewed semantically rather than requiring
byte-for-byte matches.

Still needed before formal testing:

- add fixed Reader-to-Checker packets from real documents;
- reserve an unseen real trajectory holdout.
