# Visual Reader suite inventory

- Canonical Prompt: `src/softdoc/prompts/visual_reader_v0_4.txt`
- Registry version: `visual-reader-v0.4`
- Current source: `configs/visual_reader_v0_probe_25.json`
- Runner: `scripts/run_visual_reader_probe.py`
- Frozen suite: 35 cases
  - 25 existing real-image development regressions;
  - 10 controlled boundary cases with deterministic local image assets.
- Model-visible inputs:
  `model_inputs/visual_reader_cases_v1.jsonl`
- Reviewer-only Gold:
  `review_only/visual_reader_gold_v1.jsonl`
- Manifest: `suite_manifest.json`
- Controlled asset generator: `build_controlled_assets.py`

The legacy probe config stores `problem`, image selection metadata, and `expected`
in the same file. Fresh generation conversations must use the split files in
this directory instead. Each model-visible `request` follows the current
`VisualReadRequest` contract, and its ordered `visual_inputs` reference the
actual local image assets.

The ten controlled cases isolate duplication, exhaustive listing, unreadable
or cropped targets, partial readability, irrelevant inputs, cross-image
comparison, visible conflict, and absent requested facts. They are diagnostic
regressions and must not be reported as real-data performance.

Before a fresh Codex run, model-visible cases should contain only the local
problem, ordered `visual_inputs`, and resolvable image paths. Expected facts,
readability judgments, duplication judgments, and scoring notes belong in
reviewer-only files.

Still needed before formal testing:

- review whether each crop/page actually contains the intended visible fact;
- add fixed real packets produced by the current runtime input builder;
- create a genuinely unseen holdout after the dev suite is frozen.
