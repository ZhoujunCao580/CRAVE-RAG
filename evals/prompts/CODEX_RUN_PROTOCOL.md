# Fresh Codex Prompt Evaluation Protocol

This is a component-level development evaluation. It is not a final benchmark
and does not reproduce an API call's exact system/developer hierarchy.

## Two isolated roles

1. **Generator:** sees one Prompt, its exact output Schema, model inputs, and
   only the visual assets referenced by those inputs. It never sees Gold.
2. **Reviewer:** runs only after generation and sees raw outputs plus
   `review_only/`. It must not rewrite the Generator's answers.

Use 5--10 cases per fresh Generator conversation to limit cross-case context.
Re-run important failures as one case in another fresh conversation. Formal
A/B evaluation later uses independent API requests with fixed parameters.

## Current component packets

| Component | Prompt | Output Schema | Model inputs |
| --- | --- | --- | --- |
| Planner | `src/softdoc/prompts/planner_v0_20.txt` | `evals/prompts/schemas/planner_output.schema.json` | `evals/prompts/planner/model_inputs/planner_cases_v1.jsonl` |
| Controller | `src/softdoc/prompts/controller_policy_v0_7.txt` | `evals/prompts/schemas/controller_output.schema.json` | `evals/prompts/controller/model_inputs/controller_cases_v1.jsonl` |
| Visual Reader | `src/softdoc/prompts/visual_reader_v0_4.txt` | `evals/prompts/schemas/visual_reader_output.schema.json` | `evals/prompts/visual_reader/model_inputs/visual_reader_cases_v1.jsonl` |
| Checker | `src/softdoc/prompts/checker_v1_9.txt` | `evals/prompts/schemas/checker_output.schema.json` | `evals/prompts/checker/model_inputs/checker_cases_v1.jsonl` |
| Answerer | `src/softdoc/prompts/answerer_v0_7.txt` | `evals/prompts/schemas/answerer_output.schema.json` | `evals/prompts/answerer/model_inputs/answerer_cases_v1.jsonl` |

Regenerate the exact Schemas after an approved contract change:

```powershell
python -m scripts.export_prompt_schemas
```

## Generator instruction

Append one row range and the three allowed paths from the table above.

```text
This is a read-only component Prompt evaluation. Evaluate only the named
component and Prompt version.

You may read only:
1. the named canonical Prompt;
2. the named output JSON Schema;
3. the named model-input JSONL rows;
4. for Visual Reader only, images explicitly referenced by those rows.

Do not read review_only, archived Prompts, previous outputs, source evaluators,
other Prompt candidates, or any Gold/scoring material. Do not modify project
files and do not retry or repair a model answer.

For every case, write one JSONL record with exactly:
- case_id
- raw_output: the complete unmodified component answer

Do not score semantic correctness and do not propose Prompt changes.
```

Store Generator results outside the evaluation packet:

```text
.runlogs/prompt_eval/<component>/<prompt_version>/<run_id>/
  run_manifest.json
  raw_outputs.jsonl
```

The Generator conversation may write only to that run directory. If strict
read-only operation is preferred, copy its chat output there afterwards.

## Program validation and review

Do not ask the Generator to self-certify Schema validity. Parse `raw_output`
against the exported Schema in a separate deterministic step. Then give a
Reviewer only:

```powershell
python -m scripts.validate_prompt_run `
  --component planner `
  --input .runlogs/prompt_eval/planner/planner-v0.20/<run_id>/raw_outputs.jsonl `
  --output .runlogs/prompt_eval/planner/planner-v0.20/<run_id>
```

The validator writes `schema_validation.jsonl` and `schema_summary.json` next
to the raw outputs. Then give a Reviewer only:

- the same Prompt and Schema;
- `raw_outputs.jsonl`;
- the matching `review_only/*_gold_v1.jsonl`.

The Reviewer records Schema validity, semantic result, failure category, and
whether Gold itself is ambiguous. Change a Prompt only for a recurring,
general failure pattern; create a new Prompt version and rerun the entire dev
suite. Keep a later unseen real holdout untouched during Prompt development.
