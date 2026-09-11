# Controller Post-training Guide

This is the durable handoff for resuming CRAVE-RAG Controller SFT or starting
preference training. It records interfaces and decision rules, not historical
run logs.

## Current checkpoint

- Prompt-only internal Test: **50.00% Accuracy / 49.44% generalized F1**.
- Controller SFT with 500+ reviewed decisions: **56.45% Accuracy / 55.32%
  generalized F1**.
- The paired change was ten rescued questions and two regressions, for a net
  gain of eight correct answers.
- Only the Controller was trained. Planner, Readers, Checker, Answerer, and
  retrieval remained on their base implementations.

The 124-question Test is an internal frozen split scored with the project's
content-equivalence protocol. It is not directly comparable to a complete
public MMLongBench-Doc leaderboard run.

## Assets that must survive outside Git

Large and sensitive artifacts are intentionally ignored by Git. Before another
training round, restore or locate:

1. the reviewed Controller SFT JSONL and its dataset manifest;
2. the trained adapter checkpoint and training configuration;
3. source PDFs, serialized SoftDocs, text/visual indexes, and descriptor cache
   needed for closed-loop evaluation;
4. any immutable raw trajectories used to derive new labels.

Git retains the code, prompts, frozen document split, schemas, example records,
and validators. It does not retain model weights, caches, or experiment runs.

## SFT data contract

One Controller training example is:

```text
frozen Controller system prompt
+ exact serialized ControllerInput visible at one decision point
-> one validated ControllerAction
```

Use `softdoc teacher-data` to create thin reviews over immutable
`ModelPipelineRun` packets and export accepted Controller and Checker records
separately. Every exported dataset binds the component, prompt version, prompt
hash, source pipeline version, source session, and example IDs.

Keep only actions that are valid under the exact visible state. Do not expose a
Gold source ID, a future candidate batch, or an Environment result that the
Student could not have observed. Do not turn retrieval, Reader, or Checker
failures into Controller labels.

Validate and train with:

```bash
python scripts/train_sft.py --data <CONTROLLER_SFT_JSONL> --validate-only
python scripts/train_sft.py \
  --data <CONTROLLER_SFT_JSONL> \
  --model <BASE_MODEL> \
  --output <NEW_ADAPTER_DIR> \
  --qlora
```

Use `scripts/evaluate_controller_sft_offline.py` for contract and action-policy
metrics, then use `scripts/run_model_batch.py` for closed-loop evaluation.

## What to measure

Offline Controller evaluation should report JSON validity, action validity,
visible-ID validity, Teacher exact/route agreement, duplicate-action rate,
action distribution, and latency. Closed-loop evaluation should additionally
report answer Accuracy/F1, answerable completion, Evidence sufficiency,
retrieval recall, action/model-call counts, repeated reads/searches, and cost.

Choose checkpoints on document-isolated Dev data. Test results are for final
paired reporting, not for selecting labels, prompts, or checkpoints.

## Preference data for a future DPO round

A valid Controller preference row must compare two actions under the *same*
Controller input and prompt version:

```text
state:    exact ControllerInput
chosen:   reviewed better legal action
rejected: Student action or another legal but worse action
```

Useful rejected actions include repeated searches without state change,
reading a less relevant visible source, stopping while a supported route
remains, or choosing a higher-cost route with no additional Evidence gain.
Invalid or invisible-ID actions may be retained for a separate contract-loss
dataset, but should not dominate semantic preference training.

Prefer pairs backed by a deterministic replay or real Environment continuation.
Store the immediate Observation/Evidence delta, eventual completion outcome,
extra actions, and cost as audit metadata; do not place future outcomes in the
model-visible prompt. Avoid constructing preferences when the difference was
caused by retrieval nondeterminism, Reader error, or Checker error.

The repository does not yet implement a DPO trainer or a canonical preference
schema. Those are explicit TODO items rather than silently implied features.

## Resume checklist

1. Restore external artifacts and run `softdoc datasets audit`.
2. Run `python -m pytest -q` and `softdoc doctor --profile train`.
3. Verify prompt hashes in the training manifest against the active registry.
4. Validate the SFT or preference dataset without loading a model.
5. Run a small document-isolated offline evaluation.
6. Select on Dev, then run one frozen closed-loop comparison.
