# Teacher Loop Handoff

## Project context

CRAVE-RAG is a multimodal long-document QA research prototype. A parser-neutral
SoftDoc represents Documents, Pages, typed Elements, Sections, bounding boxes,
assets, and typed Relations. Retrieval provides resumable search sessions and
small CandidatePreviews. The online reading loop is:

```text
Planner (optional decomposition)
  -> Exact/Search entry
  -> Controller chooses one reading action
  -> Reader produces source-linked Observations and limitations
  -> Checker atomically updates EvidenceMemory and the current gap
  -> repeat while incomplete
  -> Answerer uses accepted Evidence only
```

Search results, previews, Relations, and Observations are not Evidence. Confirmed
Relations are normal navigation handles; candidate Relations are uncertain
investigation opportunities. The Controller never promotes either into factual
Evidence. The program validates IDs, state transitions, action execution, and
Checker deltas.

The current canonical Prompt files are the six `.txt` files directly under
`src/softdoc/prompts/`. Previous Prompt revisions are available through Git
history and must not be recreated in a working-tree archive.

## What the Teacher Loop is for

The first Teacher Loop should primarily produce supervision for the **Reading
Controller**. The research question is whether a policy can choose useful
search, reading, typed-relation, candidate-relation, adjacent-page, and stopping
actions from the current Evidence gap while avoiding wasted reads.

Planner, Reader, Checker, and Answerer are still part of every recorded
trajectory because they create the state transition and determine whether an
action made progress. They are not mixed into the same Controller SFT dataset.
Component-specific training data may be created later from separately reviewed
component failures.

## Three deliberately separate artifacts

1. **`ModelPipelineRun`** is the canonical raw episode. Its run directory
   contains the plan, complete `ReadingRunResult`, model-call records, executed
   `ActionTrace`, Observations, Checker deltas, Evidence, search state, and
   terminal result. Controller calls are linked to executed actions by
   `action_id`; automatic Exact Anchor reads remain distinguishable from model
   decisions.
2. **Thin review files** sit beside that run. `teacher_review.json` marks each
   Controller decision `pending`, `accepted`, or `rejected`.
   `checker_review.json` independently reviews each Checker call and can hold a
   corrected delta without changing the raw run. Neither file duplicates model
   inputs, Evidence, or Observations.
3. **Component-specific SFT exports** are derived training data, with one row per
   explicitly accepted decision. `controller_sft.jsonl` preserves CRAVE-RAG's
   internal provenance-bearing `SFTExample`; `controller_sft_messages.jsonl`
   materializes the LLaMA-Factory/OpenAI `messages` form:

```text
system: canonical Controller Prompt
user: validated ControllerInput at step t
assistant: one validated ControllerAction
```

The review file is therefore not the SFT dataset. The strict exporter joins the
raw run with its review, validates the action against its recorded
`ControllerInput`, verifies the frozen Controller Prompt version and hash, and
then emits both representations plus `dataset_manifest.json` and a
LLaMA-Factory `dataset_info.json`. The two JSONL files contain the same examples
in the same order; only the model-facing messages file is passed to
LLaMA-Factory.

The Checker exporter performs the analogous join for
`EvidenceCheckInput -> EvidenceCheckResult` and additionally runs the ordinary
atomic Evidence update validator before accepting an example. It writes a
separate `checker_sft_messages.jsonl`; Checker targets are never mixed into the
Controller dataset.

## Recommended production loop

1. Select a real question and its SoftDoc without exposing answer Gold to the
   Controller.
2. Run the real environment. A strong Teacher chooses each Controller action.
3. Execute the action rather than fabricating its result.
4. Use the actual Reader and Checker contracts. If a weak component corrupts
   the state, record the failure and let a reviewer repair or reject the episode;
   do not silently rewrite history.
5. Review each Controller decision against the exact state it saw. Reject a
   bad decision with a short note and optionally mark the first corrupted
   action. Earlier valid decisions in an otherwise rejected episode may still
   be accepted individually.
6. Export only reviewed decisions. The exporter rejects stale Prompt bindings,
   invisible IDs, invalid action payloads, duplicate examples, pending reviews,
   and mismatched run/review coverage.
7. Split by document before training. Gold answer/evidence must remain outside
   all model-visible state; that is a generation protocol and evaluation rule,
   not a redundant per-episode Boolean claim.

The Teacher is not merely an answer generator. It labels grounded reading
decisions. Several actions may be reasonable from one state; v0 records whether
the executed action is acceptable, without prematurely claiming a complete set
of all acceptable alternatives. Preference pairs can be introduced later when
the project can review those alternatives reliably.

## Distillation, diversity, and quality

Teacher distillation is useful here: a strong model creates decision traces that
a smaller local model can imitate. Raw Teacher output is not automatically
training data. Distillation consists of generation, deterministic validation,
environment replay, filtering, deduplication, and human review of ambiguous or
high-impact cases.

Build diversity across dimensions that change the reading policy:

- document type: paper, report, manual, slide deck, brochure, dense financial
  filing, and visually rich pages;
- question structure: direct Root, parallel SubQuestions, dependent DAG, and
  Root-level synthesis after planned questions;
- modality: paragraph, table, chart, figure, page context, and mixed evidence;
- entry route: exact anchor, lexical/dense search, later candidate batches, and
  returning to an existing search session;
- navigation: confirmed relation, candidate relation investigated or rejected,
  and previous/next page;
- failure: irrelevant preview, unreadable visual, misleading relation, no new
  Evidence, conflict, incomplete stop, and budget pressure;
- trajectory length and action cost.

Do not manufacture diversity through paraphrases alone. Balance action and
failure types so that the student does not learn `READ_SOURCE` on every state.
Include hard negative states where a tempting preview or relation does not fill
the current gap.

Quality gates for every accepted trajectory:

- all model outputs pass their Pydantic contracts;
- every action references only handles visible in its ControllerInput;
- the recorded action was actually executed by the environment;
- accepted Evidence is grounded in stored Observations and ReadRecords;
- the selected action has plausible progress or justified exploration value;
- the final answer is supported by the final Evidence set;
- no reviewer-only Gold appears in model-visible state;
- train/dev/test splits do not share a document;
- near-duplicate states and boilerplate trajectories are removed.

## Scale and training order

Start small:

1. **Schema pilot:** 20-30 fully reviewed trajectories to find logging and
   replay defects.
2. **First SFT corpus:** roughly 100-200 high-quality trajectories, often
   yielding 500-1,500 Controller decision examples. Measure action validity,
   Evidence progress, cost, and end-to-end success before scaling.
3. Expand only according to observed error categories. Dataset size is not a
   substitute for grounded, diverse decisions.

Use SFT first. Preference training such as DPO becomes useful only after the
same ControllerInput has defensible preferred and rejected actions. RL/GRPO
should wait until the executable environment, cost model, and reward signals
are stable enough that the policy cannot exploit bookkeeping errors.

## Implemented v0 workflow

After producing a run directory with `softdoc run-model`:

```bash
softdoc teacher-data init-review path/to/run
```

Edit the generated `teacher_review.json`, finalize its episode and step labels,
then export one or more reviewed runs:

```bash
softdoc teacher-data export-controller path/to/run_a path/to/run_b \
  --output path/to/controller_dataset
softdoc teacher-data init-checker-review path/to/run
softdoc teacher-data export-checker path/to/run_a path/to/run_b \
  --output path/to/checker_dataset
python scripts/train_sft.py \
  --data path/to/controller_dataset/controller_sft.jsonl --validate-only
```

For LLaMA-Factory, copy or point its dataset directory at
`controller_sft_messages.jsonl` and the generated `dataset_info.json`, then use
the registered dataset name `crave_controller_sft`. LLaMA-Factory describes
this `role`/`content` structure as the OpenAI-format special case of ShareGPT;
the selected model template still controls how those messages become model
tokens.

This v0 intentionally does not implement alternative-action labels, preference
pairs, rewards, DPO, RL, automatic semantic review, or a claim of deterministic
full-environment replay. Those should be added only when pilot trajectories
show a concrete need.
