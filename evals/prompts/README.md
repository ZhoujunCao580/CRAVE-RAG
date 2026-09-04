# CRAVE-RAG Prompt Evaluation Hub

This directory is the single entry point for locating and preparing Prompt
evaluation suites. It does not replace `tests/`: pytest covers contracts and
runtime behavior, while this directory covers model behavior.

No evaluator should expose Gold labels to the model. For every component, the
final suite must separate:

- `model_inputs/`: the Prompt and case inputs the model may read;
- `review_only/`: expected behavior, scoring notes, and human adjudication;
- `.runlogs/`: raw outputs and reports, which remain outside this directory.

All current component suites are split this way. Source runners may still keep
case construction and scoring logic together for local automation, but a fresh
Generator conversation must use only the materialized `model_inputs/` packet.

| Component | Current Prompt | Current case inventory | Folder |
|---|---|---|---|
| Planner | `planner-v0.20` | 49 development cases with separate model input and reviewer-only Gold | [planner](planner/README.md) |
| Visual Reader | `visual-reader-v0.4` | 35 cases: 25 real-image regressions and 10 controlled boundaries | [visual_reader](visual_reader/README.md) |
| Checker | `checker-v1.9` | 26 synthetic state-transition cases, including limitation-only reading | [checker](checker/README.md) |
| Controller | `controller-policy-v0.8` | 28 historical v0.6 split cases: 20 synthetic decisions and 8 historical real trajectory steps | [controller](controller/README.md) |
| Answerer | `answerer-v0.8` | 24 English cases: 14 regressions and 10 new boundaries | [answerer](answerer/README.md) |
| End to end | multiple | real trajectory replay is not frozen yet | [integration](integration/README.md) |

## Preparation and evaluation order

1. Finish the human reading pass over every canonical Prompt.
2. Freeze component-level dev inputs and reviewer-only labels before using model
   results to tune a Prompt.
3. Reserve a new holdout that has not been exposed to a model or used for Prompt
   decisions.
4. Evaluate fixed component packets first so failures can be attributed to one
   Prompt.
5. Then replay realistic trajectories in runtime order:
   `Planner -> Controller -> Reader -> Checker -> Controller ... -> Answerer`.
6. Run end-to-end evaluation only after the component candidates are frozen.

For the recommended fresh-conversation workflow, see
[CODEX_RUN_PROTOCOL.md](CODEX_RUN_PROTOCOL.md).

## Current limitations

- Every component suite now separates model-visible inputs from reviewer-only
  Gold. Source runners may still keep cases and expected behavior together, so
  fresh generation conversations must use only the materialized input files.
- The Planner's previous seven-case holdout was already included in a 49-case
  run. It is now regression material, not an unseen holdout.
- Reader, Checker, Controller, and Answerer still need approved real trajectory
  packets in addition to their synthetic or probe cases.
- This inventory does not freeze, relabel, or promote any suite.
