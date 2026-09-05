# Development checkpoint: reading control and evidence recovery

Date: 2026-09-05

This checkpoint records the coherent local implementation batch prepared after
the first `frozen-baseline64-v0` failure audit. It deliberately excludes local
run logs, dataset artifacts, temporary launch scripts, and case-specific Gold
information.

## Frozen model contracts

- Planner: `planner-v0.21`
- Controller: `controller-policy-v0.10`
- Controller input/action: `controller-input-v0.4` / `controller-action-v0.3`
- Checker: `checker-v2.1`
- Answerer: `answerer-v0.8`
- Reading environment: `reading-environment-v0.4`
- Visual retrieval descriptor: `visual-retrieval-v0.1`
- Visual reader: `visual-reader-v0.4`

Superseded canonical prompt files are removed instead of retained as an in-repo
prompt archive.

## Implemented changes

### Planner semantic closure

Planner decompositions must preserve the requested quantity, unit, denominator,
population or entity scope, time scope, and statistical meaning. A reported
metric is searched by name unless the question itself specifies the operands;
the Planner must not invent an external formula or assume how the document
represents a value.

### Exact-anchor coverage

Exact lookup now handles a broader deterministic grammar for Pages, Slides,
Figures, Tables, Sections, cover/first/last pages, positional Figures/Tables,
and bounded compound ranges. It distinguishes unique, ambiguous, unresolved,
ignored answer-format examples, and ranges too large to expand safely. Large
range search/inventory semantics remain explicitly out of scope.

### Candidate and relation previews

Table candidates receive a query-conditioned structured preview with inherited
headers for confirmed cross-page fragments, a matched row, adjacent rows, and a
bounded list of other row labels. Visual candidate previews can use cached
visual retrieval summaries. Candidate and relation paths preserve their source
type and provenance.

### Controller page context

Controller inputs expose opened-source locations with canonical physical page
numbers and optional printed page labels. Canonical `contains` hierarchy edges
are hidden from model-visible navigation. The new
`READ_PAGE_CONTEXT(base_page_id, offset=-1/0/+1)` action reads the current or an
adjacent physical page; offset zero can supply both the whole-page image and the
most recently opened element crop on that page.

### Checker provenance

The model-facing `EvidenceCheckDecision` no longer asks the model to duplicate
`used_for_evidence`. The runtime atomically applies the Evidence delta and then
derives that audit field from the Observation IDs referenced by the resulting
EvidenceMemory. Checker evaluation and SFT export now use the same model-facing
contract.

### State-only Root finalization

After the final planned SubQuestion becomes satisfied, the runtime targets the
Root, copies the original Root question as the initial gap, and immediately
asks the Checker to judge the complete accepted Evidence without fabricating a
read or consuming a Controller action. A ready result proceeds to the Answerer;
an incomplete result supplies the next specific Root gap.

### Deterministic incomplete output and resume

Runs that still terminate as `budget_exhausted` or `stopped_incomplete` emit the
canonical existing answer shape:

```json
{"answer":"Not answerable","used_evidence_ids":[]}
```

Internal status and diagnostics remain unchanged, so route failure is not
mistaken for a Checker-verified unanswerable document. Budget checkpoints with
this fallback remain resumable; a later ready result replaces it with a normal
Answerer result. Any substantive answer still requires Evidence IDs.

## Evaluation assets and compatibility

- Planner, Controller, Checker, and Answerer schemas and manifests are updated
  to their current contracts.
- Checker includes a state-only Root-finalization evaluation case.
- Stored runtime feedback keeps the materialized provenance field, while
  Checker SFT targets omit it.
- Backward-readable defaults are retained where persisted Controller input
  artifacts predate physical page metadata.

## Local verification

The complete CPU test suite passed before this checkpoint was committed:

```text
454 passed
```

Server-side model behavior is not claimed by this checkpoint. The next server
work must run targeted A/B or replay tests for budget continuation, visual
summary integration, TablePreview selection, Exact anchors, page-context
reading, Checker provenance, Planner semantic closure, Root finalization, and
fallback replacement.

## Deliberately pending

- multi-target Evidence semantics;
- lightweight Observation recall and target-switch rechecking;
- table-level unit propagation into Reader observations;
- deterministic first/last-page metadata in Reader input;
- document/range inventory and coverage-complete counting;
- controlled recovery from Checker truncation or other invalid structured
  output.
