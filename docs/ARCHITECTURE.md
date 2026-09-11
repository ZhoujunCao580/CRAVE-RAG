# CRAVE-RAG Architecture

CRAVE-RAG stands for **Controller-guided Reading and Action Via Evidence
Gaps**. It is a multimodal long-document QA system in which retrieval proposes
where to read, while accepted Evidence—not retrieval rank—controls completion.

![CRAVE-RAG overview](assets/crave-rag-overview.svg)

## Offline document layer

MinerU output is normalized into a parser-neutral SoftDoc:

```text
Document
├─ Page
│  └─ Heading / Paragraph / Table / Figure / Chart / Caption / Footnote
├─ Section hierarchy
└─ confirmed and candidate Relations
```

Stable document, page, element, region, and visual-asset identities preserve
layout, reading order, provenance, and source paths. Confirmed relations support
navigation; candidate relations remain hypotheses.

Tables keep structured HTML/cells and visual crops. Confirmed cross-page table
fragments may inherit headers through `continued_on`; ambiguous fragments are
not merged automatically.

## Retrieval and candidate presentation

The online retrieval stack has three ranked routes:

1. BM25 over text-bearing SearchUnits;
2. text Dense retrieval over the same semantic units;
3. native visual Dense retrieval over visual assets, including Tables with a
   usable crop.

Exact Page/Figure/Table anchors are resolved before ordinary ranking. Text
scores are fused, then a fixed mixed batch presents text and visual candidates.
All routes deduplicate by stable source/`element_id`, so a Table retrieved by
both text and image occupies one slot and is read once.

Candidate previews are decision aids only. Paragraphs use matched text;
Tables prefer an HTML-derived TablePreview; visual candidates receive a short
VLM description only after their batch is frozen. That description is cached
for presentation and never inserted into BM25/Dense text retrieval, an
Observation, or Evidence.

## Agentic reading loop

```text
Question -> Planner -> current target + active gap
                         |
                         v
          mixed candidate batch + visible relations
                         |
                         v
                    Controller
        SEARCH / READ_SOURCE / READ_PAGE_CONTEXT /
                    FOLLOW_RELATION
                         |
                         v
        Text Reader / Visual Reader / Multimodal Table Reader
                         |
                         v
              source-linked Observations
                         |
                         v
                 Evidence Checker
         atomic Evidence delta + next active gap
                         |
              incomplete | ready
                  loop    | Answerer
```

The Planner may keep the Root intact or create a small validated DAG. The
Controller sees only legal actions and current visible IDs. Readers interpret a
selected source for the Controller's `local_problem`; they do not navigate or
admit Evidence.

The Multimodal Table Reader receives aligned structured cells/HTML and the
original crop when available. It uses text for exact strings and numbers when
alignment is clear and the image for layout, headers, units, and visible
context. Ambiguity becomes a limitation instead of a fabricated cell mapping.

The Checker receives the current target, new Observations, limitations, and
complete Evidence Memory. Its delta is validated and applied atomically. A
target switch may trigger a state-only Evidence recheck plus at most three
relevant unaccepted historical Observations. No fake read or Controller action
is created.

## Persistence and completion

Every run persists Planner, Controller, Reader, Checker, and Answerer inputs and
outputs; candidate batches; SearchSession cursors; action trace; Observation
Store; Evidence Memory; latency; and diagnostics. Stable identities allow a
source to be traced from final Evidence back to the original page or region.

The Answerer sees accepted Evidence only. `ready` is program-controlled from
Checker state. Explicit STOP or budget exhaustion remains incomplete and emits
`Not answerable`; it cannot silently promote weak Evidence.

Question-directed Visual Scan is a separate whole-page VLM path for exhaustive
visual questions over a resolved document/page/section scope. The earlier
MinerU-inventory Coverage route remains disabled compatibility code and is not
part of current runs.

## Post-training boundary

Controller SFT uses the exact online `ControllerInput` as the user message and
a validated legal `ControllerAction` as the target, with prompt hashes and
source-run lineage. Checker data is exported and trained separately. Preference
training must compare chosen and rejected actions under the identical visible
state; future Environment outcomes may be stored as audit metadata but cannot
leak into model input.

The current Controller-only SFT result is **56.45% Accuracy / 55.32%
generalized F1**, up from **50.00% / 49.44%** on the same internal Test.
