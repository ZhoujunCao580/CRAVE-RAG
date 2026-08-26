# CRAVE-RAG Project Guide

This guide is the short map of the repository. It deliberately avoids
duplicating frozen prompts, model contracts, experiment history, or open
research notes.

## Goal

CRAVE-RAG is a research prototype for multimodal long-document QA. Retrieval
finds a place to start; a Controller continues reading from the current
Evidence gap; Readers produce grounded Observations; an independent Checker
controls Evidence; and the Answerer responds only from accepted Evidence.

The guiding principle is:

> Start from a clue, follow useful structure, and stop only when the evidence
> is sufficient.

## Current implemented boundary

- Parser-neutral SoftDoc models for Documents, Pages, Sections, Elements,
  bounding boxes, provenance, and typed Relations.
- MinerU adapter plus deterministic, auditable document passes.
- Stable serialization, validation, in-memory document access, overlays, and
  spatial queries.
- Exact Anchor lookup, SearchUnits, BM25, multilingual-E5 dense retrieval,
  SearchSessions, and deterministic CandidatePreviews.
- Conservative Planner contracts and prompt.
- Visual Reader request/result contracts and prompt.
- Append-only ObservationStore and atomic Evidence Checker deltas.
- Derived Controller state, validated action union, and explicit incomplete
  `STOP`.
- Evidence-only Answerer contract and prompt.
- An injectable Reading Environment that exercises the complete state loop.
- Unified prompt registry, evaluation launcher, Linux bootstrap, and generic
  LoRA/QLoRA SFT entrypoint.

## Not yet claimed as complete

- Production Text/Table/Page/Visual Reader backends.
- A trained Controller policy or a production Teacher trajectory corpus.
- Deferred planning and Observation Recall.
- Citation materialization in final user-facing output.
- Full-dataset answer-quality evaluation and action-value ablations.
- RL rewards or a complete RL training pipeline.

## Source layout

```text
src/softdoc/
  models.py                 SoftDoc core models
  mineru_adapter.py         MinerU -> raw SoftDoc conversion
  pipeline.py               deterministic pass orchestration
  relations.py              deterministic relation builders
  store.py / spatial.py     document access and spatial navigation
  retrieval.py              SearchUnits, Exact, BM25, dense retrieval
  search_session.py         persistent candidate batches and previews
  planning/                 Planner models, prompt, and backend interface
  visual_reading.py         Visual Reader contract and prompt
  reading_state.py          reads, Observations, Evidence, and action trace
  checking_prompt.py        Evidence Checker prompt
  controller.py             Controller input and action contracts
  controller_prompt.py      Controller prompt
  reading_environment.py    executable reading-loop orchestration
  answering.py              Answerer contract and prompt
  prompt_registry.py        single prompt discovery/version entrypoint
```

Supporting areas:

```text
tests/                      regression tests; keep with behavior changes
scripts/evaluate_*.py       reproducible component evaluations
scripts/train_sft.py        generic validated LoRA/QLoRA SFT entry
configs/training/           example training-data contract
constraints/                reproducible CI dependency pins
docs/                       current design, setup, and research boundaries
```

## Canonical references

- [Architecture](ARCHITECTURE.md): system loop and safety boundaries.
- [Model Contracts](MODEL_CONTRACTS.md): complete Planner, Reader, Checker,
  Controller, and Answerer JSON examples.
- [Server Setup](SERVER_SETUP.md): fresh Linux GPU environment and training
  entrypoint.
- [Research Positioning](RESEARCH_POSITIONING.md): current research hypothesis
  and novelty boundary.
- [TODO](TODO.md): unresolved decisions and experiments. A TODO is not an
  implemented feature.

Frozen prompt text is not copied into documentation. Inspect the executable
source of truth instead:

```bash
softdoc prompts list
softdoc prompts show planner --question "How did revenue change?"
softdoc prompts show visual_reader
softdoc prompts show checker
softdoc prompts show controller
softdoc prompts show answerer
```

## Local validation

```bash
python -m pip install -e .
python -m pytest -q
python scripts/evaluate_prompts.py --dry-run
softdoc doctor --profile core
```

Real model evaluations write ignored artifacts under `.runlogs/`. Generated
corpora, PDFs, model weights, and caches are intentionally excluded from Git.
See [Server Setup](SERVER_SETUP.md) before moving to a GPU machine.

## Change discipline

- Keep parser-specific payloads out of core models.
- Keep model clients behind injectable interfaces and mock them in unit tests.
- Preserve stable IDs and cross-store reference validation.
- Do not silently promote Candidate Relations to confirmed facts.
- Do not treat retrieval results, previews, Relations, or raw Observations as
  Evidence.
- Version prompt changes and evaluate them instead of patching individual
  dataset examples.
- Update this guide only when repository boundaries change; put experiments
  and unresolved ideas in `TODO.md` rather than duplicating them here.
