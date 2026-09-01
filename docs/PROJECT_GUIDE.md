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
- Optional conservative Planner: zero SubQuestions when no decomposition is
  needed, or a validated DAG for genuine decomposition.
- Visual Reader request/result contracts and prompt.
- Retrieval-only visual search identities with asset and Prompt provenance.
- Append-only ObservationStore and atomic Evidence Checker deltas.
- Derived Controller state, validated action union, and explicit incomplete
  `STOP`.
- Evidence-only Answerer contract and prompt.
- An injectable Reading Environment that exercises the complete state loop.
- Ollama-backed Visual Retrieval, Planner, Controller, Visual Reader, Checker,
  and Answerer
  adapters plus an auditable end-to-end model runner; text and structured
  tables use the deterministic Reader before any visual fallback.
- Unified prompt registry, evaluation launcher, Linux bootstrap, and generic
  LoRA/QLoRA SFT entrypoint.
- Complete model-run audit packets, thin per-decision Teacher reviews, and
  separate strict Controller/Checker SFT exporters with Prompt/version lineage.

## Not yet claimed as complete

- Production-quality Reader evaluation and server-native model adapters beyond
  the current Ollama v0 backend.
- A trained Controller policy or a production Teacher trajectory corpus.
- A reviewed, diverse local Teacher corpus and preference/RL data.
- Deferred planning and Observation Recall.
- Citation materialization in final user-facing output.
- Full-dataset answer-quality evaluation and action-value ablations.
- RL rewards or a complete RL training pipeline.

## Source layout

```text
src/softdoc/
  models.py                 SoftDoc core models
  adapters/mineru.py        MinerU -> raw SoftDoc conversion
  pipeline.py               deterministic pass orchestration
  relations.py              deterministic relation builders
  store.py / spatial.py     document access and spatial navigation
  retrieval/                SearchUnits, Exact, BM25, Dense, sessions/previews
  planning/                 Planner models, renderer, and backend interface
  prompts/                  current versioned prompt text for all components
  visual_reading.py         Visual Reader contract and user-prompt renderer
  visual_retrieval.py       Offline visual search identity and provenance
  reading_state.py          reads, Observations, Evidence, and action trace
  checking_prompt.py        Checker version and compatibility loader
  controller.py             Controller input and action contracts
  controller_prompt.py      Controller version and compatibility loader
  reading_environment.py    executable reading-loop orchestration
  model_backends.py         Ollama Reader/Checker/Answerer adapters
  model_runner.py           Planner-to-Answerer runner and audit artifacts
  teacher_data.py           thin Controller/Checker reviews and separate SFT exports
  answering.py              Answerer contract and user-prompt renderer
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

`tests/fixtures/controller_gold_5_diagnostics_v0.json` is a historical
diagnostic captured with Controller prompt v0.2 and action v0.1. Its version
headers describe the model run that produced the observations; they are not
the current canonical Prompt or action versions.

## Canonical references

- [Prompt workspace](../src/softdoc/prompts/README.md): editable versioned
  model instructions and prompt-change discipline.
- [Architecture](ARCHITECTURE.md): system loop and safety boundaries.
- [Model Contracts](MODEL_CONTRACTS.md): complete Planner, Reader, Checker,
  Controller, and Answerer JSON examples.
- [Server Setup](SERVER_SETUP.md): fresh Linux GPU environment and training
  entrypoint.
- [External Datasets](EXTERNAL_DATASETS.md): native adapters, portable
  manifests, fail-fast corpus auditing, and Gold-free batch export.
- [Research Positioning](RESEARCH_POSITIONING.md): current research hypothesis
  and novelty boundary.
- [TODO](TODO.md): unresolved decisions and experiments. A TODO is not an
  implemented feature.
- [Teacher Loop Handoff](TEACHER_LOOP_HANDOFF.md): artifact boundaries,
  review/export workflow, and quality rules for trajectory distillation.

Frozen prompt text is not copied into documentation. Edit or inspect the
versioned files in `src/softdoc/prompts/`, and use the executable registry to
render or export the exact runtime form:

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

## End-to-end model run

```bash
softdoc run-model <SOFTDOC_OUTPUT_DIR> \
  --question "<ROOT_QUESTION>" \
  --output .runlogs/<RUN_NAME> \
  --text-model qwen3:8b \
  --visual-model qwen3-vl:4b
```

The output packet contains:

```text
run_manifest.json
planner.json
controller_calls.jsonl
reader_calls.jsonl
checker_calls.jsonl
answerer_calls.jsonl
reading_run.json
```

The JSONL rows preserve the validated input and output of each executed module
call. `reading_run.json` is the canonical end state, including SearchSessions,
Observations, Evidence, actions, diagnostics, and the optional final answer.

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
