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
  native visual Dense retrieval, SearchSessions, fixed mixed candidate batches,
  Element-level route deduplication, and deterministic CandidatePreviews.
- Optional conservative Planner: zero SubQuestions when no decomposition is
  needed, or a validated DAG for genuine decomposition.
- Visual Reader and question-directed full-page Visual Scan contracts.
- Retrieval-only visual identities and on-demand Controller-facing visual
  descriptions with asset and Prompt provenance. Descriptions do not enter
  text retrieval or Evidence.
- Dual-route Table retrieval plus a Multimodal Table Reader that combines
  HTML/cells, crop pixels, table units, page/source identity, and confirmed
  cross-page header inheritance.
- Append-only ObservationStore and atomic Evidence Checker deltas.
- Target-switch Evidence rechecking plus bounded, Checker-only Observation
  Recall for relevant historical claims that never entered Evidence.
- Derived Controller state, validated action union, and explicit incomplete
  `STOP`.
- Evidence-only Answerer contract and prompt.
- An injectable Reading Environment that exercises the complete state loop.
- Ollama/OpenAI-compatible backends for Planner, Controller, visual retrieval,
  Readers, Checker, Answerer, and Visual Scan, plus a persistent concurrent
  end-to-end batch runner.
- Unified prompt registry, evaluation launcher, Linux bootstrap, and generic
  LoRA/QLoRA SFT entrypoint.
- Complete model-run audit packets, thin per-decision Teacher reviews, and
  separate strict Controller/Checker SFT exporters with Prompt/version lineage.
- A completed Controller-only SFT round using 500+ reviewed decisions, improving
  the internal Test from 50.00%/49.44% to 56.45%/55.32% Accuracy/F1.

## Not yet claimed as complete

- Production-quality Reader evaluation and server-native model adapters beyond
  the current Ollama v0 backend.
- A production-scale Teacher trajectory corpus or externally released adapter.
- A canonical preference/DPO dataset, DPO trainer, or RL training pipeline.
- Deferred planning and broader semantic/source recall beyond the bounded
  target-switch Observation Recall implementation.
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
  visual_scan.py            scoped whole-page visual scan contract
  table_reading.py          Multimodal Table Reader contract and prompt renderer
  reading_state.py          reads, Observations, Evidence, and action trace
  checking_prompt.py        Checker version and compatibility loader
  controller.py             Controller input and action contracts
  controller_prompt.py      Controller version and compatibility loader
  reading_environment.py    executable reading-loop orchestration
  model_backends.py         Ollama Reader/Checker/Answerer adapters
  model_runner.py           Planner-to-Answerer runner and audit artifacts
  teacher_data.py           thin Controller/Checker reviews and separate SFT exports
  training_data.py          version-bound model-facing SFT records
  answering.py              Answerer contract and user-prompt renderer
  prompt_registry.py        single prompt discovery/version entrypoint
  evaluation_protocol.py    frozen metrics and immutable experiment snapshots
```

Supporting areas:

```text
tests/                      core contract and runtime regression tests
scripts/run_model_batch.py  persistent closed-loop evaluation runner
scripts/train_sft.py        validated LoRA/QLoRA SFT entry
scripts/evaluate_controller_sft_offline.py  policy-level adapter comparison
configs/training/           frozen document split and training-data examples
constraints/                reproducible CI dependency pins
docs/                       current design, setup, and research boundaries
```

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
- [Evaluation Protocol](EVALUATION_PROTOCOL.md): canonical development/reference
  scoring boundary, metric definitions, and immutable experiment IDs.
- [Research Positioning](RESEARCH_POSITIONING.md): current research hypothesis
  and novelty boundary.
- [TODO](TODO.md): unresolved decisions and experiments. A TODO is not an
  implemented feature.
- [Post-training Guide](POST_TRAINING.md): durable SFT/DPO artifact boundaries,
  review/export workflow, metrics, and resume checklist.

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

Production retrieval experiments may additionally pass `--dense` and
`--visual-search-index <INDEX_DIR>`. That combination uses a five-candidate
mixed view: three text-RRF candidates and two visual candidates, with stable
per-batch mixing and Element-level deduplication. The quota and source ranks
remain backend provenance and are not exposed to the Controller.

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
