# CRAVE-RAG: Controller-guided Reading and Action Via Evidence Gaps

> Start from a clue, follow useful structure, and stop only when the evidence is sufficient.

Licensed under the [Apache License 2.0](LICENSE).

CRAVE-RAG is a research prototype for question answering over long, visually rich PDFs. Its core is an **agentic reading Controller** that keeps working from the current evidence gap instead of treating retrieved items as a fixed final context. Retrieval only provides a place to begin reading.

The Controller maintains a current question and evidence gap, then chooses how to continue reading. Each reading move produces source-linked Observations. An independent Evidence Checker decides which observations become Evidence, whether the current gap has been closed, and what remains unresolved. The updated gap is returned to the Controller, which chooses another move. The loop ends only when the Evidence is sufficient.

![CRAVE-RAG overview](docs/assets/crave-rag-overview.svg)

## Core Design

This loop gives the system four important properties:

- **Active reading:** it can keep reading from a useful clue instead of repeatedly rebuilding a fixed final context.
- **Structured navigation:** page structure and document relations become explicit reading opportunities rather than automatically expanded answer context.
- **Independent evidence control:** the Controller explores, while the Checker separately decides whether an Observation deserves to enter Evidence.
- **Traceable decisions:** every reading action, Observation, and accepted Evidence item remains connected to its source, providing the inputs needed for deterministic citation materialization.

## Soft Document Structure

PDFs are represented through a parser-neutral intermediate structure:

```text
Document
|-- Pages
|   `-- Heading / Paragraph / Table / Figure / Chart / Caption / ...
|-- Sections
`-- Relations
```

The structure preserves page layout, element bounding boxes, reading order, visual assets, semantic hierarchy, provenance, and confirmed or candidate document relations. Relations guide navigation but never become Evidence by themselves.

## Current Status

The repository currently provides the SoftDoc representation, MinerU adaptation pipeline, deterministic document relations, spatial navigation, exact/sparse/dense retrieval, candidate previews, search sessions, frozen contracts for planning, reading, evidence checking, and answering, and an executable stateful reading loop with explicit incomplete termination.

The next research stage is integrating production model-backed Readers and policies, materializing final source citations, and evaluating whether evidence-gap-driven reading improves answer quality and reading efficiency.

## Quick Start

```bash
conda env create -f environment.yml
conda activate multimodal_pdf_rag
python -m pip install -e .
python -m pytest -q
```

The CPU test dependency set used by CI is pinned in `constraints/ci.txt`; model and GPU dependencies remain optional.

```bash
softdoc parse-mineru <MINERU_OUTPUT_DIR> --output <SOFTDOC_OUTPUT_DIR>
softdoc validate <SOFTDOC_OUTPUT_DIR>
```

With representative SoftDocs available locally, the model-free replay audit
exercises Search, Exact routing, reading, Checker deltas, Relation navigation,
and answering with scripted backends:

```bash
python scripts/audit_reading_environment_v0.py --softdoc-root <SOFTDOC_ROOT>
```

This replay checks interfaces and state transitions; it is not a model-quality
benchmark.

## Prompts and Evaluations

All five model-facing prompts are discoverable through one versioned registry:

```bash
softdoc prompts list
softdoc prompts show controller
softdoc prompts show planner --question "How did revenue change?"
softdoc prompts export --output .runlogs/prompts
```

The unified evaluation launcher records the exact prompt versions and hashes
used by every run. Text-only evaluations require a running Ollama-compatible
endpoint; the dry run is model-free.

```bash
python scripts/evaluate_prompts.py --dry-run
python scripts/evaluate_prompts.py --component all_text --text-model qwen3:8b
```

## Fresh Linux GPU Server

Start from an Ubuntu GPU image with a working NVIDIA driver, then clone this
repository and run:

```bash
export CRAVE_PROFILE=train
bash scripts/bootstrap_server.sh
```

This creates an isolated environment, installs runtime and LoRA/QLoRA
dependencies, runs the test suite, exports the prompt manifest, and validates
the training-data contract. Actual SFT additionally requires a model checkpoint
and a Teacher JSONL dataset; neither large model weights nor research data are
committed to Git. See [Server Setup](docs/SERVER_SETUP.md).

See [Project Guide](docs/PROJECT_GUIDE.md), [Model Contracts](docs/MODEL_CONTRACTS.md), [Architecture](docs/ARCHITECTURE.md), and [TODO](docs/TODO.md) for the current implementation boundary, complete JSON interfaces, and open research questions.
