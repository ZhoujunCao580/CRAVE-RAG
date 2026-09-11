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

The repository currently provides the SoftDoc representation, MinerU adaptation pipeline, deterministic document relations, spatial navigation, exact/sparse/dense/visual retrieval, candidate previews, search sessions, multimodal table reading, frozen contracts for planning, reading, evidence checking, and answering, plus an executable stateful reading loop with explicit incomplete termination.

The first Controller QLoRA/SFT iteration is complete. After a 78-decision
pipeline pilot, the current Controller dataset was expanded to more than 500
reviewed SFT decisions. On the frozen internal 124-question Test split, the
prompt-only system scored **50.00% content accuracy / 49.44% generalized F1**.
The latest Controller-SFT run scored **56.45% content accuracy / approximately
55.32% generalized F1**, rescuing ten previously incorrect questions while two
previously correct questions regressed (net +8). Planner, Reader, Checker, and
Answerer remain on the base model; this is a Controller-only training result.

| System | Evaluation scope | Accuracy | Generalized F1 |
| --- | --- | ---: | ---: |
| ColBERTv2 | Published full MMLongBench-Doc | 30.56% | 20.43% |
| M3DocRAG | Published full MMLongBench-Doc | 38.21% | 37.52% |
| G2-Reader | Published full MMLongBench-Doc | 46.96% | 45.29% |
| CRAVE-RAG, prompt-only | Internal frozen Test, 124 questions | 50.00% | 49.44% |
| **CRAVE-RAG, Controller SFT (500+ decisions)** | **Internal frozen Test, 124 questions** | **56.45%** | **~55.32%** |

The published baselines and the internal CRAVE-RAG split are shown as context,
not as a leaderboard claim: the evaluation subsets and answer-equivalence
protocols are not identical. The SFT F1 remains approximate until the complete
post-SFT status counts are persisted; accuracy is exact from 70/124 correct.
The next research stage is expanding verified online Teacher rollouts, measuring
paired rescue/regression behavior on document-isolated development data, and
separately deciding whether Checker supervision is warranted.

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

With the required Ollama text and visual models available, run a low-level
single-case component smoke on one serialized SoftDoc:

```bash
softdoc run-model <SOFTDOC_OUTPUT_DIR> \
  --question "What evidence explains the reported change?" \
  --output .runlogs/example \
  --text-model qwen3:8b \
  --visual-model qwen3-vl:4b
```

`softdoc run-model` deliberately remains a lightweight component/debug entry
point. It does not provide the frozen-batch visual descriptor cache or
on-demand preview enrichment, so it must not be used to report the canonical
full-architecture baseline.

Add `--dense` to combine BM25 with multilingual-E5 Dense retrieval. Dense
dependencies are optional, so the default command remains lightweight and uses
Exact Anchor lookup plus BM25. Every run writes separate Planner, Controller,
Reader, Checker, and Answerer records alongside the complete reading state.

When a completed visual embedding index is available, add
`--visual-search-index <INDEX_DIR>`. This enables the frozen mixed policy: each
five-card batch draws three candidates from BM25/Dense weighted RRF and two
from visual retrieval, deduplicates them, and presents one stable mixed list.
The Controller sees candidate content, not route quotas or rank metadata.
Visual summaries are attached only after a candidate batch is frozen. They are
Controller-facing preview text and never enter the BM25/Dense SearchUnit corpus.

Use the fail-fast named profile for every canonical baseline, including a
one-question end-to-end smoke:

```bash
python scripts/run_model_batch.py \
  --runtime-profile crave-baseline-v1 \
  --execution-mode persistent \
  --cases <GOLD_FREE_CASES_JSONL> \
  --path-root <PATH_ROOT> \
  --output-root <NEW_OUTPUT_DIR> \
  --inference-backend vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --text-model Qwen/Qwen3.5-27B \
  --visual-model Qwen/Qwen3.5-27B \
  --dense \
  --embedding-cache <DENSE_CACHE_DIR> \
  --visual-search-index <COMPLETED_VISUAL_INDEX_DIR> \
  --visual-descriptor-cache <DESCRIPTOR_CACHE_JSONL> \
  --visual-descriptor-on-demand \
  --multimodal-table-reader
```

The profile aborts before any question runs if a required module or completed
visual index is missing.

## Prompts and Evaluations

Editable, versioned prompt text lives together under
[`src/softdoc/prompts/`](src/softdoc/prompts/README.md). The registry remains
the runtime discovery and hashing interface for the seven active model-facing
prompts. The retired Coverage Checker remains available only as an explicitly
exported `legacy_inactive` historical prompt:

```bash
softdoc prompts list
softdoc prompts list --include-inactive
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
committed to Git. Use the [external-dataset manifest and
auditor](docs/EXTERNAL_DATASETS.md) before batch execution. See [Server
Setup](docs/SERVER_SETUP.md).

See [Project Guide](docs/PROJECT_GUIDE.md), [Server Experiment Plan](docs/SERVER_EXPERIMENT_PLAN_CN.md), [Model Contracts](docs/MODEL_CONTRACTS.md), [Architecture](docs/ARCHITECTURE.md), and [TODO](docs/TODO.md) for the current implementation boundary, complete JSON interfaces, validation order, and open research questions.
