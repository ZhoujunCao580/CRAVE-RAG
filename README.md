# CRAVE-RAG: Controller-guided Reading and Action Via Evidence Gaps

> Find a promising entry point, read the document like a human, and judge whether each finding helps answer the question.

Licensed under the [Apache License 2.0](LICENSE).

CRAVE-RAG is a research prototype for question answering over long, visually rich PDFs. Its core is an **agentic reading Controller** that keeps working from the current evidence gap instead of treating retrieved items as a fixed final context. Retrieval only provides a place to begin reading.

The Controller maintains a current question and evidence gap, then chooses how to continue reading. Each reading move produces source-linked Observations. An independent Evidence Checker decides which observations become Evidence, whether the current gap has been closed, and what remains unresolved. The updated gap is returned to the Controller, which chooses another move. The loop ends only when the Evidence is sufficient.

![CRAVE-RAG overview](docs/assets/crave-rag-overview.svg)

## Core Design

CRAVE-RAG separates *finding*, *reading*, and *believing* a source instead of
placing one retrieved context directly in front of an answer model.

- **Three-route retrieval:** BM25, text Dense, and native visual Dense retrieval
  produce one mixed candidate batch. Exact anchors are resolved before ranking;
  duplicate routes to the same `element_id` occupy only one candidate slot.
- **Preview/read separation:** text snippets, HTML-derived `TablePreview`s, and
  short VLM visual descriptions help the Controller choose what to open. A
  visual description is generated only after a batch is frozen, is cached by
  source identity, and never becomes searchable text or Evidence.
- **Multimodal table reading:** every Table may be recalled through text and
  visual routes, but `READ_SOURCE` resolves both to one Table identity. The
  Table Reader receives structured HTML/cells, the original crop, inherited
  headers for confirmed continuations, page/source identity, and the current
  local problem. It emits grounded Observations plus explicit limitations.
- **Action-controlled reading:** the Planner proposes a small question DAG;
  the Controller selects Search, Read, page-context, or relation-navigation
  actions from only currently visible IDs. Readers never navigate on their own.
- **Evidence-gated completion:** Readers create source-linked Observations; an
  independent Checker atomically updates Evidence Memory and the active gap.
  Candidate previews, relations, and unaccepted Observations cannot reach the
  Answerer.
- **Long-horizon state:** search cursors, candidate batches, reads,
  Observations, Evidence, limitations, and actions are persisted. Target-switch
  Evidence rechecks and bounded Observation Recall can reuse earlier facts
  without creating fake reads.
- **Post-training boundary:** Controller supervision is exported as versioned
  `ControllerInput -> Action` records with prompt hashes and source-run lineage.
  Checker supervision stays separate; future preference pairs must compare
  actions under the identical visible state.

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
The latest Controller-SFT run scored **56.45% content accuracy / 55.32%
generalized F1**, rescuing ten previously incorrect questions while two
previously correct questions regressed (net +8). Planner, Reader, Checker, and
Answerer remain on the base model; this is a Controller-only training result.

| System | Accuracy | Generalized F1 |
| --- | ---: | ---: |
| ColBERTv2 | 30.56% | 20.43% |
| M3DocRAG | 38.21% | 37.52% |
| G2-Reader | 46.96% | 45.29% |
| CRAVE-RAG, prompt-only | 50.00% | 49.44% |
| **CRAVE-RAG, Controller SFT (500+ decisions)** | **56.45%** | **55.32%** |

The published baselines and the internal CRAVE-RAG split are shown as context,
not as a leaderboard claim: the evaluation subsets and answer-equivalence
protocols are not identical. Both reported CRAVE-RAG metrics use the project's
frozen content-equivalence scoring protocol; accuracy is 70/124 correct.

### Runtime and Efficiency

The frozen Test run used one A100 SXM 80 GB GPU, a persistent vLLM backend, and
four concurrent workers. SoftDoc conversion and offline index construction were
completed before timing. Controller SFT replaces only the Controller adapter;
the serving topology and online QA pipeline remain the same.

| Online QA metric | CRAVE-RAG |
| --- | ---: |
| Mean end-to-end latency | 132.0 s/question |
| Median end-to-end latency | 120.3 s/question |
| P95 end-to-end latency | 333.3 s/question |
| Four-worker wall-clock throughput | 35.35 s/question (about 102 questions/hour) |

Latency and throughput are different measurements: concurrent requests overlap,
so the wall-clock seconds per completed question are lower than the latency of
one trajectory. These figures describe online QA over prebuilt document
artifacts; they do not include PDF parsing or retrieval-index construction.

The cited comparison systems do not publish a directly comparable
MMLongBench-Doc end-to-end latency under the same hardware and serving setup.
[M3DocRAG](https://openaccess.thecvf.com/content/ICCV2025W/Findings/papers/Cho_M3DocVQA_Multi-modal_Multi-page_Multi-document_Understanding_ICCVW_2025_paper.pdf)
reports a retrieval-only result of roughly 20 s/query with exact search and
under 2 s/query with an IVF index over 40K pages; answer generation is excluded.
[G2-Reader](https://arxiv.org/html/2601.22055) reports offline Content Graph
construction averaging 233.6 s/document for its full method and 130.4
s/document for its Lite variant; these are not question-answering latencies.
[ColBERTv2](https://aclanthology.org/2022.naacl-main.272/) reports roughly 100
ms/query retrieval on its IR benchmarks using a Titan V, but that measurement
excludes PDF parsing, OCR, and downstream answer generation. For that reason,
this README does not rank the systems by incompatible speed measurements.

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
[`src/softdoc/prompts/`](src/softdoc/prompts/README.md). The registry is the
runtime discovery and hashing interface for active model-facing prompts:

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
committed to Git. Use the [external-dataset manifest and
auditor](docs/EXTERNAL_DATASETS.md) before batch execution. See [Server
Setup](docs/SERVER_SETUP.md).

See [Project Guide](docs/PROJECT_GUIDE.md), [Post-training Guide](docs/POST_TRAINING.md), [Model Contracts](docs/MODEL_CONTRACTS.md), [Architecture](docs/ARCHITECTURE.md), and [TODO](docs/TODO.md) for the current implementation boundary, complete JSON interfaces, training-data rules, and open research questions.
