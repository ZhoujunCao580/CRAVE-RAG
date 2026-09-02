# Fresh Linux GPU Server

This guide covers reproducible setup after cloning the public CRAVE-RAG
repository. It does not assume that model weights, PDFs, MinerU outputs, or
Teacher trajectories are stored in Git.

## 1. Base image

Use Ubuntu with a working NVIDIA driver and Python 3.11 or 3.12. For initial
prompt evaluation and 8B-class QLoRA experiments, one 24 GB NVIDIA GPU is the
minimum practical target. Larger contexts, visual models, or bigger batches may
require 48 GB or more.

Verify the machine before installing Python packages:

```bash
nvidia-smi
python3 --version
git --version
```

## 2. Clone and bootstrap

```bash
git clone https://github.com/ZhoujunCao580/crave-rag.git
cd crave-rag
export CRAVE_PROFILE=train
bash scripts/bootstrap_server.sh
```

Profiles:

- `core`: SoftDoc conversion, validation, and CPU tests.
- `eval`: core plus PyTorch, Transformers, and the visual-retrieval stack
  (`sentence_transformers`) for dense/visual evaluation.
- `train`: eval plus Accelerate, PEFT, Datasets, and Linux bitsandbytes.

The bootstrap script is intentionally fail-fast. `softdoc doctor` reports a
missing dependency, unsupported Python version, or missing CUDA before a long
training run starts. It also runs `pip check`, the test suite, Prompt
export/dry-run checks, and SFT-data validation. Some visual regression tests
reference external development corpora, so those assets must be installed before
running the complete suite.

The bootstrap does **not** install a model server, download model weights, or
copy datasets. Those steps depend on the GPU provider and dataset license.

## 3. Add the non-Git artifacts

Git contains the code and small controlled fixtures, not the real PDFs or their
generated assets. Copy or mount the following before a real run:

- one serialized SoftDoc directory per document;
- its referenced page/element images below that document directory;
- optional local dense-model weights and embedding cache;
- a model endpoint and its text/visual model weights.

Build and audit an external-dataset manifest before paying for model inference.
For MMLongBench-Doc, see `docs/EXTERNAL_DATASETS.md`. The audit verifies source
files, SoftDocs, visual assets, page counts, and question-document mappings and
fails explicitly if the external corpus is absent.

Before a benchmark run, also validate the frozen evaluation protocol and create
an immutable experiment snapshot as described in `docs/EVALUATION_PROTOCOL.md`:

```bash
python scripts/freeze_experiment.py \
  --protocol configs/evaluation/mmlongbench_doc_reference_v0_1.json \
  --validate-only
```

MMLongBench-Doc is currently a development/reference benchmark, not a clean
holdout. A run report must retain that distinction.

For a single manually transferred SoftDoc, also validate it directly:

```bash
softdoc validate /workspace/data/softdoc/example
```

The current production adapter speaks the Ollama `/api/chat` protocol. A plain
vLLM OpenAI-compatible endpoint is not interchangeable yet; it needs a separate
backend adapter. Start with an Ollama-compatible endpoint and verify its model
names independently before running CRAVE-RAG.

## 4. Prompt evaluation

Ollama is optional. If an Ollama-compatible endpoint is available:

```bash
python scripts/evaluate_prompts.py \
  --component all_text \
  --text-model qwen3:8b \
  --base-url http://127.0.0.1:11434
```

Visual evaluation additionally needs the ignored local corpus and its config:

```bash
python scripts/evaluate_prompts.py \
  --component visual_reader \
  --visual-model qwen3-vl:4b \
  --visual-config path/to/config.json \
  --visual-corpus path/to/corpus
```

All outputs go under `.runlogs/` by default and include a prompt manifest.

Run the complete loop after copying or generating a serialized SoftDoc:

```bash
softdoc run-model /workspace/data/softdoc/example \
  --question "What evidence supports the reported conclusion?" \
  --output /workspace/runs/example \
  --text-model qwen3:8b \
  --visual-model qwen3-vl:4b \
  --dense \
  --dense-device cuda \
  --embedding-cache /workspace/cache/e5 \
  --visual-search-index /workspace/cache/visual-retrieval/colsmol-500m \
  --visual-search-device cuda
```

The default transport is Ollama-compatible HTTP. The core runner itself uses
injectable backend protocols, so a later vLLM or hosted-API adapter does not
require changing the reading state machine.

First run one question. Only after its `run_manifest.json`, stage-call JSONL,
Reader limitations, and Evidence transitions look sensible should you start a
batch.

For an isolated batch, export a Gold-free JSONL file from an audited dataset
manifest. A hand-written UTF-8 JSONL file remains supported for small ad hoc
experiments:

```json
{"case_id":"Q305","question_id":"benchmark:Q305","document_dir":"data/softdoc/doc_a","question":"What value is reported?"}
{"case_id":"Q462","question_id":"benchmark:Q462","document_dir":"data/softdoc/doc_b","question":"What conclusion follows from the table?"}
```

Then run:

```bash
python scripts/run_model_batch.py \
  --cases /workspace/data/pilot_cases.jsonl \
  --path-root /workspace \
  --output-root /workspace/runs/pilot-01 \
  --text-model qwen3:8b \
  --visual-model qwen3-vl:4b \
  --dense \
  --dense-device cuda \
  --visual-search-index /workspace/cache/visual-retrieval/colsmol-500m \
  --case-timeout 3600
```

Each case invokes the canonical `softdoc run-model` entry point in an isolated
process. One malformed model response is recorded as a failed case but does not
discard later cases. `batch_manifest.json` is rewritten after every case and
the command returns a nonzero exit code if any case failed. Per-case process
logs are stored under `_logs/` instead of bloating the manifest. Use a new
output directory for every rerun; existing nonempty outputs are never
overwritten.

## 5. Teacher data and SFT

Teacher records use UTF-8 JSONL. Each row binds one model component to the
exact frozen prompt version, input text, and expected output. Validate data
without downloading a model:

```bash
python scripts/train_sft.py \
  --data configs/training/sft_example.jsonl \
  --validate-only
```

Reviewed `ModelPipelineRun` directories are exported separately for Controller
and Checker supervision:

```bash
softdoc teacher-data init-review /workspace/runs/pilot-01/Q305
softdoc teacher-data init-checker-review /workspace/runs/pilot-01/Q305

# After human/Teacher review changes pending labels to accepted/rejected:
softdoc teacher-data export-controller \
  /workspace/runs/pilot-01/Q305 \
  --output /workspace/data/controller_dataset
softdoc teacher-data export-checker \
  /workspace/runs/pilot-01/Q305 \
  --output /workspace/data/checker_dataset
```

The repository's training script consumes the component-specific
`controller_sft.jsonl` or `checker_sft.jsonl`. The accompanying
`*_sft_messages.jsonl` and `dataset_info.json` are the OpenAI-messages form for
LLaMA-Factory-style tooling.

Run a QLoRA SFT experiment only after the review/export validators pass:

```bash
python scripts/train_sft.py \
  --data /workspace/data/controller_dataset/controller_sft.jsonl \
  --model Qwen/Qwen3-8B \
  --output /workspace/runs/controller-qlora \
  --qlora --bf16 --gradient-checkpointing
```

This repository currently provides a generic, version-bound LoRA/QLoRA SFT
entry. It does **not** yet provide a finished RL reward, production Teacher
dataset, or claim that the sample record is sufficient for training. Those are
research artifacts to be created after trajectory collection and evaluation.

## 6. What must be transferred separately

The following are deliberately ignored by Git because of size, licensing, or
privacy:

- PDF datasets and MinerU output directories;
- generated SoftDoc corpora and embedding caches;
- model checkpoints and LoRA adapters;
- API keys and `.env` files;
- experiment outputs under `.runlogs/`.

Use object storage, a mounted data volume, or an approved dataset download
script for these artifacts. Do not commit credentials or copyrighted corpora.

## 7. Recommended first-server checklist

1. Run `nvidia-smi` and install the `train` dependency profile.
2. Confirm the hermetic repository suite passes with `python -m pytest -q`.
   Real-corpus semantic evaluation remains a separate, explicit external-data
   operation.
3. Build and run `softdoc datasets audit` for the installed corpus. Transfer
   one SoftDoc and also run `softdoc validate` while diagnosing individual
   document failures.
4. Confirm the Ollama-compatible endpoint and both model names respond.
5. Run exactly one `softdoc run-model` question and inspect all stage logs.
6. Run the small batch with a fresh output directory.
7. Review Controller and Checker decisions before exporting any SFT rows.
8. Train a tiny smoke adapter before committing to a long QLoRA run.
