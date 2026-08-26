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
- `eval`: core plus PyTorch and Transformers for dense/model evaluation.
- `train`: eval plus Accelerate, PEFT, Datasets, and Linux bitsandbytes.

The bootstrap script is intentionally fail-fast. `softdoc doctor` reports a
missing dependency, unsupported Python version, or missing CUDA before a long
training run starts.

## 3. Prompt evaluation

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

## 4. Teacher data and SFT

Teacher records use UTF-8 JSONL. Each row binds one model component to the
exact frozen prompt version, input text, and expected output. Validate data
without downloading a model:

```bash
python scripts/train_sft.py \
  --data configs/training/sft_example.jsonl \
  --validate-only
```

Run a QLoRA SFT experiment after replacing the example with a real Teacher
dataset:

```bash
python scripts/train_sft.py \
  --data /workspace/data/controller_teacher.jsonl \
  --model Qwen/Qwen3-8B \
  --output /workspace/runs/controller-qlora \
  --qlora --bf16 --gradient-checkpointing
```

This repository currently provides a generic, version-bound LoRA/QLoRA SFT
entry. It does **not** yet provide a finished RL reward, production Teacher
dataset, or claim that the sample record is sufficient for training. Those are
research artifacts to be created after trajectory collection and evaluation.

## 5. What must be transferred separately

The following are deliberately ignored by Git because of size, licensing, or
privacy:

- PDF datasets and MinerU output directories;
- generated SoftDoc corpora and embedding caches;
- model checkpoints and LoRA adapters;
- API keys and `.env` files;
- experiment outputs under `.runlogs/`.

Use object storage, a mounted data volume, or an approved dataset download
script for these artifacts. Do not commit credentials or copyrighted corpora.
