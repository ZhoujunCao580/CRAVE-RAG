#!/usr/bin/env bash
set -euo pipefail

PROFILE="${CRAVE_PROFILE:-eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

case "${PROFILE}" in
  core)
    python -m pip install -e ".[dev]"
    ;;
  eval)
    python -m pip install -e ".[dev,dense]"
    ;;
  train)
    python -m pip install -e ".[dev,dense,train]"
    ;;
  *)
    echo "CRAVE_PROFILE must be core, eval, or train" >&2
    exit 2
    ;;
esac

python -m pip check
softdoc doctor --profile "${PROFILE}"
python -m pytest -q
softdoc prompts export --output .runlogs/prompts
python scripts/evaluate_prompts.py --dry-run --output-root .runlogs/prompt_eval_dry_run
python scripts/train_sft.py --data configs/training/sft_example.jsonl --validate-only

echo "CRAVE-RAG ${PROFILE} environment is ready."
