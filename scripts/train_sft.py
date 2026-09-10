"""Minimal LoRA/QLoRA SFT entry for CRAVE-RAG Teacher records.

This is intentionally not an RL pipeline.  ``--validate-only`` is model-free
and is used by CI/server smoke tests.  Actual training lazily imports the GPU
stack so core users do not need heavyweight dependencies.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from numbers import Integral
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from softdoc.training_data import SFTExample, load_sft_jsonl
from softdoc.prompt_registry import PromptComponent


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


def _extract_input_ids(value: Any, *, field: str) -> list[int]:
    """Normalize tokenizer outputs across Transformers 4.x and 5.x.

    ``apply_chat_template(tokenize=True)`` historically returned a flat list,
    while newer tokenizers may return a ``BatchEncoding``.  Iterating the
    latter yields string keys (for example ``"input_ids"``), which must never
    reach the tensor collator.
    """

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(f"{field} tokenizer output has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = list(value[0])
    if not isinstance(value, list) or not all(
        isinstance(item, Integral) for item in value
    ):
        raise TypeError(f"{field} input_ids must be a flat integer sequence")
    return [int(item) for item in value]


def _encode_example(example: SFTExample, tokenizer: Any, max_length: int) -> EncodedExample:
    prompt_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            example.messages(), tokenize=True, add_generation_prompt=True
        ),
        field="prompt",
    )
    target_text = example.target_text() + (tokenizer.eos_token or "")
    target_ids = _extract_input_ids(
        tokenizer(target_text, add_special_tokens=False), field="target"
    )
    if len(target_ids) >= max_length:
        raise ValueError(
            f"{example.example_id}: target alone has {len(target_ids)} tokens, "
            f"which does not fit max_length={max_length}"
        )
    prompt_ids = prompt_ids[-(max_length - len(target_ids)) :]
    return EncodedExample(
        input_ids=prompt_ids + target_ids,
        labels=[-100] * len(prompt_ids) + target_ids,
    )


def _train(args: argparse.Namespace, examples: list[SFTExample]) -> None:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            'Training dependencies are missing. Install with: pip install -e ".[dense,train]"'
        ) from exc

    if args.qlora and not torch.cuda.is_available():
        raise SystemExit("QLoRA requires a CUDA GPU in this v0 training entry")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.qlora:
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            ),
            device_map="auto",
        )
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16

    model_config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model_class = args.model_class
    if model_class == "auto":
        model_class = (
            "image_text_to_text"
            if getattr(model_config, "vision_config", None) is not None
            else "causal_lm"
        )
    model_loader = (
        AutoModelForImageTextToText
        if model_class == "image_text_to_text"
        else AutoModelForCausalLM
    )
    model = model_loader.from_pretrained(args.model, **model_kwargs)
    if args.qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=(
                "all-linear"
                if args.lora_target_modules == "all-linear"
                else [
                    item.strip()
                    for item in args.lora_target_modules.split(",")
                    if item.strip()
                ]
            ),
        ),
    )
    if args.gradient_checkpointing and not args.qlora:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    encoded = [_encode_example(item, tokenizer, args.max_length) for item in examples]

    class Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(encoded)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            item = encoded[index]
            return {"input_ids": item.input_ids, "labels": item.labels}

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for item in batch:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            labels.append(item["labels"] + [-100] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    training_kwargs: dict[str, Any] = dict(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy=("steps" if args.max_steps > 0 else "epoch"),
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim=args.optim,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        bf16=bool(args.bf16 and torch.cuda.is_available()),
        fp16=bool(not args.bf16 and torch.cuda.is_available()),
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
    )
    # Transformers 5 renamed ``warmup_ratio`` to a float-capable
    # ``warmup_steps``.  Preserve the CLI contract across both generations:
    # a value below 1 remains a ratio in either implementation.
    training_parameters = inspect.signature(TrainingArguments).parameters
    if "warmup_ratio" in training_parameters:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
    else:
        training_kwargs["warmup_steps"] = args.warmup_ratio
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset(),
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(str(args.output / "adapter"))
    tokenizer.save_pretrained(str(args.output / "adapter"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or run LoRA/QLoRA SFT")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--require-component",
        choices=tuple(item.value for item in PromptComponent),
        help="Fail if any record belongs to another component.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", type=Path, default=Path(".runlogs/sft"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--model-class",
        choices=("auto", "causal_lm", "image_text_to_text"),
        default="auto",
        help=(
            "Model AutoClass. auto selects ImageTextToText when the config has "
            "a vision_config, otherwise CausalLM."
        ),
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values override --epochs for a time-bounded pilot.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("linear", "cosine", "constant", "constant_with_warmup"),
        default="linear",
    )
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="all-linear",
        help=(
            "Comma-separated module-name suffixes, or all-linear. For the "
            "multimodal Qwen3.5 Controller pilot, explicitly name language "
            "projection modules so the vision tower is not adapted."
        ),
    )
    args = parser.parse_args()

    examples = load_sft_jsonl(args.data)
    component_counts: dict[str, int] = {}
    for example in examples:
        key = example.component.value
        component_counts[key] = component_counts.get(key, 0) + 1
    if args.require_component is not None and set(component_counts) != {
        args.require_component
    }:
        raise SystemExit(
            "Dataset component mismatch: required "
            f"{args.require_component!r}, found {sorted(component_counts)!r}"
        )
    print(
        json.dumps(
            {
                "valid": True,
                "examples": len(examples),
                "components": component_counts,
                "mode": "validate_only" if args.validate_only else "train",
            },
            ensure_ascii=False,
        )
    )
    if args.validate_only:
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    _train(args, examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
