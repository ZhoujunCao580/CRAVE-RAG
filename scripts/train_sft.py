"""Minimal LoRA/QLoRA SFT entry for CRAVE-RAG Teacher records.

This is intentionally not an RL pipeline.  ``--validate-only`` is model-free
and is used by CI/server smoke tests.  Actual training lazily imports the GPU
stack so core users do not need heavyweight dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from softdoc.training_data import SFTExample, load_sft_jsonl


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


def _encode_example(example: SFTExample, tokenizer: Any, max_length: int) -> EncodedExample:
    prompt_ids = list(
        tokenizer.apply_chat_template(
            example.messages(), tokenize=True, add_generation_prompt=True
        )
    )
    target_text = example.target_text() + (tokenizer.eos_token or "")
    target_ids = list(tokenizer(target_text, add_special_tokens=False)["input_ids"])
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
            AutoModelForCausalLM,
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

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
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
            target_modules="all-linear",
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

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        bf16=bool(args.bf16 and torch.cuda.is_available()),
        fp16=bool(not args.bf16 and torch.cuda.is_available()),
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
    )
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
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", type=Path, default=Path(".runlogs/sft"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    examples = load_sft_jsonl(args.data)
    component_counts: dict[str, int] = {}
    for example in examples:
        key = example.component.value
        component_counts[key] = component_counts.get(key, 0) + 1
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
