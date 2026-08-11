"""Injectable text encoders and the concrete multilingual-E5 adapter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from softdoc.retrieval.models import (
    DenseDevice,
    EncoderFingerprint,
    EncoderInputType,
    EncoderTokenSpan,
)


class DenseEncoderError(RuntimeError):
    """Base error for explicit Dense encoder failures."""


class DenseModelUnavailableError(DenseEncoderError):
    """Raised when the configured model or its dependencies are unavailable."""


class DenseInputTooLongError(DenseEncoderError):
    """Raised instead of silently truncating an encoder input."""


class TextEncoder(Protocol):
    """Software boundary implemented by real and test encoders."""

    @property
    def fingerprint(self) -> EncoderFingerprint: ...

    def prepared_token_count(
        self,
        text: str,
        input_type: EncoderInputType,
    ) -> int: ...

    def content_token_spans(self, text: str) -> list[EncoderTokenSpan]: ...

    def encode(
        self,
        texts: Sequence[str],
        input_type: EncoderInputType,
    ) -> list[list[float]]: ...


def e5_prefixed_text(text: str, input_type: EncoderInputType) -> str:
    """Apply the prefixes used when multilingual-E5 was trained."""

    return f"{input_type.value}: {text}"


def resolve_dense_device(
    requested: DenseDevice | str,
    *,
    cuda_available: bool,
) -> str:
    value = DenseDevice(requested)
    if value == DenseDevice.AUTO:
        return "cuda" if cuda_available else "cpu"
    if value == DenseDevice.CUDA and not cuda_available:
        raise DenseModelUnavailableError(
            "CUDA was requested for Dense retrieval but is unavailable."
        )
    return value.value


class HuggingFaceE5Encoder:
    """Direct transformers adapter; no sentence-transformers dependency."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        *,
        model_revision: str = "main",
        tokenizer_revision: str | None = None,
        model_path: Path | None = None,
        device: DenseDevice | str = DenseDevice.AUTO,
        max_length: int = 512,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise DenseModelUnavailableError(
                "Dense retrieval requires torch and transformers. "
                "Install the project's 'dense' optional dependencies."
            ) from exc

        tokenizer_revision = tokenizer_revision or model_revision
        load_source = str(Path(model_path)) if model_path is not None else model_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                load_source,
                revision=tokenizer_revision,
                use_fast=True,
                local_files_only=local_files_only,
            )
            model = AutoModel.from_pretrained(
                load_source,
                revision=model_revision,
                local_files_only=local_files_only,
            )
        except (OSError, ValueError) as exc:
            location = str(model_path) if model_path is not None else (
                "the local cache" if local_files_only else "Hugging Face"
            )
            raise DenseModelUnavailableError(
                f"Could not load {model_name!r} from {location}: {exc}"
            ) from exc

        if not getattr(tokenizer, "is_fast", False):
            raise DenseModelUnavailableError(
                "Dense safe segmentation requires a fast tokenizer with offset mappings."
            )
        if max_length > 512:
            raise ValueError(
                "multilingual-e5-small supports at most 512 tokens; "
                "max_length cannot exceed 512"
            )

        resolved_device = resolve_dense_device(
            device,
            cuda_available=torch.cuda.is_available(),
        )

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model.to(resolved_device)
        self._model.eval()
        self.device = resolved_device
        self._max_length = max_length
        resolved_model_revision = (
            getattr(model.config, "_commit_hash", None) or model_revision
        )
        resolved_tokenizer_revision = (
            tokenizer.init_kwargs.get("_commit_hash") or tokenizer_revision
        )
        self._fingerprint = EncoderFingerprint(
            model_name=model_name,
            model_revision=str(resolved_model_revision),
            tokenizer_revision=str(resolved_tokenizer_revision),
            embedding_dimension=int(model.config.hidden_size),
            max_length=max_length,
            pooling_method="attention_mask_mean",
            normalize_embeddings=True,
            dtype="float32",
        )

    @property
    def fingerprint(self) -> EncoderFingerprint:
        return self._fingerprint

    def prepared_token_count(
        self,
        text: str,
        input_type: EncoderInputType,
    ) -> int:
        encoded = self._tokenizer(
            e5_prefixed_text(text, input_type),
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        return len(encoded["input_ids"])

    def content_token_spans(self, text: str) -> list[EncoderTokenSpan]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        return [
            EncoderTokenSpan(start=int(start), end=int(end))
            for start, end in encoded["offset_mapping"]
            if int(end) > int(start)
        ]

    def encode(
        self,
        texts: Sequence[str],
        input_type: EncoderInputType,
    ) -> list[list[float]]:
        if not texts:
            return []
        prepared = [e5_prefixed_text(text, input_type) for text in texts]
        counts = [
            self.prepared_token_count(text, input_type) for text in texts
        ]
        oversized = [count for count in counts if count > self._max_length]
        if oversized:
            raise DenseInputTooLongError(
                "Refusing to truncate Dense input: "
                f"maximum observed length {max(oversized)} exceeds "
                f"{self._max_length} tokens"
            )

        batch = self._tokenizer(
            prepared,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        batch = {name: value.to(self.device) for name, value in batch.items()}
        with self._torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).bool()
            summed = hidden.masked_fill(~mask, 0.0).sum(dim=1)
            counts_tensor = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts_tensor
            normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.detach().cpu().to(self._torch.float32).tolist()
