"""Versioned prompt assets used by CRAVE-RAG model-facing components."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=None)
def load_prompt_text(filename: str) -> str:
    """Load one packaged UTF-8 prompt with platform-independent newlines."""

    if not filename or filename != filename.strip() or "/" in filename or "\\" in filename:
        raise ValueError("Prompt filename must be one local nonblank filename")
    text = files(__package__).joinpath(filename).read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n")


__all__ = ["load_prompt_text"]
