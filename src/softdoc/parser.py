"""Parser contract implemented by parser-specific adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from softdoc.models import Document


@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, input_path: Path, output_dir: Path) -> Document:
        """Parse a parser-specific artifact directory into the neutral IR."""
        ...
