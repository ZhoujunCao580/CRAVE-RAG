from __future__ import annotations

from pathlib import Path

import pytest

from softdoc.adapters import MinerUAdapter
from softdoc.models import Document


@pytest.fixture
def mineru_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "mineru_sample"


@pytest.fixture
def parsed_document(mineru_fixture_dir: Path, tmp_path: Path) -> Document:
    return MinerUAdapter().parse(mineru_fixture_dir, tmp_path / "adapter_output")

