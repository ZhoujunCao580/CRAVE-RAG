from __future__ import annotations

from pathlib import Path
import shutil

import pytest
from PIL import Image

from softdoc.adapters import MinerUAdapter
from softdoc.models import Document
from softdoc.pipeline import SoftDocPipeline


@pytest.fixture
def mineru_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "mineru_sample"


@pytest.fixture
def parsed_document(mineru_fixture_dir: Path, tmp_path: Path) -> Document:
    return SoftDocPipeline(MinerUAdapter()).parse(
        mineru_fixture_dir,
        tmp_path / "adapter_output",
    )


@pytest.fixture
def mineru_degraded_fixture_dir(tmp_path: Path) -> Path:
    source = Path(__file__).parent / "fixtures" / "mineru_degraded_blocks"
    destination = tmp_path / "mineru_degraded_blocks"
    shutil.copytree(source, destination)
    pages_dir = destination / "pages"
    pages_dir.mkdir()
    for page_index in range(8):
        image = Image.new(
            "RGB",
            (200, 200),
            color=(240 - page_index * 10, 245, 250),
        )
        image.save(pages_dir / f"page_{page_index}.png")
        image.close()
    return destination
