from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from softdoc.models import ContentAvailability, ElementType
from softdoc.retrieval.visual_dense import (
    VISUAL_INDEX_SCHEMA_VERSION,
    VisualDenseIndex,
    fixed_text_visual_quota_ranking,
    is_visual_retrieval_candidate,
    resolve_softdoc_asset,
    weighted_rrf_three_route_ranking,
)
from softdoc.retrieval import SubQuestionInput


class _FakeVisualModel:
    def encode_query(self, sentences, **_):
        assert sentences == ["Which figure contains the answer?"]
        return [np.ones((2, 3), dtype=np.float32)]

    def similarity(self, queries, documents, **_):
        assert len(queries) == 1
        assert len(documents) == 1
        return np.asarray([[0.75]], dtype=np.float32)


def test_visual_selection_uses_real_figures_and_only_visual_only_tables(
    parsed_document,
) -> None:
    base = parsed_document.elements[0]
    figure = base.model_copy(
        update={
            "element_id": "figure:1",
            "element_type": ElementType.FIGURE,
            "text": None,
            "html": None,
            "image_path": Path("figure.png"),
            "content_availability": ContentAvailability.VISUAL_ONLY,
        }
    )
    visual_table = figure.model_copy(
        update={"element_id": "table:1", "element_type": ElementType.TABLE}
    )
    structured_table = visual_table.model_copy(
        update={
            "element_id": "table:2",
            "html": "<table><tr><td>10</td></tr></table>",
            "content_availability": ContentAvailability.MIXED,
        }
    )

    assert is_visual_retrieval_candidate(figure)
    assert is_visual_retrieval_candidate(visual_table)
    assert not is_visual_retrieval_candidate(structured_table)


def test_linux_can_resolve_windows_authored_asset_paths(tmp_path: Path) -> None:
    expected = tmp_path / "assets" / "elements" / "figure.jpg"
    expected.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), "white").save(expected)

    assert resolve_softdoc_asset(tmp_path, r"assets\elements\figure.jpg") == expected


def test_fixed_three_text_two_visual_is_batched_and_deduplicated() -> None:
    ranking = fixed_text_visual_quota_ranking(
        [["t1", 1], ["shared", 2], ["t2", 3], ["t3", 4], ["t4", 5]],
        [["v1", 8], ["shared", 2], ["v2", 9], ["v3", 10]],
        excluded_ids={"t1"},
    )

    assert ranking[:5] == [
        ["shared", 2],
        ["t2", 3],
        ["t3", 4],
        ["v1", 8],
        ["v2", 9],
    ]
    assert len({item[0] for item in ranking}) == len(ranking)


def test_three_route_rrf_rewards_cross_route_agreement() -> None:
    ranking = weighted_rrf_three_route_ranking(
        [["text", 1], ["shared", 2]],
        [["dense", 3], ["shared", 2]],
        [["visual", 4], ["shared", 2]],
        rrf_k=20,
        bm25_weight=1.0,
        dense_weight=1.25,
        visual_weight=1.0,
    )

    assert ranking[0] == ["shared", 2]
    assert {item[0] for item in ranking} == {"text", "dense", "visual", "shared"}


def test_visual_dense_index_returns_runtime_candidates(
    parsed_document, tmp_path: Path
) -> None:
    base = parsed_document.elements[0]
    figure = base.model_copy(
        update={
            "element_id": "figure:runtime",
            "element_type": ElementType.FIGURE,
            "text": None,
            "html": None,
            "reference_label": "Figure 7",
            "content_availability": ContentAvailability.VISUAL_ONLY,
        }
    )
    document = parsed_document.model_copy(update={"elements": [figure]})
    index_dir = tmp_path / "visual-index"
    shards = index_dir / "shards"
    shards.mkdir(parents=True)
    fingerprint = "fixture-fingerprint"
    model_name = "fixture-visual-model"
    digest = "a" * 64
    (index_dir / "config.json").write_text(
        json.dumps(
            {
                "schema_version": VISUAL_INDEX_SCHEMA_VERSION,
                "model": model_name,
                "inventory_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    (index_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": VISUAL_INDEX_SCHEMA_VERSION,
                "state": "completed",
                "pending_image_count": 0,
                "inventory_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    (index_dir / "assets.jsonl").write_text(
        json.dumps(
            {
                "visual_asset_id": "visual:runtime",
                "document_id": document.document_id,
                "element_id": figure.element_id,
                "element_type": figure.element_type.value,
                "page_id": figure.page_id,
                "page_number": figure.page_number,
                "image_sha256": digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(shards / "shard-00000.npz", **{digest: np.ones((2, 3))})

    result = VisualDenseIndex(
        document,
        index_dir,
        model=_FakeVisualModel(),
    ).search(
        SubQuestionInput(
            subquestion_id="Q1",
            text="Which figure contains the answer?",
        )
    )

    assert result.total_candidates == 1
    assert result.candidates[0].element_id == "figure:runtime"
    assert result.candidates[0].visual_rank == 1
    assert result.candidates[0].visual_score == pytest.approx(0.75)
    assert "Figure 7" in result.candidates[0].preview_text
