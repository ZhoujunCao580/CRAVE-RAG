import pytest
from pydantic import ValidationError

from softdoc.visual_reading import (
    ObservationRegion,
    VisualInput,
    VisualLimitation,
    VisualObservation,
    VisualReadRequest,
    VisualReadResult,
    VISUAL_READER_SYSTEM_PROMPT,
    VISUAL_READER_PROMPT_VERSION,
    validate_visual_read_result,
    visual_reader_user_prompt,
)


def _request() -> VisualReadRequest:
    return VisualReadRequest(
        action_id="action:visual:1",
        document_id="doc:test",
        source_name="test.pdf",
        problem="What value is shown?",
        visual_inputs=[
            VisualInput(
                input_id="I1",
                visual_asset_id="visual:page:1",
                page_id="page:1",
                page_number=1,
                page_image_path="assets/pages/page_0001.png",
            )
        ],
    )


def test_visual_read_result_contains_only_model_owned_content():
    result = VisualReadResult(
        observations=[
            VisualObservation(
                text="The value shown in I1 is 79.",
                sources=[ObservationRegion(input_id="I1", bbox=None)],
            )
        ],
        limitations=[],
    )

    assert set(result.model_dump()) == {"observations", "limitations"}


@pytest.mark.parametrize("forbidden", ["answer", "action_id", "observation_id"])
def test_visual_read_result_rejects_non_model_owned_fields(forbidden):
    payload = {"observations": [], "limitations": [], forbidden: "not allowed"}

    with pytest.raises(ValidationError, match=forbidden):
        VisualReadResult.model_validate(payload)


def test_visual_reader_request_distinguishes_local_input_and_stable_asset_ids():
    request = _request()

    assert request.visual_inputs[0].input_id == "I1"
    assert request.visual_inputs[0].visual_asset_id == "visual:page:1"


def test_joint_observation_references_both_local_inputs():
    observation = VisualObservation(
        text="The series rises in both images.",
        sources=[
            ObservationRegion(input_id="I1", bbox=None),
            ObservationRegion(input_id="I2", bbox=None),
        ],
    )

    assert [source.input_id for source in observation.sources] == ["I1", "I2"]


def test_visual_observation_rejects_duplicate_input_grounding():
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        VisualObservation(
            text="Repeated grounding.",
            sources=[
                ObservationRegion(input_id="I1"),
                ObservationRegion(input_id="I1"),
            ],
        )


def test_visual_result_cannot_reference_an_unsupplied_input():
    result = VisualReadResult(
        observations=[
            VisualObservation(
                text="A value is visible.",
                sources=[ObservationRegion(input_id="I2")],
            )
        ],
        limitations=[],
    )

    with pytest.raises(ValueError, match="not supplied"):
        validate_visual_read_result(_request(), result)


def test_limitation_is_free_text_grounded_to_inputs_without_code_taxonomy():
    limitation = VisualLimitation(
        description="The legend is too small to read reliably.",
        input_ids=["I1"],
    )

    assert set(limitation.model_dump()) == {"description", "input_ids"}


def test_visual_reader_prompt_forbids_answer_and_global_ids():
    prompt = visual_reader_user_prompt(_request())

    assert VISUAL_READER_PROMPT_VERSION == "visual-reader-v0.4"
    assert '"observations"' in prompt
    assert "Do not\nadd an answer or conclusion field" in prompt
    assert '"action_id"' not in prompt.split("return exactly this JSON shape:", 1)[1]
    assert "Do not generate action IDs" in VISUAL_READER_SYSTEM_PROMPT
    assert "bbox" not in prompt
    assert "bbox" not in VISUAL_READER_SYSTEM_PROMPT


def test_visual_reader_prompt_shows_joint_shape_for_multiple_images():
    request = _request().model_copy(
        update={
            "visual_inputs": [
                *_request().visual_inputs,
                VisualInput(
                    input_id="I2",
                    visual_asset_id="visual:page:2",
                    page_id="page:2",
                    page_number=2,
                    page_image_path="assets/pages/page_0002.png",
                ),
            ]
        }
    )

    prompt = visual_reader_user_prompt(request)

    output_example = prompt.split("return exactly this JSON shape:", 1)[1]
    assert '"input_id": "I1"' in output_example
    assert '"input_id": "I2"' in output_example


def test_visual_read_result_rejects_unbounded_repetition():
    observations = [
        {
            "text": "Repeated fact.",
            "sources": [{"input_id": "I1", "bbox": None}],
        }
        for _ in range(17)
    ]
    with pytest.raises(ValidationError, match="at most 16 items"):
        VisualReadResult.model_validate(
            {"observations": observations, "limitations": []}
        )
