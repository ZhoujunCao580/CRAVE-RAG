import json
from pathlib import Path

import pytest

from softdoc.cli import main
from softdoc.prompt_registry import (
    PromptComponent,
    PromptLifecycle,
    get_prompt,
    prompt_manifest,
)
from softdoc.prompts import load_prompt_text


PROMPT_DIRECTORY = Path(__file__).parents[1] / "src" / "softdoc" / "prompts"


def test_registry_contains_every_model_facing_prompt() -> None:
    manifest = prompt_manifest()
    assert [item["component"] for item in manifest] == [
        item.value
        for item in PromptComponent
        if item != PromptComponent.COVERAGE_CHECKER
    ]
    assert all(item["lifecycle"] == "active" for item in manifest)
    assert all(len(str(item["sha256"])) == 64 for item in manifest)
    assert all(item["version"] for item in manifest)

    complete_catalog = prompt_manifest(include_inactive=True)
    assert [item["component"] for item in complete_catalog] == [
        item.value for item in PromptComponent
    ]
    coverage = get_prompt(PromptComponent.COVERAGE_CHECKER)
    assert coverage.lifecycle == PromptLifecycle.LEGACY_INACTIVE


def test_registry_renders_dynamic_and_static_prompts() -> None:
    planner = get_prompt("planner")
    assert "What changed?" in planner.render("What changed?")
    with pytest.raises(ValueError, match="requires non-blank"):
        planner.render()

    controller = get_prompt("controller")
    assert controller.render() == controller.canonical_text
    with pytest.raises(ValueError, match="takes no input"):
        controller.render("unexpected")


def test_registry_text_comes_from_central_versioned_prompt_assets() -> None:
    assert get_prompt("visual_reader").canonical_text == load_prompt_text(
        "visual_reader_v0_5.txt"
    )
    assert get_prompt("visual_retrieval").canonical_text == load_prompt_text(
        "visual_retrieval_v0_1.txt"
    ).removesuffix("\n")
    assert get_prompt("visual_scan").canonical_text == load_prompt_text(
        "visual_scan_v0_1.txt"
    )
    assert get_prompt("checker").canonical_text == load_prompt_text(
        "checker_v2_4.txt"
    ).removesuffix("\n")
    assert get_prompt("controller").canonical_text == load_prompt_text(
        "controller_policy_v0_13.txt"
    )
    assert get_prompt("answerer").canonical_text == load_prompt_text(
        "answerer_v0_8.txt"
    )
    table_reader = get_prompt("multimodal_table_reader")
    assert table_reader.canonical_text == (
        load_prompt_text("multimodal_table_reader_v0_3_system.txt")
        + "\n# User message template\n\n"
        + load_prompt_text("multimodal_table_reader_v0_3_user.txt")
    )
    assert table_reader.prompt_kind == "system_and_user_prompt_template"
    assert "<ROOT_QUESTION>" in get_prompt("planner").canonical_text
    assert get_prompt("planner").prompt_kind == "system_and_user_prompt"
    assert "# User message" in get_prompt("planner").canonical_text


def test_all_canonical_prompts_use_markdown_sections() -> None:
    for component in PromptComponent:
        prompt = get_prompt(component).canonical_text
        assert "# Role" in prompt
        assert "# Goal" in prompt
        assert "# Output" in prompt


def test_prompt_directory_contains_only_current_assets() -> None:
    current = {path.name for path in PROMPT_DIRECTORY.glob("*.txt")}

    assert current == {
        "planner_v0_21.txt",
        "planner_v0_25_visual_scan.txt",
        "visual_retrieval_v0_1.txt",
        "visual_reader_v0_5.txt",
        "checker_v2_4.txt",
        "controller_policy_v0_13.txt",
        "answerer_v0_8.txt",
        "multimodal_table_reader_v0_3_system.txt",
        "multimodal_table_reader_v0_3_user.txt",
        "coverage_checker_v0_1.txt",
        "visual_scan_v0_1.txt",
    }
    assert not (PROMPT_DIRECTORY / "archive").exists()


def test_coverage_checker_prompt_keeps_completeness_environment_owned() -> None:
    prompt = " ".join(get_prompt("coverage_checker").canonical_text.split())

    assert "Judge every supplied canonical inventory item independently" in prompt
    assert "Environment, not you, decides" in prompt
    assert "matched / not_matched / unresolved" not in prompt
    assert "Return exactly one assessment for every supplied `inventory_id`" in prompt
    assert "Do not output an overall count" in prompt


def test_checker_prompt_explains_root_target_progression() -> None:
    prompt = get_prompt("checker").canonical_text
    normalized_prompt = " ".join(prompt.split())

    assert "you do not output or predict root_status" in normalized_prompt
    assert "Once all SubQuestions are satisfied" in normalized_prompt
    assert "If there are no SubQuestions" in normalized_prompt
    assert "derives Root ready exactly when" in normalized_prompt
    assert "current_target_status answers:" not in prompt
    assert '"observation_assessments"' in prompt
    assert '"evidence_updates"' in prompt
    assert '"remaining_gap_description"' in prompt
    assert "not only for the current_target" in normalized_prompt
    assert "it does not contain the Root Question" in normalized_prompt
    assert "the stored status for every other SubQuestion" in normalized_prompt
    assert "Promote reliable and relevant Observations to Evidence" in normalized_prompt
    assert "remaining_gap_description always belongs only to the current_target" in normalized_prompt
    assert "Never describe the next question's gap." in normalized_prompt
    assert "copies that question's text into the next current_target" in normalized_prompt
    assert "Your output is validated and applied by the program" not in prompt
    assert "New or replaced Evidence must be concise" in prompt
    assert "state-only target recheck" in normalized_prompt
    assert "supports_question_ids must contain exactly the current_target" in normalized_prompt
    assert "List in reused_evidence_ids only the existing Evidence" in normalized_prompt
    assert "lightweight Observation Recall invocation" in normalized_prompt
    assert "recalled_observations" in normalized_prompt
    assert "Never change the status of a non-current question" in normalized_prompt
    assert "input_id values such as I1 are local to one read call" in normalized_prompt
    assert "stable source_id, element_id, and page_id" in normalized_prompt


def test_controller_prompt_explains_missing_table_header_recovery() -> None:
    prompt = " ".join(get_prompt("controller").canonical_text.split())

    assert "missing_header_context limitation preserves" in prompt
    assert "read the original data fragment and that header fragment together" in prompt
    assert "Do not combine merely adjacent or unrelated tables" in prompt


def test_table_reader_user_prompt_shows_complete_missing_header_example() -> None:
    prompt = get_prompt("multimodal_table_reader").canonical_text

    assert '"code": "missing_header_context"' in prompt
    assert '"input_ids": ["I1"]' in prompt
    assert '"relevant_visible_content": [' in prompt
    assert "Do not emit this limitation when the header mapping is reliable" in prompt


def test_table_reader_prompt_requires_complete_local_enumeration() -> None:
    prompt = " ".join(get_prompt("multimodal_table_reader").canonical_text.split())

    assert "inspect the complete supplied table fragment" in prompt
    assert "do not stop after a few examples" in prompt
    assert "add a limitation" in prompt


def test_visual_scan_prompt_maps_ordered_images_to_global_input_ids() -> None:
    prompt = " ".join(get_prompt("visual_scan").canonical_text.split())

    assert "correspond to those input_ids in exactly the same order" in prompt
    assert "exactly one assessment for every supplied input_id" in prompt


def test_visual_scan_prompt_requires_named_section_boundary_reporting() -> None:
    prompt = " ".join(get_prompt("visual_scan").canonical_text.split())

    assert "scope_membership" in prompt
    assert "distinct next major section" in prompt
    assert "later pages in the same batch" in prompt


def test_cli_exports_versioned_prompts(tmp_path, capsys) -> None:
    output = tmp_path / "prompts"
    assert main(["prompts", "export", "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == len(PromptComponent) - 1
    assert not any(item["component"] == "coverage_checker" for item in manifest)
    for item in manifest:
        assert (output / f'{item["component"]}__{item["version"]}.txt').is_file()
    assert "Exported" in capsys.readouterr().out


def test_cli_can_explicitly_export_inactive_prompts(tmp_path, capsys) -> None:
    output = tmp_path / "all-prompts"
    assert main(
        [
            "prompts",
            "export",
            "--output",
            str(output),
            "--include-inactive",
        ]
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    coverage = next(
        item for item in manifest if item["component"] == "coverage_checker"
    )
    assert coverage["lifecycle"] == "legacy_inactive"
    assert (output / "coverage_checker__coverage-checker-v0.1.txt").is_file()
    assert "Exported" in capsys.readouterr().out
