import json
from pathlib import Path

import pytest

from softdoc.cli import main
from softdoc.prompt_registry import PromptComponent, get_prompt, prompt_manifest
from softdoc.prompts import load_prompt_text


PROMPT_DIRECTORY = Path(__file__).parents[1] / "src" / "softdoc" / "prompts"


def test_registry_contains_every_model_facing_prompt() -> None:
    manifest = prompt_manifest()
    assert [item["component"] for item in manifest] == [item.value for item in PromptComponent]
    assert all(len(str(item["sha256"])) == 64 for item in manifest)
    assert all(item["version"] for item in manifest)


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
        "visual_reader_v0_4.txt"
    )
    assert get_prompt("visual_retrieval").canonical_text == load_prompt_text(
        "visual_retrieval_v0_1.txt"
    ).removesuffix("\n")
    assert get_prompt("checker").canonical_text == load_prompt_text(
        "checker_v1_9.txt"
    ).removesuffix("\n")
    assert get_prompt("controller").canonical_text == load_prompt_text(
        "controller_policy_v0_7.txt"
    )
    assert get_prompt("answerer").canonical_text == load_prompt_text(
        "answerer_v0_7.txt"
    )
    assert "<ROOT_QUESTION>" in get_prompt("planner").canonical_text
    assert get_prompt("planner").prompt_kind == "system_and_user_prompt"
    assert "# User message" in get_prompt("planner").canonical_text


def test_all_canonical_prompts_use_markdown_sections() -> None:
    for component in PromptComponent:
        prompt = get_prompt(component).canonical_text
        assert "# Role" in prompt
        assert "# Goal" in prompt
        assert "# Output" in prompt


def test_prompt_directory_contains_only_current_prompt_assets() -> None:
    current = {path.name for path in PROMPT_DIRECTORY.glob("*.txt")}

    assert current == {
        "planner_v0_20.txt",
        "visual_retrieval_v0_1.txt",
        "visual_reader_v0_4.txt",
        "checker_v1_9.txt",
        "controller_policy_v0_7.txt",
        "answerer_v0_7.txt",
    }
    assert not (PROMPT_DIRECTORY / "archive").exists()


def test_checker_prompt_explains_root_target_progression() -> None:
    prompt = get_prompt("checker").canonical_text
    normalized_prompt = " ".join(prompt.split())

    assert (
        "root_status remains incomplete if any registered SubQuestion is"
        in normalized_prompt
    )
    assert "Once all SubQuestions are satisfied" in normalized_prompt
    assert "If there are no SubQuestions" in normalized_prompt
    assert "The process completes only when root_status is ready." in normalized_prompt
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


def test_cli_exports_versioned_prompts(tmp_path, capsys) -> None:
    output = tmp_path / "prompts"
    assert main(["prompts", "export", "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == len(PromptComponent)
    for item in manifest:
        assert (output / f'{item["component"]}__{item["version"]}.txt').is_file()
    assert "Exported" in capsys.readouterr().out
