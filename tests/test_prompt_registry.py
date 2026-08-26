import json

import pytest

from softdoc.cli import main
from softdoc.prompt_registry import PromptComponent, get_prompt, prompt_manifest


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


def test_cli_exports_versioned_prompts(tmp_path, capsys) -> None:
    output = tmp_path / "prompts"
    assert main(["prompts", "export", "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == len(PromptComponent)
    for item in manifest:
        assert (output / f'{item["component"]}__{item["version"]}.txt').is_file()
    assert "Exported" in capsys.readouterr().out
