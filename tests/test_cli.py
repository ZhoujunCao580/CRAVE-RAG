from softdoc.cli import build_parser, main


def test_cli_parse_and_validate(mineru_fixture_dir, tmp_path, capsys) -> None:
    output = tmp_path / "cli_output"
    assert main(["parse-mineru", str(mineru_fixture_dir), "--output", str(output)]) == 0
    assert main(["validate", str(output)]) == 0
    assert (output / "rule_coverage_report.json").is_file()
    assert (output / "rule_coverage_report.md").is_file()
    captured = capsys.readouterr()
    assert "Parsed" in captured.out
    assert "Valid" in captured.out


def test_cli_parses_model_runner_without_calling_models(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "run-model",
            str(tmp_path / "softdoc"),
            "--question",
            "What changed?",
            "--output",
            str(tmp_path / "run"),
            "--dense",
            "--dense-device",
            "cpu",
        ]
    )
    assert args.command == "run-model"
    assert args.question == "What changed?"
    assert args.dense is True
    assert args.action_budget == 7
