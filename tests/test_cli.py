from softdoc.cli import main


def test_cli_parse_and_validate(mineru_fixture_dir, tmp_path, capsys) -> None:
    output = tmp_path / "cli_output"
    assert main(["parse-mineru", str(mineru_fixture_dir), "--output", str(output)]) == 0
    assert main(["validate", str(output)]) == 0
    assert (output / "rule_coverage_report.json").is_file()
    assert (output / "rule_coverage_report.md").is_file()
    captured = capsys.readouterr()
    assert "Parsed" in captured.out
    assert "Valid" in captured.out
