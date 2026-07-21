from softdoc.cli import main


def test_cli_parse_and_validate(mineru_fixture_dir, tmp_path, capsys) -> None:
    output = tmp_path / "cli_output"
    assert main(["parse-mineru", str(mineru_fixture_dir), "--output", str(output)]) == 0
    assert main(["validate", str(output)]) == 0
    captured = capsys.readouterr()
    assert "Parsed" in captured.out
    assert "Valid" in captured.out
