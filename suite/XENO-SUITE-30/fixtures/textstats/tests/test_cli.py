from __future__ import annotations

from pathlib import Path

import pytest

from textstats.cli import main
from textstats.report import TITLE

SAMPLE = "The cat sat on the mat. The mat was flat.\n"


def _sample_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


def test_main_prints_text_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(_sample_file(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert out.startswith(TITLE)
    assert "Words: 10" in out
    assert "Reading ease: 117.16" in out


def test_main_missing_file_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "nope.txt")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot read" in captured.err


def test_top_option_appends_word_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(_sample_file(tmp_path)), "--top", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2:] == ["the\t3", "mat\t2"]


def test_top_defaults_to_no_extra_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(_sample_file(tmp_path))]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert not [line for line in lines if "\t" in line]
