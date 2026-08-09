"""ZS-19 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import textstats
from textstats import render_json_report
from textstats.cli import main

SAMPLE = "The cat sat on the mat. The mat was flat.\n"

REQUIRED_KEYS = {
    "word_count",
    "unique_words",
    "sentence_count",
    "avg_word_length",
    "flesch_reading_ease",
}


def _sample_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


def test_render_json_report_is_exported_from_the_package() -> None:
    assert "render_json_report" in textstats.__all__
    assert callable(textstats.render_json_report)


def test_json_report_has_the_required_keys_and_values() -> None:
    payload: dict[str, Any] = json.loads(render_json_report(SAMPLE))
    assert REQUIRED_KEYS <= set(payload)
    assert payload["word_count"] == 10
    assert payload["unique_words"] == 7
    assert payload["sentence_count"] == 2
    assert payload["avg_word_length"] == pytest.approx(3.0)
    assert payload["flesch_reading_ease"] == pytest.approx(117.16)


def test_cli_json_format_prints_parseable_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(_sample_file(tmp_path)), "--format", "json"]) == 0
    payload: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert REQUIRED_KEYS <= set(payload)
    assert payload["word_count"] == 10


def test_cli_format_text_is_the_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _sample_file(tmp_path)
    assert main([str(path), "--format", "text"]) == 0
    explicit = capsys.readouterr().out
    assert main([str(path)]) == 0
    default = capsys.readouterr().out
    assert explicit == default
    assert explicit.startswith("Text statistics")


def test_cli_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(_sample_file(tmp_path)), "--format", "xml"])
