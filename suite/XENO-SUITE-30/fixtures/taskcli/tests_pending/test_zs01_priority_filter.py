"""ZS-01 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from pathlib import Path

import pytest

from taskcli.cli import main


def _seed(db: Path) -> None:
    main(["--db", str(db), "add", "low thing", "--priority", "low"])
    main(["--db", str(db), "add", "high thing", "--priority", "high"])
    main(["--db", str(db), "add", "normal thing"])


def test_list_filters_by_priority(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "db.json"
    _seed(db)
    capsys.readouterr()
    assert main(["--db", str(db), "list", "--priority", "high"]) == 0
    out = capsys.readouterr().out
    assert "high thing" in out
    assert "low thing" not in out
    assert "normal thing" not in out


def test_priority_filter_combines_with_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "db.json"
    _seed(db)
    main(["--db", str(db), "done", "2"])
    capsys.readouterr()
    assert main(["--db", str(db), "list", "--priority", "high", "--status", "todo"]) == 0
    out = capsys.readouterr().out
    assert "high thing" not in out
    assert "(no tasks)" in out
