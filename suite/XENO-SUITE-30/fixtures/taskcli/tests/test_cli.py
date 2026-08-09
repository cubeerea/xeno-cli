from __future__ import annotations

from pathlib import Path

import pytest

from taskcli.cli import main
from taskcli.formatting import HEADER


def test_add_then_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "db.json"
    assert main(["--db", str(db), "add", "write tests"]) == 0
    assert main(["--db", str(db), "list"]) == 0
    out = capsys.readouterr().out
    assert HEADER in out
    assert "write tests" in out


def test_list_empty_shows_header(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", str(tmp_path / "db.json"), "list"]) == 0
    out = capsys.readouterr().out
    assert HEADER in out
    assert "(no tasks)" in out


def test_done_and_status_filter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "db.json"
    main(["--db", str(db), "add", "alpha"])
    main(["--db", str(db), "add", "beta"])
    assert main(["--db", str(db), "done", "1"]) == 0
    capsys.readouterr()
    assert main(["--db", str(db), "list", "--status", "done"]) == 0
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "beta" not in out


def test_rm_unknown_id_returns_one(tmp_path: Path) -> None:
    assert main(["--db", str(tmp_path / "db.json"), "rm", "42"]) == 1


def test_add_accepts_priority_and_tags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "db.json"
    assert main(["--db", str(db), "add", "urgent", "--priority", "high", "--tag", "ops"]) == 0
    capsys.readouterr()
    main(["--db", str(db), "list"])
    assert "high" in capsys.readouterr().out
