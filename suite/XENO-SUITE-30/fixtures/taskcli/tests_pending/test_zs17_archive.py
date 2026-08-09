"""ZS-17 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from pathlib import Path

import pytest

from taskcli.cli import main
from taskcli.models import Status
from taskcli.store import TaskStore


def test_archived_status_exists() -> None:
    assert Status("archived") is Status.ARCHIVED


def test_store_archive_sets_status(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "db.json")
    store.load()
    task = store.add("obsolete")
    store.archive(task.id)
    assert store.get(task.id).status is Status.ARCHIVED
    assert not store.get(task.id).is_open


def test_archive_survives_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "db.json"
    store = TaskStore(path)
    store.load()
    task = store.add("obsolete")
    store.archive(task.id)
    store.save()

    reloaded = TaskStore(path)
    reloaded.load()
    assert reloaded.get(task.id).status is Status.ARCHIVED


def test_archive_subcommand_hides_task_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "db.json"
    main(["--db", str(db), "add", "stale"])
    main(["--db", str(db), "add", "fresh"])
    assert main(["--db", str(db), "archive", "1"]) == 0
    capsys.readouterr()
    assert main(["--db", str(db), "list"]) == 0
    out = capsys.readouterr().out
    assert "stale" not in out
    assert "fresh" in out


def test_archived_visible_with_explicit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "db.json"
    main(["--db", str(db), "add", "stale"])
    main(["--db", str(db), "archive", "1"])
    capsys.readouterr()
    assert main(["--db", str(db), "list", "--status", "archived"]) == 0
    assert "stale" in capsys.readouterr().out
