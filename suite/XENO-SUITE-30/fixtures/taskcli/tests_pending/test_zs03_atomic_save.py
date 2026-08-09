"""ZS-03 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from taskcli.store import TaskStore


def _boom(*_args: Any, **_kwargs: Any) -> None:
    raise OSError("simulated rename failure")


def test_failed_replace_preserves_original_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "db.json"
    store = TaskStore(db)
    store.load()
    store.add("original")
    store.save()
    before = db.read_text(encoding="utf-8")

    store.add("second")
    monkeypatch.setattr("taskcli.store.os.replace", _boom)
    with pytest.raises(OSError):
        store.save()

    assert db.read_text(encoding="utf-8") == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["db.json"]


def test_save_propagates_replace_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = TaskStore(tmp_path / "db.json")
    store.load()
    store.add("only")
    monkeypatch.setattr("taskcli.store.os.replace", _boom)
    with pytest.raises(OSError):
        store.save()
