from __future__ import annotations

from pathlib import Path

import pytest

from taskcli.models import Priority, Status
from taskcli.store import StoreError, TaskStore


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "db.json")
    store.load()
    assert store.tasks == []


def test_add_save_reload(tmp_path: Path) -> None:
    path = tmp_path / "db.json"
    store = TaskStore(path)
    store.load()
    store.add("first", Priority.HIGH, ["a"])
    store.add("second")
    store.save()

    reloaded = TaskStore(path)
    reloaded.load()
    assert [t.title for t in reloaded.tasks] == ["first", "second"]
    assert reloaded.tasks[0].priority is Priority.HIGH


def test_next_id_fills_gaps(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "db.json")
    store.load()
    store.add("one")
    store.add("two")
    store.remove(1)
    assert store.next_id() == 1


def test_complete_marks_done(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "db.json")
    store.load()
    task = store.add("thing")
    store.complete(task.id)
    assert store.get(task.id).status is Status.DONE


def test_get_missing_raises(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "db.json")
    store.load()
    with pytest.raises(StoreError):
        store.get(99)


def test_load_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "db.json"
    path.write_text("{ not json", encoding="utf-8")
    store = TaskStore(path)
    with pytest.raises(StoreError):
        store.load()


def test_load_rejects_non_array(tmp_path: Path) -> None:
    path = tmp_path / "db.json"
    path.write_text('{"id": 1}', encoding="utf-8")
    store = TaskStore(path)
    with pytest.raises(StoreError):
        store.load()
