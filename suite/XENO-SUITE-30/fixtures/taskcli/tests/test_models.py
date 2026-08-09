from __future__ import annotations

import pytest

from taskcli.models import Priority, Status, Task, ValidationError, sort_tasks


def test_task_normalises_title_and_tags() -> None:
    task = Task(id=1, title="  write docs  ", tags=["Docs", "docs", " ", "Build"])
    assert task.title == "write docs"
    assert task.tags == ["build", "docs"]


def test_task_rejects_bad_id() -> None:
    with pytest.raises(ValidationError):
        Task(id=0, title="nope")


def test_task_rejects_blank_title() -> None:
    with pytest.raises(ValidationError):
        Task(id=1, title="   ")


def test_round_trip_dict() -> None:
    task = Task(id=3, title="ship it", status=Status.DONE, priority=Priority.HIGH, tags=["rel"])
    assert Task.from_dict(task.to_dict()) == task


def test_from_dict_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        Task.from_dict({"id": 1, "title": "x", "status": "sideways"})


def test_from_dict_rejects_non_list_tags() -> None:
    with pytest.raises(ValidationError):
        Task.from_dict({"id": 1, "title": "x", "tags": "nope"})


def test_is_open() -> None:
    assert Task(id=1, title="a").is_open
    assert not Task(id=1, title="a", status=Status.DONE).is_open


def test_priority_rank_ordering() -> None:
    assert Priority.HIGH.rank > Priority.NORMAL.rank > Priority.LOW.rank


def test_sort_tasks_open_first_then_priority() -> None:
    tasks = [
        Task(id=1, title="done high", status=Status.DONE, priority=Priority.HIGH),
        Task(id=2, title="open low", priority=Priority.LOW),
        Task(id=3, title="open high", priority=Priority.HIGH),
    ]
    assert [t.id for t in sort_tasks(tasks)] == [3, 2, 1]
