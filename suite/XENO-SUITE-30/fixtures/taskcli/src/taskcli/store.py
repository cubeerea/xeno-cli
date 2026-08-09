"""JSON-file persistence for tasks."""

from __future__ import annotations

import json
from pathlib import Path

from taskcli.models import Priority, Status, Task, ValidationError


class StoreError(RuntimeError):
    """Raised when the backing store cannot be read or a task is missing."""


class TaskStore:
    """A list of tasks persisted as a JSON array on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tasks: list[Task] = []

    @property
    def tasks(self) -> list[Task]:
        """A copy of the in-memory task list."""
        return list(self._tasks)

    def load(self) -> None:
        """Read the backing file, treating a missing file as an empty store."""
        if not self.path.exists():
            self._tasks = []
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StoreError(f"{self.path} is not valid JSON") from exc
        if not isinstance(raw, list):
            raise StoreError(f"{self.path} must contain a JSON array")
        try:
            self._tasks = [Task.from_dict(record) for record in raw]
        except ValidationError as exc:
            raise StoreError(str(exc)) from exc

    def save(self) -> None:
        """Write the in-memory task list back to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([task.to_dict() for task in self._tasks], indent=2)
        self.path.write_text(payload + "\n", encoding="utf-8")

    def next_id(self) -> int:
        """The smallest positive id not already in use."""
        used = {task.id for task in self._tasks}
        candidate = 1
        while candidate in used:
            candidate += 1
        return candidate

    def add(
        self,
        title: str,
        priority: Priority = Priority.NORMAL,
        tags: list[str] | None = None,
    ) -> Task:
        """Append a new task and return it."""
        task = Task(id=self.next_id(), title=title, priority=priority, tags=tags or [])
        self._tasks.append(task)
        return task

    def get(self, task_id: int) -> Task:
        """Return the task with ``task_id`` or raise :class:`StoreError`."""
        for task in self._tasks:
            if task.id == task_id:
                return task
        raise StoreError(f"no task with id {task_id}")

    def complete(self, task_id: int) -> Task:
        """Mark a task done and return it."""
        task = self.get(task_id)
        task.status = Status.DONE
        return task

    def remove(self, task_id: int) -> None:
        """Delete a task by id."""
        task = self.get(task_id)
        self._tasks.remove(task)
