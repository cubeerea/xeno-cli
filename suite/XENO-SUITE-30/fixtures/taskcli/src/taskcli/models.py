"""Domain objects for the task tracker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Status(StrEnum):
    """Lifecycle state of a task."""

    TODO = "todo"
    DONE = "done"


class Priority(StrEnum):
    """How urgent a task is."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

    @property
    def rank(self) -> int:
        """Sort key: higher rank sorts first in listings."""
        return {Priority.LOW: 0, Priority.NORMAL: 1, Priority.HIGH: 2}[self]


class ValidationError(ValueError):
    """Raised when a task cannot be constructed from raw data."""


@dataclass
class Task:
    """A single tracked unit of work."""

    id: int
    title: str
    status: Status = Status.TODO
    priority: Priority = Priority.NORMAL
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise ValidationError("task id must be a positive integer")
        if not self.title.strip():
            raise ValidationError("task title must not be blank")
        self.title = self.title.strip()
        self.tags = sorted({tag.strip().lower() for tag in self.tags if tag.strip()})

    @property
    def is_open(self) -> bool:
        """True while the task still needs doing."""
        return self.status is Status.TODO

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible mapping."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "priority": self.priority.value,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Task:
        """Rebuild a task from a mapping produced by :meth:`to_dict`."""
        try:
            task_id = int(raw["id"])
            title = str(raw["title"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed task record: {raw!r}") from exc

        status_raw = str(raw.get("status", Status.TODO.value))
        priority_raw = str(raw.get("priority", Priority.NORMAL.value))
        try:
            status = Status(status_raw)
            priority = Priority(priority_raw)
        except ValueError as exc:
            raise ValidationError(f"unknown status/priority in record: {raw!r}") from exc

        tags_raw = raw.get("tags", [])
        if not isinstance(tags_raw, list):
            raise ValidationError(f"tags must be a list, got {type(tags_raw)!r}")

        return cls(
            id=task_id,
            title=title,
            status=status,
            priority=priority,
            tags=[str(tag) for tag in tags_raw],
        )


def sort_tasks(tasks: list[Task]) -> list[Task]:
    """Open tasks first, then descending priority, then ascending id."""
    return sorted(tasks, key=lambda t: (not t.is_open, -t.priority.rank, t.id))
