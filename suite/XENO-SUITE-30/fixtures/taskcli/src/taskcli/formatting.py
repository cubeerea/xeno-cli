"""Plain-text rendering of tasks."""

from __future__ import annotations

from taskcli.models import Status, Task

HEADER = "ID  ST  PRI     TITLE"
SEPARATOR = "-" * len(HEADER)


def status_symbol(status: Status) -> str:
    """A two-character marker for a status."""
    return "[x]" if status is Status.DONE else "[ ]"


def format_row(task: Task) -> str:
    """Render a single task as one fixed-width row."""
    return f"{task.id:<3} {status_symbol(task.status)} {task.priority.value:<7} {task.title}"


def render_table(tasks: list[Task]) -> str:
    """Render a full table, always including the header."""
    lines = [HEADER, SEPARATOR]
    lines.extend(format_row(task) for task in tasks)
    if not tasks:
        lines.append("(no tasks)")
    return "\n".join(lines)
