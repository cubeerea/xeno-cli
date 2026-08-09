"""A minimal task-tracking CLI.

Fixture for XENO-SUITE-30. Deliberately small, dependency-free and green on
ruff / mypy / pytest before any suite task is applied.
"""

from __future__ import annotations

from taskcli.models import Priority, Status, Task
from taskcli.store import StoreError, TaskStore

__all__ = ["Priority", "Status", "StoreError", "Task", "TaskStore"]
__version__ = "0.1.0"
