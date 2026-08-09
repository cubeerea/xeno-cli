"""Command-line entry point for the task tracker."""

from __future__ import annotations

import argparse
from pathlib import Path

from taskcli.formatting import render_table
from taskcli.models import Priority, Status, sort_tasks
from taskcli.store import StoreError, TaskStore

DEFAULT_DB = Path(".taskcli.json")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``taskcli`` command."""
    parser = argparse.ArgumentParser(prog="taskcli", description="Track small units of work.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="path to the JSON store")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="add a task")
    add.add_argument("title")
    add.add_argument(
        "--priority",
        choices=[p.value for p in Priority],
        default=Priority.NORMAL.value,
    )
    add.add_argument("--tag", action="append", dest="tags", default=[])

    listing = sub.add_parser("list", help="list tasks")
    listing.add_argument("--status", choices=[s.value for s in Status], default=None)

    done = sub.add_parser("done", help="mark a task complete")
    done.add_argument("task_id", type=int)

    remove = sub.add_parser("rm", help="delete a task")
    remove.add_argument("task_id", type=int)

    return parser


def _open_store(db_path: Path) -> TaskStore:
    store = TaskStore(db_path)
    store.load()
    return store


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        store = _open_store(args.db)
    except StoreError as exc:
        print(f"error: {exc}")
        return 2

    try:
        if args.command == "add":
            task = store.add(args.title, Priority(args.priority), list(args.tags))
            store.save()
            print(f"added task {task.id}")
        elif args.command == "list":
            tasks = store.tasks
            if args.status is not None:
                wanted = Status(args.status)
                tasks = [task for task in tasks if task.status is wanted]
            print(render_table(sort_tasks(tasks)))
        elif args.command == "done":
            task = store.complete(args.task_id)
            store.save()
            print(f"completed task {task.id}")
        elif args.command == "rm":
            store.remove(args.task_id)
            store.save()
            print(f"removed task {args.task_id}")
    except StoreError as exc:
        print(f"error: {exc}")
        return 1

    return 0
