"""ZS-16 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from pathlib import Path

import pytest

from taskcli.cli import main
from taskcli.models import Task, filter_by_tags


def test_filter_by_tags_requires_all_tags() -> None:
    a = Task(id=1, title="a", tags=["ops", "urgent"])
    b = Task(id=2, title="b", tags=["ops"])
    assert filter_by_tags([a, b], ["ops"]) == [a, b]
    assert filter_by_tags([a, b], ["ops", "urgent"]) == [a]
    assert filter_by_tags([a, b], []) == [a, b]


def test_filter_by_tags_is_case_insensitive() -> None:
    a = Task(id=1, title="a", tags=["Ops"])
    assert filter_by_tags([a], ["OPS"]) == [a]


def test_list_filters_by_tag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "db.json"
    main(["--db", str(db), "add", "tagged", "--tag", "ops"])
    main(["--db", str(db), "add", "untagged"])
    capsys.readouterr()
    assert main(["--db", str(db), "list", "--tag", "ops"]) == 0
    out = capsys.readouterr().out
    assert "tagged" in out
    assert "untagged" not in out
