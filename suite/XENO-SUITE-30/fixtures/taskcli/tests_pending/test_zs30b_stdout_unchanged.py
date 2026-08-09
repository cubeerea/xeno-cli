"""ZS-30 acceptance spec, half B. Deliberately contradicts test_zs30a.

Not collected by the baseline run (see testpaths).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taskcli.cli import main


def test_json_flag_leaves_stdout_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "db.json"
    main(["--db", str(db), "add", "alpha", "--priority", "high"])
    capsys.readouterr()

    assert main(["--db", str(db), "list"]) == 0
    plain = capsys.readouterr().out

    assert main(["--db", str(db), "list", "--json"]) == 0
    with_flag = capsys.readouterr().out

    assert with_flag == plain
