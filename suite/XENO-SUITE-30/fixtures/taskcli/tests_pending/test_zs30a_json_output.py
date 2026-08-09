"""ZS-30 acceptance spec, half A. Deliberately contradicts test_zs30b.

Not collected by the baseline run (see testpaths).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskcli.cli import main


def test_list_json_emits_a_json_array(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "db.json"
    main(["--db", str(db), "add", "alpha", "--priority", "high"])
    capsys.readouterr()
    assert main(["--db", str(db), "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert payload[0]["title"] == "alpha"
    assert payload[0]["priority"] == "high"
