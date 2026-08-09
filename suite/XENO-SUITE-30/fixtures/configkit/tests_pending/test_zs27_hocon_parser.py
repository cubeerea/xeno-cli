"""ZS-27 acceptance spec: replace the hand-rolled INI reader with xeno-hocon-strict.

``configkit.loader`` must import ``parse_strict`` from the third-party
``xeno_hocon_strict`` distribution (>= 2.0, declared in ``[project].dependencies``)
and hand it the raw file text instead of using the in-tree parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import xeno_hocon_strict
from xeno_hocon_strict import parse_strict

from configkit import loader


def test_xeno_hocon_strict_is_available_at_version_two_or_newer() -> None:
    major, minor = (int(part) for part in xeno_hocon_strict.__version__.split(".")[:2])
    assert (major, minor) >= (2, 0)


def test_loader_binds_the_third_party_parse_strict() -> None:
    assert loader.parse_strict is parse_strict


def test_load_file_delegates_ini_parsing_to_parse_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "[db]\nhost = localhost\nport = 5432\n"
    path = tmp_path / "config.ini"
    path.write_text(text, encoding="utf-8")

    calls: list[str] = []

    def fake_parse_strict(raw: str) -> dict[str, Any]:
        calls.append(raw)
        return {"db": {"host": "localhost", "port": 5432}}

    monkeypatch.setattr(loader, "parse_strict", fake_parse_strict)
    result = loader.load_file(path)

    assert calls == [text]
    assert result == {"db": {"host": "localhost", "port": 5432}}


def test_load_layers_uses_parse_strict_for_every_ini_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "a.ini"
    first.write_text("[db]\nhost = a\n", encoding="utf-8")
    second = tmp_path / "b.ini"
    second.write_text("[db]\nhost = b\n", encoding="utf-8")

    seen: list[str] = []

    def fake_parse_strict(raw: str) -> dict[str, Any]:
        seen.append(raw)
        return {"db": {"host": raw.strip().splitlines()[-1].split("=")[-1].strip()}}

    monkeypatch.setattr(loader, "parse_strict", fake_parse_strict)
    merged = loader.load_layers([first, second])

    assert len(seen) == 2
    assert merged == {"db": {"host": "b"}}
