"""Reading configuration documents from disk and layering them together.

Two formats are understood:

``.json``
    Parsed with the standard library :mod:`json` module. The top-level value
    must be an object.

``.ini`` / ``.cfg``
    Parsed by :func:`parse_ini`, a small hand-rolled reader. Section headers
    become nested dictionaries (``[db.pool]`` nests under ``db``) and every
    value is kept as a raw string.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from configkit.errors import ParseError
from configkit.merge import deep_merge

COMMENT_PREFIXES = ("#", ";")
"""Line prefixes that mark a comment in an INI document."""

INI_SUFFIXES = frozenset({".ini", ".cfg"})
"""File suffixes routed to :func:`parse_ini`."""


def parse_ini(text: str, *, source: str = "<string>") -> dict[str, Any]:
    """Parse an INI-style document into a nested dictionary of strings.

    Keys appearing before the first section header land at the top level. A
    dotted section name such as ``[db.pool]`` creates nested dictionaries.
    """
    root: dict[str, Any] = {}
    section: dict[str, Any] = root
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(COMMENT_PREFIXES):
            continue
        if line.startswith("["):
            section = _open_section(root, line, source=source, lineno=lineno)
            continue
        if "=" not in line:
            raise ParseError(
                f"expected 'key = value', got {line!r}",
                source=source,
                line=lineno,
            )
        raw_key, _, raw_value = line.partition("=")
        key = raw_key.strip()
        if not key:
            raise ParseError("empty key name", source=source, line=lineno)
        section[key] = raw_value.strip()
    return root


def _open_section(
    root: dict[str, Any],
    header: str,
    *,
    source: str,
    lineno: int,
) -> dict[str, Any]:
    """Resolve a ``[section.name]`` header to the dictionary it addresses."""
    if not header.endswith("]"):
        raise ParseError("unterminated section header", source=source, line=lineno)
    name = header[1:-1].strip()
    if not name:
        raise ParseError("empty section header", source=source, line=lineno)
    node: dict[str, Any] = root
    for raw_part in name.split("."):
        part = raw_part.strip()
        if not part:
            raise ParseError(
                f"empty component in section header {name!r}",
                source=source,
                line=lineno,
            )
        child = node.get(part)
        if isinstance(child, dict):
            nxt: dict[str, Any] = child
        elif child is None:
            nxt = {}
            node[part] = nxt
        else:
            raise ParseError(
                f"section {part!r} conflicts with an existing value",
                source=source,
                line=lineno,
            )
        node = nxt
    return node


def parse_json(text: str, *, source: str = "<string>") -> dict[str, Any]:
    """Parse a JSON document whose top-level value must be an object."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(exc.msg, source=source, line=exc.lineno) from exc
    if not isinstance(data, dict):
        raise ParseError(
            f"top-level JSON value must be an object, got {type(data).__name__}",
            source=source,
        )
    return dict(data)


def load_file(path: str | Path) -> dict[str, Any]:
    """Read a single configuration file, dispatching on its suffix."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix != ".json" and suffix not in INI_SUFFIXES:
        raise ParseError(
            f"unsupported configuration format {file_path.suffix!r}",
            source=str(file_path),
        )
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return parse_json(text, source=str(file_path))
    return parse_ini(text, source=str(file_path))


def load_layers(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Load every path in order and deep-merge the results, later paths winning."""
    result: dict[str, Any] = {}
    for path in paths:
        result = deep_merge(result, load_file(path))
    return result


def existing_layers(paths: Sequence[str | Path]) -> list[Path]:
    """Return the subset of ``paths`` that currently exist on disk, in order."""
    return [Path(p) for p in paths if Path(p).is_file()]
