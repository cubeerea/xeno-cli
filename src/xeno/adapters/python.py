"""PythonAdapter (PRD S12): ruff, mypy, pytest, pip.

Release v1's only `LanguageAdapter`. Every `*_cmd` here is a fixed argument
vector executed inside the sandbox by `xeno.sandbox.pool` — this module never
runs a subprocess itself, so it stays usable from both the sandboxed Talos
gate chain and a host-side unit test that just checks the argv shape.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from xeno.adapters.base import LanguageAdapter, TestSummary

#: PRD S11.1: "Base image built per LanguageAdapter, cached locally." Pinned
#: minor version so the gate toolchain does not silently drift between runs.
_IMAGE_TAG = "xeno-python-adapter:3.11"

#: The interpreter INSIDE the sandbox container, not `sys.executable` — that
#: would be the harness's own (host) interpreter, a path that means nothing
#: once this argv crosses into the container's filesystem. `_IMAGE_TAG`'s
#: Dockerfile is `python:3.11-slim`, which puts `python3` on PATH.
_PYTHON = "python3"

_SUMMARY_RE = re.compile(r"(\d+) (passed|failed|error|skipped)")
#: mypy's error lines are `file:line: error: message [error-code]` — the
#: trailing `[code]` is optional (older mypy, or a note-adjacent error with
#: no code attached), so it is captured but not required to match.
_MYPY_ERROR_RE = re.compile(
    r"^(?P<file>[^:\n]+):(?P<line>\d+): error:.*?(?:\[(?P<code>[\w-]+)\])?$"
)
#: pytest's short summary line is `FAILED <nodeid> - <detail>`, and `<detail>`
#: takes two shapes depending on how the test failed: `ExceptionType: msg`
#: when a real exception propagated, or just the bare assertion expression
#: (e.g. `assert 4 == 5`, no "AssertionError:" prefix) for pytest's rewritten
#: `assert` statements — the overwhelmingly common case in generated test
#: failures. The type prefix is therefore OPTIONAL here; when it is absent,
#: `_parse_failed_line` below defaults to "AssertionError", which is what
#: pytest's rewriter is reporting in that shape either way.
_FAILED_LINE_RE = re.compile(r"^FAILED (\S+) - (?:(\w+(?:\.\w+)*): )?(.*)$", re.MULTILINE)
_COVERAGE_TOTAL_RE = re.compile(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", re.MULTILINE)

#: The Dockerfile for `_IMAGE_TAG`. Built once by `ensure_image` and cached by
#: the Docker daemon's own layer cache thereafter (PRD S11.1).
DOCKERFILE = """\
FROM python:3.11-slim
RUN pip install --no-cache-dir ruff==0.5.* mypy==1.10.* pytest==8.* pytest-cov==5.*
RUN useradd --uid 65532 --create-home --shell /usr/sbin/nologin xeno
USER xeno
WORKDIR /workspace
"""


class PythonAdapter(LanguageAdapter):
    def detect(self, repo_path: Path) -> bool:
        return (repo_path / "pyproject.toml").is_file() or any(repo_path.glob("*.py"))

    def image(self) -> str:
        return _IMAGE_TAG

    def dockerfile(self) -> str | None:
        return DOCKERFILE

    def parse_cmd(self, paths: list[str]) -> list[str] | None:
        if not paths:
            return None
        # py_compile is a real syntax-only check (no execution), matching
        # gate 1's contract (PRD S10: "Tree-sitter catches PARSE ERRORS
        # only"). Using the stdlib compiler in-container keeps this a fixed
        # command vector instead of host-side ast.parse (PRD S13 Phase 2:
        # "zero host execution of generated code").
        return [_PYTHON, "-m", "py_compile", *paths]

    def lint_cmd(self, paths: list[str]) -> list[str]:
        return [_PYTHON, "-m", "ruff", "check", "--output-format=json", *paths]

    def parse_lint(self, output: str) -> int:
        import json

        findings = json.loads(output) if output.strip() else []
        if not isinstance(findings, list):
            raise ValueError(f"unexpected ruff JSON shape: {output[:200]!r}")
        return len(findings)

    def lint_signature(self, output: str) -> str:
        import json

        try:
            findings = json.loads(output) if output.strip() else []
        except json.JSONDecodeError:
            return ""
        if not isinstance(findings, list):
            return ""
        # code + file + row, NOT column: a reflow that shifts the same
        # violation a few characters right is still the same defect, and
        # counting it as "different" would mask real thrash the same way the
        # count-only signature already did.
        parts = sorted(
            f"{f.get('code', '')}:{f.get('filename', '')}:{f.get('location', {}).get('row', '')}"
            for f in findings
            if isinstance(f, dict)
        )
        return ",".join(parts)

    def typecheck_cmd(self, paths: list[str]) -> list[str]:
        return [_PYTHON, "-m", "mypy", "--no-error-summary", "--hide-error-context", *paths]

    def parse_typecheck(self, output: str) -> int:
        return sum(1 for line in output.splitlines() if ": error:" in line)

    def type_signature(self, output: str) -> str:
        parts = []
        for line in output.splitlines():
            if ": error:" not in line:
                continue
            match = _MYPY_ERROR_RE.match(line)
            if match:
                parts.append(f"{match['code'] or ''}:{match['file']}:{match['line']}")
            else:
                parts.append(line.strip())
        return ",".join(sorted(parts))

    def test_cmd(self) -> list[str]:
        return [_PYTHON, "-m", "pytest", "-q", "--tb=short"]

    def parse_test_failures(self, output: str) -> TestSummary:
        counts: dict[str, int] = {}
        for match in _SUMMARY_RE.finditer(output):
            counts[match.group(2)] = counts.get(match.group(2), 0) + int(match.group(1))
        passed = counts.get("passed", 0)
        failed = counts.get("failed", 0) + counts.get("error", 0)

        failing_ids: list[str] = []
        exception_type = ""
        failing_location = ""
        for match in _FAILED_LINE_RE.finditer(output):
            failing_ids.append(match.group(1))
            if not exception_type:
                exception_type = match.group(2) or "AssertionError"
                failing_location = match.group(1)

        return TestSummary(
            tests_run=passed + failed,
            tests_failed=failed,
            failing_test_ids=tuple(failing_ids),
            exception_type=exception_type,
            failing_location=failing_location,
        )

    def coverage_cmd(self) -> list[str] | None:
        return [
            _PYTHON,
            "-m",
            "pytest",
            "-q",
            "--cov=.",
            "--cov-report=term-missing:skip-covered",
        ]

    def parse_coverage(self, output: str) -> float | None:
        match = _COVERAGE_TOTAL_RE.search(output)
        return float(match.group(1)) if match else None

    def install_cmd(self, worktree: Path) -> list[str] | None:
        # PRD S11.2 'install' mode runs this as root against a writable
        # filesystem (xeno.sandbox.profile.install_container_kwargs), the
        # same as the base image's own Dockerfile — no --user needed.
        requirements = worktree / "requirements.txt"
        if requirements.is_file() and requirements.read_text().strip():
            return [_PYTHON, "-m", "pip", "install", "-r", "requirements.txt"]

        pyproject = worktree / "pyproject.toml"
        if pyproject.is_file():
            try:
                data = tomllib.loads(pyproject.read_text())
            except tomllib.TOMLDecodeError:
                return None
            deps = data.get("project", {}).get("dependencies", [])
            if deps:
                return [_PYTHON, "-m", "pip", "install", "-e", "."]
        return None
