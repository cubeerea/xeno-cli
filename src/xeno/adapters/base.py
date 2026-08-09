"""LanguageAdapter interface (PRD S12).

"Language scope" means which languages the harness can competently WRITE,
TEST, and VALIDATE code in. Release v1 ships one implementation
(`PythonAdapter`); the interface exists so Talos's gate chain and the
sandbox never hardcode a toolchain.

A polyglot repository selects one primary adapter per run (PRD S12,
non-goal N10) — `detect()` is how the harness chooses it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TestSummary:
    """Structured extraction from a test-runner log (PRD S12 parse_failures)."""

    tests_run: int
    tests_failed: int
    failing_test_ids: tuple[str, ...]
    exception_type: str
    failing_location: str


class LanguageAdapter(ABC):
    """One language's toolchain, exposed as fixed argv builders.

    Every `*_cmd` method returns a fixed argument vector, never a shell
    string (PRD T3, S10: no model-authored shell strings; a `LanguageAdapter`
    is the harness's own fixed vector, not the model's).
    """

    @abstractmethod
    def detect(self, repo_path: Path) -> bool:
        """True if this adapter can competently handle `repo_path`."""

    @abstractmethod
    def image(self) -> str:
        """Docker image tag for the sandbox (PRD S11.1). Built or pulled once
        and cached locally by the Docker daemon; `ensure_image` in
        `xeno.sandbox.pool` is what actually triggers the build."""

    @abstractmethod
    def dockerfile(self) -> str | None:
        """Dockerfile text to build `image()`, or None if it should be
        pulled from a registry instead of built locally."""

    def grammar(self) -> Any:
        """Tree-sitter grammar for symbol extraction (PRD S12).

        This is Argus's tool, not Talos's or Chiron's — Argus does not exist
        until Phase 3 (PRD S10, S13), so there is no caller yet. Declared here
        for interface completeness; raises until Argus lands.
        """
        raise NotImplementedError("grammar() is reserved for Argus (PRD S13 Phase 3)")

    @abstractmethod
    def parse_cmd(self, paths: list[str]) -> list[str] | None:
        """Fixed argv for a syntax-only parse check (gate 1, PRD S10), or
        None if this adapter has no separate parse step."""

    @abstractmethod
    def lint_cmd(self, paths: list[str]) -> list[str]:
        """Fixed argv for gate 2 (style, obvious errors, unused imports)."""

    @abstractmethod
    def parse_lint(self, output: str) -> int:
        """Number of lint findings in `lint_cmd`'s output."""

    @abstractmethod
    def lint_signature(self, output: str) -> str:
        """Stable identity of the findings in `lint_cmd`'s output — which
        rule(s) fired and where, not just how many (empty string when there
        are none). CB-4 (PRD S7.3) hashes this alongside the test-failure
        signature so it can tell "the same lint defect persists" apart from
        "a different lint defect replaced it" — a distinction `parse_lint`'s
        bare count cannot make."""

    @abstractmethod
    def typecheck_cmd(self, paths: list[str]) -> list[str]:
        """Fixed argv for gate 3 (undefined names, signature mismatches)."""

    @abstractmethod
    def parse_typecheck(self, output: str) -> int:
        """Number of type errors in `typecheck_cmd`'s output."""

    @abstractmethod
    def type_signature(self, output: str) -> str:
        """Stable identity of the findings in `typecheck_cmd`'s output, same
        contract as `lint_signature`."""

    @abstractmethod
    def test_cmd(self) -> list[str]:
        """Fixed argv for gate 4 (behavior), over the whole worktree."""

    @abstractmethod
    def parse_test_failures(self, output: str) -> TestSummary:
        """Structured extraction from `test_cmd`'s output (PRD S12
        parse_failures)."""

    @abstractmethod
    def coverage_cmd(self) -> list[str] | None:
        """Fixed argv for gate 5 (advisory, non-blocking), or None if this
        adapter has no coverage tool wired up."""

    @abstractmethod
    def parse_coverage(self, output: str) -> float | None:
        """Total coverage percentage from `coverage_cmd`'s output, or None if
        it could not be determined."""

    @abstractmethod
    def install_cmd(self, worktree: Path) -> list[str] | None:
        """Fixed argv to install the worktree's declared dependencies, or
        None if it declares none. Only ever run with network access, under
        the S11.2 'install' policy."""
