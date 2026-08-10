"""GenericAdapter (PRD S12, revised): one language-agnostic adapter, backed
by a per-repo `DiscoveredToolchain` instead of a hand-written class per
language.

The gates stay deterministic tools (PRD S8.2) — nothing here or in
`xeno.graph.gates` calls a model. What changed is *who decides which fixed
argv to run*: `xeno.adapters.discovery` asks a model once per repo (cached),
never per gate call, and never for the pass/fail verdict itself. This module
only holds the resulting data and the one concrete adapter class that
executes it — see `xeno.adapters.discovery` for how a `DiscoveredToolchain`
gets built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: PRD S11.1: one base image, built once and cached locally by the Docker
#: daemon. Broader than the old PythonAdapter's image on purpose — adding
#: language #N means adding a runtime to THIS Dockerfile, not writing a new
#: adapter class.
_IMAGE_TAG = "xeno-generic-adapter:1"

#: Same hardening pattern as the old PythonAdapter.DOCKERFILE (non-root user,
#: slim base) extended with a Node.js runtime. Pinned versions so the gate
#: toolchain does not silently drift between runs.
DOCKERFILE = """\
FROM python:3.11-slim
RUN pip install --no-cache-dir ruff==0.5.* mypy==1.10.* pytest==8.* pytest-cov==5.*
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y --no-install-recommends nodejs
RUN apt-get purge -y curl gnupg && rm -rf /var/lib/apt/lists/*
RUN useradd --uid 65532 --create-home --shell /usr/sbin/nologin xeno
USER xeno
WORKDIR /workspace
"""


@dataclass(frozen=True, slots=True)
class DiscoveredCommand:
    """One fixed argv, executed inside the sandbox by `xeno.sandbox.pool`.

    `name` is a free-form label (e.g. "lint", "typecheck", "test", "build")
    — there is no fixed enum of phases, because not every ecosystem separates
    them the same way (a `go build` is parse+typecheck in one step; plain JS
    has no separate type-check phase at all).
    """

    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscoveredToolchain:
    """The cached result of `xeno.adapters.discovery.discover_toolchain`.

    `required` runs in order, fail-fast (PRD S10 gate-order rationale,
    generalized: there is nothing for a linter to say about code that does
    not even parse or build). `advisory` runs only if every required command
    passed, and never flips the verdict — the same "advisory, non-blocking"
    contract PRD S10 gate 5 (coverage) already had, generalized to any
    advisory command instead of one hardcoded coverage step.
    """

    install: tuple[str, ...] | None
    required: tuple[DiscoveredCommand, ...]
    advisory: tuple[DiscoveredCommand, ...]
    #: The manifest fingerprint this toolchain was discovered from
    #: (`xeno.adapters.discovery.manifest_fingerprint`) — stored alongside the
    #: commands so a loaded cache entry can be sanity-checked against the
    #: fingerprint its filename encodes.
    fingerprint: str


class GenericAdapter:
    """Wraps a `DiscoveredToolchain`. Not an ABC — there is exactly one
    concrete adapter now; per-language subclassing is gone by design."""

    def __init__(self, toolchain: DiscoveredToolchain) -> None:
        self.toolchain = toolchain

    def image(self) -> str:
        return _IMAGE_TAG

    def dockerfile(self) -> str | None:
        return DOCKERFILE

    def install_cmd(self, worktree: Path) -> list[str] | None:
        del worktree  # discovery already resolved this once per repo
        return list(self.toolchain.install) if self.toolchain.install else None
