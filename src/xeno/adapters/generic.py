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

from xeno.core.types import GateProfile

#: PRD S11.1: one base image, built once and cached locally by the Docker
#: daemon. Broader than the old PythonAdapter's image on purpose — adding
#: language #N means adding a runtime to THIS Dockerfile, not writing a new
#: adapter class.
_IMAGE_TAG = "xeno-generic-adapter:1"

#: The build/manifest files that declare a repository's toolchain. Lives here
#: rather than in `xeno.adapters.discovery` because three unrelated layers
#: need it — discovery (fingerprint + prompt content), the sandbox's
#: dependency-image cache key, and the greenfield check — and this module is
#: the one with no heavy imports for them to pull in.
MANIFEST_FILENAMES = (
    "pyproject.toml",
    "requirements.txt",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Gemfile",
    "pom.xml",
    "build.gradle",
    "Makefile",
)

#: Same hardening pattern as the old PythonAdapter.DOCKERFILE (non-root user,
#: slim base) extended with a Node.js runtime. Pinned versions so the gate
#: toolchain does not silently drift between runs.
#:
#: Node arrives by multi-stage copy rather than the usual
#: `apt-get install curl gnupg` -> nodesource -> `apt-get install nodejs`
#: chain. That chain made building the ONE image every gate run depends on
#: contingent on three network services and a Debian mirror agreeing with
#: each other; in practice a mirror serving an index whose hash did not match
#: the package it then served (same filesize, different SHA) failed the build
#: outright, and with it every `xeno run` on the machine. Copying from the
#: official Node image needs no package manager at all, so there is nothing
#: left to be stale. Both stages pin the same Debian release, which is what
#: makes the copied binary's glibc match the base it lands on.
DOCKERFILE = """\
FROM node:20-bookworm-slim AS node
FROM python:3.11-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \\
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
RUN pip install --no-cache-dir ruff==0.5.* mypy==1.10.* pytest==8.* pytest-cov==5.*
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

    `is_test` is the one classification the harness does need, and it is
    DERIVED rather than declared: `xeno.adapters.discovery._classify` decides
    it from the label and the argv on both the freshly-discovered and the
    cached path. Deriving it keeps the on-disk cache format unchanged, so
    every discovery file written before this field existed still loads.
    """

    name: str
    argv: tuple[str, ...]
    #: Whether this command runs the repository's tests, and so must be held
    #: back until the milestone it belongs to has tests to run (`xeno.core.
    #: types.GateProfile`).
    is_test: bool = False


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
    #: fingerprint its filename encodes. It is also the staleness key: a
    #: mid-run write that changes a manifest makes this fingerprint no longer
    #: match the worktree, which is what triggers re-discovery
    #: (`xeno.graph.toolchain.ToolchainSession`).
    fingerprint: str

    @classmethod
    def unestablished(cls, fingerprint: str) -> DiscoveredToolchain:
        """The greenfield case: a repository with no manifest, and therefore
        nothing for the gates to run *yet*.

        This is a real state, not an error and not an empty success. A run
        may legitimately start here — the harness's own first task is then to
        scaffold a toolchain into existence — but nothing may ever be
        reported as *passing* against it, which is why `run_gates` refuses an
        unestablished toolchain outright rather than falling through its
        loops to a vacuous green (`xeno.graph.gates`).
        """
        return cls(install=None, required=(), advisory=(), fingerprint=fingerprint)

    @property
    def established(self) -> bool:
        """True once there is at least one required command to gate on.

        `required` being non-empty IS the definition — there is deliberately
        no separate boolean that could drift out of sync with the command
        list it describes.
        """
        return bool(self.required)

    def required_for(self, profile: GateProfile) -> tuple[DiscoveredCommand, ...]:
        """The required commands this profile runs, never an empty tuple.

        A repository whose ONLY required command is its test command (`npm
        test` and nothing else is a perfectly ordinary package.json) would
        otherwise reduce to zero commands under `IMPLEMENTATION`, and a gate
        chain that executes nothing reports a pass. Falling back to the full
        list makes the degenerate case merely stricter than intended rather
        than silently vacuous — the direction an error here has to fail in.
        """
        if profile is GateProfile.FULL:
            return self.required
        return tuple(c for c in self.required if not c.is_test) or self.required


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
