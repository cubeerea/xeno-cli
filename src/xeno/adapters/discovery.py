"""Toolchain discovery (PRD S8.2, S12 revised): Argus proposes the fixed
commands a repository's gates should run, once per repo, cached thereafter.

"THE GATES ARE DETERMINISTIC TOOLS, not model calls" (PRD S8.2) still holds:
the one model call here only ever proposes WHICH argv to run — it never
decides pass/fail, and its output is validated against a fixed executable
allowlist (`_ALLOWED_EXECUTABLES`) before it is ever cached or executed, on
top of the sandbox's own hardening (non-root, read-only rootfs, `cap_drop:
ALL`, network disabled by default). Everything downstream still runs as a
fixed argument vector, never a shell string (same discipline as every other
model-authored text in this codebase, e.g. `xeno.core.vcs`).

Caching is keyed by a fingerprint of the repo's manifest files
(`manifest_fingerprint`), not by run — a cache hit costs zero model calls,
and only changing `pyproject.toml`/`package.json`/etc. invalidates it.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

from xeno.adapters.generic import (
    MANIFEST_FILENAMES,
    DiscoveredCommand,
    DiscoveredToolchain,
)
from xeno.core.config import STATE_DIRNAME, XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.state import AgentState
from xeno.core.types import NodeRole
from xeno.graph.context import build_codebase_map
from xeno.graph.nodeops import MAX_FORMAT_ATTEMPTS, complete_with_format_retry
from xeno.graph.prompts import (
    ARGUS_SYSTEM,
    DISCOVERY_FORMAT_CORRECTION,
    DISCOVERY_PREFIX,
    DiscoveryOutput,
    parse_discovery_output,
)
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router

#: Per-file cap on how much manifest content the prompt shows — a lockfile
#: masquerading as a manifest (rare, but package-lock.json exists) should
#: never blow the prompt budget.
_MAX_MANIFEST_BYTES = 8_000

#: Re-exported from `xeno.adapters.generic`, which owns the list — discovery
#: uses it both to fingerprint the repo's toolchain declaration and to show
#: the model their actual content.
__all__ = [
    "MANIFEST_FILENAMES",
    "DiscoveryError",
    "discover_toolchain",
    "has_manifests",
    "load_cached_toolchain",
    "manifest_fingerprint",
    "save_toolchain",
]

#: Defense in depth on top of the sandbox's own hardening: a hallucinated or
#: injected destructive command (`rm`, `curl`, `sudo`, ...) is rejected
#: before it is ever written to the cache or executed. One entry per
#: ecosystem named in `MANIFEST_FILENAMES` — each is the actual package
#: manager/toolchain entry point for that ecosystem, never a build-backend
#: or plugin name (e.g. Python's `hatchling`, `setuptools`, `poetry-core`),
#: since those are invoked *through* pip, not run directly.
_ALLOWED_EXECUTABLES = frozenset(
    {
        # Python
        "python3",
        "pip",
        "pytest",
        "ruff",
        "mypy",
        # Node
        "node",
        "npm",
        "npx",
        "yarn",
        "pnpm",
        # Rust
        "cargo",
        # Go
        "go",
        # Ruby
        "bundle",
        "gem",
        "rspec",
        # Java
        "mvn",
        "gradle",
        "./gradlew",
        # Make-based / language-agnostic
        "make",
    }
)


#: Test runners recognizable from argv[0] alone.
_TEST_ENTRYPOINTS = frozenset({"pytest", "rspec"})

#: Tools whose test mode is a subcommand: `cargo test`, `go test`, `npm test`,
#: `mvn test`, `gradle test`, `bundle exec rspec`, `npx jest`.
_TEST_SUBCOMMANDS = frozenset({"test", "jest", "vitest", "mocha", "rspec"})


def _classify(name: str, argv: tuple[str, ...]) -> bool:
    """Whether a discovered command runs the repository's tests.

    Two independent signals, because either one alone has a failure mode.
    The label is what the model actually controls and JOB 3 asks it to use
    the word "test" for, but a label is prose and prose drifts. The argv is
    ground truth but has to be read positionally: scanning every token for
    "test" would classify `mypy src tests` — which type-checks a directory
    that happens to be called tests — as a test command, and then hold the
    type checker back from every implementation gate.
    """
    if "test" in name.lower() or "spec" in name.lower():
        return True
    if not argv:
        return False
    if argv[0] in _TEST_ENTRYPOINTS:
        return True
    # Only the first two positions: `npm run test`, `bundle exec rspec`.
    return any(token in _TEST_SUBCOMMANDS for token in argv[1:3])


def _command(name: str, argv: tuple[str, ...]) -> DiscoveredCommand:
    return DiscoveredCommand(name=name, argv=argv, is_test=_classify(name, argv))


class DiscoveryError(Exception):
    """Discovery could not produce a usable toolchain for this repo.

    Raised before the graph or the sandbox pool exist — the caller (`cli.py`)
    treats this as a pre-run setup failure, the same class of error as
    `git`/`gh` missing from PATH, not a graph-level halt.
    """


def manifest_fingerprint(worktree: Path) -> str:
    """Deterministic hash of which manifest files are present and their
    content. Changing a dependency or script invalidates the cache;
    unrelated source edits do not."""
    hasher = hashlib.sha256()
    for name in MANIFEST_FILENAMES:
        path = worktree / name
        if path.is_file():
            hasher.update(name.encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def has_manifests(worktree: Path) -> bool:
    """Whether this repository declares a toolchain at all.

    The greenfield test, and deliberately a purely deterministic one: if no
    manifest exists there is provably nothing for a model to discover, so the
    unestablished result is reached without spending a call. Asking anyway
    puts the model in an unsatisfiable position — JOB 3 forbids inventing a
    toolchain that is not evidenced, while requiring at least one command —
    and the likeliest answer is a hallucinated `pytest`/`npm test` that then
    gates every task in the run against a command the repo never had.
    """
    return any((worktree / name).is_file() for name in MANIFEST_FILENAMES)


def _cache_path(repo_root: Path, fingerprint: str) -> Path:
    return repo_root / STATE_DIRNAME / "discovery" / f"{fingerprint}.json"


def load_cached_toolchain(repo_root: Path, fingerprint: str) -> DiscoveredToolchain | None:
    """Returns `None` on anything short of a fully-formed match — a missing,
    corrupt, or stale-fingerprint cache file is just a cache miss, never an
    error; the caller re-discovers.

    A *well-formed* cache entry that fails the executable allowlist is a
    different matter and raises: this file is advertised as human-editable
    (`save_toolchain`), so it is an input like any other, and skipping
    `_validate` here would mean the allowlist this module presents as its
    safety boundary could be bypassed simply by writing the command to disk
    instead of having a model propose it. Loud beats silent — quietly
    treating it as a miss would re-discover over the top and leave whoever
    (or whatever) wrote it none the wiser.
    """
    path = _cache_path(repo_root, fingerprint)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("fingerprint") != fingerprint:
        return None
    try:
        toolchain = DiscoveredToolchain(
            install=tuple(data["install"]) if data.get("install") else None,
            required=tuple(_command(c["name"], tuple(c["argv"])) for c in data["required"]),
            advisory=tuple(
                _command(c["name"], tuple(c["argv"])) for c in data.get("advisory", [])
            ),
            fingerprint=fingerprint,
        )
    except (KeyError, TypeError):
        return None
    _validate(toolchain, source=str(path))
    return toolchain


def save_toolchain(repo_root: Path, toolchain: DiscoveredToolchain) -> None:
    """Writes human-readable, human-editable JSON — a user who disagrees
    with a discovered command can hand-edit the cache file directly rather
    than fighting the discovery prompt."""
    path = _cache_path(repo_root, toolchain.fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "fingerprint": toolchain.fingerprint,
        "install": list(toolchain.install) if toolchain.install else None,
        "required": [{"name": c.name, "argv": list(c.argv)} for c in toolchain.required],
        "advisory": [{"name": c.name, "argv": list(c.argv)} for c in toolchain.advisory],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def _validate(toolchain: DiscoveredToolchain, *, source: str = "discovery") -> None:
    """The executable allowlist boundary. Runs on BOTH paths a toolchain can
    reach the sandbox by — freshly proposed by a model, and loaded from the
    on-disk cache — since either one ends up as an argv executed in a
    container. `source` names which, so the error says where to look."""
    if not toolchain.required:
        raise DiscoveryError(f"{source} produced no required commands to gate on")
    commands = (*toolchain.required, *toolchain.advisory)
    for command in commands:
        if not command.argv:
            raise DiscoveryError(f"{source}: command '{command.name}' has an empty argv")
        if command.argv[0] not in _ALLOWED_EXECUTABLES:
            raise DiscoveryError(
                f"{source}: command '{command.name}' proposed disallowed executable "
                f"'{command.argv[0]}' — allowed: {sorted(_ALLOWED_EXECUTABLES)}"
            )
    if toolchain.install is not None:
        if not toolchain.install:
            raise DiscoveryError(f"{source}: install command has an empty argv")
        if toolchain.install[0] not in _ALLOWED_EXECUTABLES:
            raise DiscoveryError(
                f"{source}: install command proposed disallowed executable "
                f"'{toolchain.install[0]}' — allowed: {sorted(_ALLOWED_EXECUTABLES)}"
            )


def _tokenize(argv_text: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(argv_text))
    except ValueError:
        return ()


def _build_toolchain(output: DiscoveryOutput, fingerprint: str) -> DiscoveredToolchain:
    install = _tokenize(output.install) if output.install else ()
    return DiscoveredToolchain(
        install=install or None,
        required=tuple(_command(name, _tokenize(argv_text)) for name, argv_text in output.required),
        advisory=tuple(_command(name, _tokenize(argv_text)) for name, argv_text in output.advisory),
        fingerprint=fingerprint,
    )


def _manifest_contents(worktree: Path) -> str:
    chunks = []
    for name in MANIFEST_FILENAMES:
        path = worktree / name
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")[:_MAX_MANIFEST_BYTES]
        chunks.append(f"--- {name} ---\n{text}")
    return "\n\n".join(chunks) if chunks else "(no known manifest file found at the repo root)"


def discover_toolchain(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    state: AgentState,
    paths: RunPaths,
    repo_root: Path,
    worktree: Path,
    allow_unestablished: bool = False,
) -> DiscoveredToolchain:
    """Cache lookup first; on a miss, one Argus JOB 3 call (`NodeRole.
    RESEARCHER`), validated and cached before being returned.

    Raises `DiscoveryError` if the model's chain is exhausted or its output
    stays malformed/unsafe after the standard corrective retry — there is
    nothing safe to gate a run on otherwise.

    `allow_unestablished` covers the greenfield case: a repository with no
    manifest at all resolves to `DiscoveredToolchain.unestablished` instead
    of raising, letting a run start against an empty directory and scaffold
    its own toolchain as its first task. The distinction matters — a repo
    with manifests whose discovery fails is still a hard error, because
    something IS declared there and we could not read it.
    """
    fingerprint = manifest_fingerprint(worktree)
    cached = load_cached_toolchain(repo_root, fingerprint)
    if cached is not None:
        return cached

    if not has_manifests(worktree):
        if not allow_unestablished:
            raise DiscoveryError(
                "no manifest or build file found at the repository root "
                f"(looked for: {', '.join(MANIFEST_FILENAMES)}) — nothing to discover a "
                "toolchain from"
            )
        # Deliberately NOT cached: "there is nothing here" is cheap to
        # recompute, and every manifest-less repo shares one fingerprint, so
        # writing it would persist a stale answer past the very first commit
        # that establishes a toolchain.
        return DiscoveredToolchain.unestablished(fingerprint)

    builder = PromptBuilder(
        node=NodeRole.RESEARCHER,
        keyring=keyring,
        system_text=ARGUS_SYSTEM,
        caching_enabled=config.caching.enabled,
    )
    builder.set_codebase_map(
        build_codebase_map(worktree, max_content_bytes=0), require_fresh=False
    )
    current_turn = "\n\n".join([DISCOVERY_PREFIX, _manifest_contents(worktree)])

    try:
        output = complete_with_format_retry(
            router=router,
            builder=builder,
            node=NodeRole.RESEARCHER,
            state=state,
            paths=paths,
            current_turn=current_turn,
            correction=DISCOVERY_FORMAT_CORRECTION,
            parse=parse_discovery_output,
        )
    except ChainExhaustedError as exc:
        raise DiscoveryError(f"toolchain discovery: {exc}") from exc

    if output.malformed:
        # The one caller that names the saved response itself: discovery runs
        # BEFORE the graph, so its failure exits through the CLI's error path
        # and never reaches the halt panel where every other node's saved
        # response is reported from `state.unparsed_response`.
        saved = state.unparsed_response
        raise DiscoveryError(
            "toolchain discovery produced no usable required-command list after "
            f"{MAX_FORMAT_ATTEMPTS} attempt(s)"
            + (f"; raw response saved to {saved.path}" if saved is not None else "")
        )

    toolchain = _build_toolchain(output, fingerprint)
    _validate(toolchain)
    save_toolchain(repo_root, toolchain)
    return toolchain
