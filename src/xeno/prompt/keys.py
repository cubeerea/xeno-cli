"""Cache-key derivation and invalidation (PRD S9.6.5).

Cache keys are derived, never manually managed:

    system_cache_key   = hash(node_role, xeno_version, system_prompt_text)
    codebase_cache_key = hash(run_id, worktree_head_content_hash)

The second one carries the sharp edge. A stale codebase map served from cache
is strictly worse than a cache miss, because Argus's summaries would describe
pre-edit code with full confidence. So the map is invalidated the moment
Daedalus or Chiron writes to the worktree — checkpoints included — and this
module refuses to hand out a key it knows to be stale rather than leaving that
discipline to each caller.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from xeno import XENO_VERSION
from xeno.core.types import NodeRole

#: Directories never included in the worktree content hash: build noise and
#: harness state churn constantly and would invalidate the map on every call.
DEFAULT_IGNORES: frozenset[str] = frozenset(
    {
        ".git",
        ".xeno",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "node_modules",
        ".tox",
        "dist",
        "build",
        ".DS_Store",
    }
)


class StaleCodebaseMapError(RuntimeError):
    """Raised when a codebase map is used after the worktree changed under it.

    Not a warning. PRD S9.6.5 makes re-fetching from Argus after every source
    write mandatory, even at the cost of a cache miss on that layer.
    """


def system_cache_key(
    node: NodeRole, system_prompt_text: str, *, version: str = XENO_VERSION
) -> str:
    """Stable across the life of a Xeno CLI version, per node.

    Including the version means an upgrade deliberately busts every system
    prompt cache — serving a prior version's system prompt would be a
    correctness bug wearing a performance win's clothing.
    """
    return _digest("system", node.value, version, system_prompt_text)


def codebase_cache_key(run_id: str, worktree_content_hash: str) -> str:
    """Scoped to the run, so two concurrent runs never share a map."""
    return _digest("codebase", run_id, worktree_content_hash)


def law_cache_key(run_id: str, law_text: str) -> str:
    """Keyed on the law text ITSELF, not on a dirty flag.

    The codebase map can afford `mark_worktree_written` because there is
    exactly one class of writer (Daedalus and Chiron, through the worktree) and
    the map is far too large to re-hash per call. Project law has neither
    property: it is a few kilobytes, and it changes through channels no write
    signal covers — a human editing `.xeno/memory.md` between runs, an
    extraction step appending a preference, Odysseus rewriting the roadmap on a
    re-plan. A dirty flag would have to be raised correctly at every one of
    those sites, and the failure mode when one is missed is the worst kind:
    a cache hit serving the PREVIOUS law, so a node confidently obeys a
    preference the human just changed, with nothing in the log to show it.

    Hashing the content makes the key self-invalidating, so there is no
    discipline left for a caller to get wrong.
    """
    return _digest("law", run_id, hashlib.sha256(law_text.encode()).hexdigest())


def history_cache_key(run_id: str, node: NodeRole, turn_index: int) -> str:
    """Append-only: the key advances with the turn count, so the prefix through
    the prior turn stays cache-eligible (PRD S9.6.1)."""
    return _digest("history", run_id, node.value, str(turn_index))


def worktree_content_hash(
    root: Path,
    *,
    ignores: Iterable[str] = DEFAULT_IGNORES,
    max_file_bytes: int = 2_000_000,
) -> str:
    """Content hash of the worktree: every tracked path plus its digest.

    Hashes content rather than mtimes so a touch-without-edit does not
    needlessly invalidate the map, and so the value is reproducible across
    machines when replaying a run from the JSONL log.
    """
    ignore_set = set(ignores)
    hasher = hashlib.sha256()
    for path in sorted(_walk(root, ignore_set)):
        rel = path.relative_to(root).as_posix()
        hasher.update(rel.encode())
        hasher.update(b"\0")
        try:
            size = path.stat().st_size
            if size > max_file_bytes:
                # Oversized files (checked-in binaries, fixtures) are keyed by
                # size alone. Argus would not inline them anyway.
                hasher.update(f"size:{size}".encode())
            else:
                hasher.update(hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            hasher.update(b"unreadable")
        hasher.update(b"\0")
    return hasher.hexdigest()


def _walk(root: Path, ignore_set: set[str]) -> Iterable[Path]:
    for entry in root.iterdir():
        if entry.name in ignore_set:
            continue
        if entry.is_symlink():
            continue
        if entry.is_dir():
            yield from _walk(entry, ignore_set)
        elif entry.is_file():
            yield entry


class CacheKeyring:
    """Per-run owner of cache keys and codebase-map freshness.

    Nodes with SOURCE write authority (Daedalus, Chiron) must call
    `mark_worktree_written` after every write. Everything downstream of that
    signal is automatic.
    """

    def __init__(self, run_id: str, worktree_root: Path) -> None:
        self.run_id = run_id
        self.worktree_root = worktree_root
        self._content_hash: str | None = None
        self._dirty = True
        self._history_turns: dict[NodeRole, int] = {}
        self.invalidations: list[str] = []

    @property
    def codebase_is_stale(self) -> bool:
        return self._dirty

    def content_hash(self) -> str:
        if self._dirty or self._content_hash is None:
            self._content_hash = worktree_content_hash(self.worktree_root)
            self._dirty = False
        return self._content_hash

    def codebase_key(self) -> str:
        return codebase_cache_key(self.run_id, self.content_hash())

    def law_key(self, law_text: str) -> str:
        return law_cache_key(self.run_id, law_text)

    def system_key(self, node: NodeRole, system_prompt_text: str) -> str:
        return system_cache_key(node, system_prompt_text)

    def history_key(self, node: NodeRole) -> str:
        return history_cache_key(self.run_id, node, self._history_turns.get(node, 0))

    def advance_history(self, node: NodeRole) -> int:
        turn = self._history_turns.get(node, 0) + 1
        self._history_turns[node] = turn
        return turn

    def mark_worktree_written(
        self, paths: Sequence[Path] | None = None, *, reason: str = ""
    ) -> None:
        """Invalidate the codebase-map layer. Mandatory after any source write.

        Cheap to call redundantly and expensive to skip, so callers should err
        toward calling it.
        """
        self._dirty = True
        detail = reason or (
            ", ".join(str(p) for p in paths[:5]) if paths else "unspecified worktree write"
        )
        self.invalidations.append(detail)

    def assert_codebase_fresh(self) -> None:
        if self._dirty:
            raise StaleCodebaseMapError(
                "the worktree changed since the codebase map was built; re-run Argus "
                "before assembling this prompt (PRD S9.6.5)"
            )


def _digest(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode())
        hasher.update(b"\0")
    return hasher.hexdigest()
