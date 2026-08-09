"""Minimal git wrapper for the Phase 3 checkpoint substrate (PRD S8.4, S13).

"Minimal" is deliberate: PRD S13 Phase 3 delivers only what checkpointing
and L3 rollback need (commit, hard reset). Branch naming, squashing, and PR
creation are a Phase 4 concern, built on top of this later.

Every call is a fixed argument vector, never a shell string — the same rule
`LanguageAdapter` follows for the same reason (PRD T3): nothing here is
built from model-authored text.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

#: A commit identity is required for `git commit` to succeed in an
#: environment with no global user.name/user.email configured (e.g. CI, or
#: a fresh machine) — the throwaway worktree is not a place a human commits
#: from, so a fixed harness identity is correct, not a workaround.
_GIT_ENV_IDENTITY = {
    "GIT_AUTHOR_NAME": "xeno-cli",
    "GIT_AUTHOR_EMAIL": "xeno-cli@localhost",
    "GIT_COMMITTER_NAME": "xeno-cli",
    "GIT_COMMITTER_EMAIL": "xeno-cli@localhost",
}


class GitError(RuntimeError):
    """A git subprocess exited non-zero."""


def _run(worktree: Path, args: list[str], *, env_identity: bool = False) -> str:
    env = {**os.environ, **_GIT_ENV_IDENTITY} if env_identity else None
    result = subprocess.run(
        ["git", *args],
        cwd=worktree,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def init_repo(worktree: Path) -> str:
    """Initialize the checkpoint substrate: a fresh repo with one commit of
    whatever `worktree` looked like when the run started.

    Idempotent — a second call against an already-initialized worktree is a
    no-op that just returns HEAD, since `xeno.graph.build` calls this once
    per run and there is no scenario where re-running it should discard the
    run's own history.
    """
    if (worktree / ".git").exists():
        return _run(worktree, ["rev-parse", "HEAD"])
    _run(worktree, ["init", "-q"])
    _run(worktree, ["add", "-A"])
    _run(
        worktree,
        ["commit", "-q", "-m", "xeno: initial snapshot", "--allow-empty"],
        env_identity=True,
    )
    return _run(worktree, ["rev-parse", "HEAD"])


def commit(worktree: Path, message: str) -> str:
    """Checkpoint the worktree's current state, returning the new commit sha.

    `--allow-empty`: a task can pass Talos's gates with a no-op diff (e.g. a
    Chiron patch that reverted a file back to its prior green content), and
    a checkpoint must still exist to advance `task_cursor` and give L3 a
    rollback target for the NEXT task.
    """
    _run(worktree, ["add", "-A"])
    _run(worktree, ["commit", "-q", "-m", message, "--allow-empty"], env_identity=True)
    return _run(worktree, ["rev-parse", "HEAD"])


def reset_hard(worktree: Path, sha: str) -> None:
    """L3 rollback: discard everything since `sha`, tracked and untracked.

    `git clean -fd` matters as much as the reset: a failed attempt's new
    files were never committed (checkpoints only happen on a Talos pass), so
    a reset alone would leave them behind for the next attempt to trip over.
    """
    _run(worktree, ["reset", "-q", "--hard", sha])
    _run(worktree, ["clean", "-q", "-fd"])
