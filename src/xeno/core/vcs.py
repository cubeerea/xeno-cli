"""Git wrapper for the checkpoint substrate (PRD S13 Phase 3) and the full
git layer (PRD S8.4, S13 Phase 4): branch naming, squashing, and optional PR
creation on top of it.

Every call is a fixed argument vector, never a shell string — the same rule
the adapter layer follows for the same reason (PRD T3): nothing here is
built from model-authored text. `open_pr`'s title/body ARE model-authored
(Cerberus's commit message), but that is not the same hazard: passing them
as argv elements to `subprocess.run` with `shell=False` never invokes a
shell, so there is no metacharacter-injection surface regardless of content.
"""

from __future__ import annotations

import os
import shutil
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


def root_commit(worktree: Path) -> str:
    """The run's starting commit (from `init_repo`) — recovered by walking
    history rather than threaded through as extra state, for callers (like
    `xeno.cli`, after `run_graph` has already returned) that only have the
    worktree, not the closure-local `initial_sha` `build_graph` computed.
    Safe because this repo's history is always linear and single-rooted: one
    `init_repo` commit, then only `commit`/`squash_to_one_commit` calls —
    never a merge.
    """
    return _run(worktree, ["rev-list", "--max-parents=0", "HEAD"])


def create_branch(worktree: Path, name: str) -> str:
    """Move the run onto its dedicated branch (PRD S8.4:
    `xeno/<slug>-<short_run_id>`). Idempotent like `init_repo`: a second call
    with the same name is a no-op, since nothing about re-entering a run
    already on its own branch should fail.
    """
    current = _run(worktree, ["rev-parse", "--abbrev-ref", "HEAD"])
    if current == name:
        return name
    _run(worktree, ["checkout", "-q", "-b", name])
    return name


def diff_since(worktree: Path, sha: str) -> str:
    """The full accumulated diff from `sha` to the CURRENT worktree state —
    Cerberus's review target (PRD S8.2) and the human gate's diff summary.

    `git add -A` before diffing mirrors exactly what `commit()` already does
    before every checkpoint, and matters here specifically: Cerberus can be
    entered mid-attempt with uncommitted changes (a halt reached before a
    checkpoint), and those must be visible in "the current diff" a human or
    the reviewer sees. `git diff --cached` alone would miss committed history
    before the working-tree edit; `git diff <sha> HEAD` alone would miss the
    uncommitted edit. Staging first and diffing against the index captures
    both in one pass.
    """
    _run(worktree, ["add", "-A"])
    return _run(worktree, ["diff", "--cached", sha])


def squash_to_one_commit(worktree: Path, *, since: str, message: str) -> str:
    """On APPROVE (PRD S8.4): collapse every checkpoint commit made since
    `since` into one commit with Cerberus's conventional-commit message.

    `reset --soft` moves the branch pointer back to `since` while leaving the
    index and working tree untouched, so everything checkpointed since
    becomes one large staged diff for the follow-up commit. Chosen over
    `rebase -i` (needs a non-interactive sequence editor to script reliably)
    or `filter-branch` (deprecated upstream, built for rewriting history
    wholesale — overkill for collapsing N commits into 1): two fixed
    commands, no interactive editor, same discipline as everything else here.
    `since` is always reachable as HEAD's ancestor no matter how many
    checkpoints or L3 rollbacks happened in the run: every checkpoint commits
    on top of the current tip, and `reset_hard` only ever targets an
    existing checkpoint sha or `since` itself, so linear ancestry is never
    broken.
    """
    _run(worktree, ["reset", "-q", "--soft", since])
    _run(worktree, ["commit", "-q", "-m", message], env_identity=True)
    return _run(worktree, ["rev-parse", "HEAD"])


def inherit_origin_remote(worktree: Path, repo_root: Path) -> str | None:
    """Best-effort, never raises: copies `repo_root`'s (the user's real repo)
    `origin` URL into the isolated worktree, which starts with zero remotes
    since its `.git` was never part of the copy that built it. Read-only
    against `repo_root` — this never mutates the user's actual repository.
    Returns `None` if there is no origin to copy or the worktree already has
    one; the caller only needs this when `open_pr` is requested.
    """
    probe = subprocess.run(
        ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    url = probe.stdout.strip()
    if probe.returncode != 0 or not url:
        return None
    try:
        _run(worktree, ["remote", "add", "origin", url])
    except GitError:
        return None
    return url


def push_branch(worktree: Path, branch: str) -> bool:
    """Push `branch` to `origin`. Never `--force` (PRD N4: no autonomous
    merge to a protected branch, and force-pushing an isolated branch is
    still a needless way to lose someone else's concurrent work on it).
    Returns `False` rather than raising on any failure — no remote
    configured, no auth, network down — since PR creation is an optional
    convenience (`GitConfig.open_pr`) that must never fail an otherwise
    successful, already-approved run.
    """
    try:
        _run(worktree, ["push", "-u", "origin", branch])
        return True
    except GitError:
        return False


def open_pr(worktree: Path, *, branch: str, title: str, body: str) -> str | None:
    """`gh pr create` on the pushed branch. Returns `None` (never raises) if
    `gh` is missing or the call fails for any reason — the caller has
    already warned upfront (PRD S8.4's PR step is optional) if `gh` is
    absent, and a failed PR creation must not fail the run.
    """
    if shutil.which("gh") is None:
        return None
    result = subprocess.run(
        ["gh", "pr", "create", "--title", title, "--body", body, "--head", branch],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()
