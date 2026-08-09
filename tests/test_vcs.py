"""`xeno.core.vcs` (PRD S13 Phase 3): the checkpoint substrate. Exercised
against a real git binary and a real temp repo, not mocked — the same
philosophy `test_sandbox.py` uses for live Docker, since what matters here
is that real git behaves the way the checkpoint/rollback logic assumes."""

from __future__ import annotations

from pathlib import Path

import pytest

from xeno.core import vcs


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    d = tmp_path / "repo"
    d.mkdir()
    return d


def test_init_repo_creates_a_commit_of_the_starting_state(repo: Path) -> None:
    (repo / "a.txt").write_text("hello\n")
    sha = vcs.init_repo(repo)
    assert len(sha) == 40
    assert (repo / ".git").is_dir()


def test_init_repo_works_on_an_empty_directory(repo: Path) -> None:
    sha = vcs.init_repo(repo)
    assert len(sha) == 40


def test_init_repo_is_idempotent(repo: Path) -> None:
    (repo / "a.txt").write_text("hello\n")
    first = vcs.init_repo(repo)
    second = vcs.init_repo(repo)
    assert first == second


def test_commit_returns_a_new_sha_and_captures_new_content(repo: Path) -> None:
    initial = vcs.init_repo(repo)
    (repo / "a.txt").write_text("v2\n")
    second = vcs.commit(repo, "second")
    assert second != initial


def test_commit_allows_an_empty_diff(repo: Path) -> None:
    """A checkpoint must exist even when a task's patch net out to nothing
    (e.g. Chiron reverted a file back to its prior content) — L3's next
    rollback target still needs to be this commit, not the one before it."""
    initial = vcs.init_repo(repo)
    second = vcs.commit(repo, "no-op checkpoint")
    assert second != initial


def test_reset_hard_discards_tracked_changes(repo: Path) -> None:
    (repo / "a.txt").write_text("v1\n")
    initial = vcs.init_repo(repo)
    (repo / "a.txt").write_text("v2\n")
    vcs.reset_hard(repo, initial)
    assert (repo / "a.txt").read_text() == "v1\n"


def test_reset_hard_removes_untracked_files(repo: Path) -> None:
    """The `git clean -fd` half of L3's rollback: a failed attempt's new
    files were never committed, so a reset alone would leave them behind."""
    initial = vcs.init_repo(repo)
    (repo / "new.txt").write_text("never committed\n")
    vcs.reset_hard(repo, initial)
    assert not (repo / "new.txt").exists()


def test_git_error_includes_stderr(repo: Path) -> None:
    vcs.init_repo(repo)
    with pytest.raises(vcs.GitError, match="git"):
        vcs.reset_hard(repo, "0" * 40)
