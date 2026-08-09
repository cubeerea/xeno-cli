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


def test_root_commit_finds_the_initial_commit(repo: Path) -> None:
    (repo / "a.txt").write_text("v1\n")
    initial = vcs.init_repo(repo)
    (repo / "a.txt").write_text("v2\n")
    vcs.commit(repo, "checkpoint")
    assert vcs.root_commit(repo) == initial


def test_create_branch_moves_head(repo: Path) -> None:
    vcs.init_repo(repo)
    vcs.create_branch(repo, "xeno/add-widget-abc123")
    assert _current_branch(repo) == "xeno/add-widget-abc123"


def test_create_branch_is_idempotent(repo: Path) -> None:
    vcs.init_repo(repo)
    vcs.create_branch(repo, "xeno/add-widget-abc123")
    vcs.create_branch(repo, "xeno/add-widget-abc123")
    assert _current_branch(repo) == "xeno/add-widget-abc123"


def test_diff_since_captures_committed_changes(repo: Path) -> None:
    (repo / "a.txt").write_text("v1\n")
    initial = vcs.init_repo(repo)
    (repo / "a.txt").write_text("v2\n")
    vcs.commit(repo, "checkpoint")
    diff = vcs.diff_since(repo, initial)
    assert "-v1" in diff
    assert "+v2" in diff


def test_diff_since_captures_uncommitted_and_untracked_changes(repo: Path) -> None:
    (repo / "a.txt").write_text("v1\n")
    initial = vcs.init_repo(repo)
    (repo / "a.txt").write_text("v2\n")
    (repo / "new.txt").write_text("brand new\n")
    diff = vcs.diff_since(repo, initial)
    assert "+v2" in diff
    assert "new.txt" in diff
    assert "brand new" in diff


def test_squash_to_one_commit_collapses_checkpoints(repo: Path) -> None:
    (repo / "a.txt").write_text("v1\n")
    initial = vcs.init_repo(repo)
    (repo / "a.txt").write_text("v2\n")
    vcs.commit(repo, "task 0 checkpoint")
    (repo / "b.txt").write_text("v1\n")
    vcs.commit(repo, "task 1 checkpoint")

    squashed = vcs.squash_to_one_commit(repo, since=initial, message="feat: add widget\n\nWhy: ...")

    assert squashed != initial
    parent = _run(repo, ["rev-parse", f"{squashed}^"])
    assert parent == initial
    assert (repo / "a.txt").read_text() == "v2\n"
    assert (repo / "b.txt").read_text() == "v1\n"
    message = _run(repo, ["log", "-1", "--pretty=%B", squashed])
    assert "feat: add widget" in message


def test_squash_to_one_commit_message_matches(repo: Path) -> None:
    (repo / "a.txt").write_text("v1\n")
    initial = vcs.init_repo(repo)
    (repo / "a.txt").write_text("v2\n")
    vcs.commit(repo, "task 0 checkpoint")
    squashed = vcs.squash_to_one_commit(repo, since=initial, message="chore: squash")
    subject = _run(repo, ["log", "-1", "--pretty=%s", squashed])
    assert subject == "chore: squash"


def test_inherit_origin_remote_copies_url(tmp_path: Path, repo: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    vcs.init_repo(upstream)
    _run(upstream, ["remote", "add", "origin", "https://example.invalid/x.git"])
    vcs.init_repo(repo)

    url = vcs.inherit_origin_remote(repo, upstream)

    assert url == "https://example.invalid/x.git"
    assert _run(repo, ["remote", "get-url", "origin"]) == "https://example.invalid/x.git"


def test_inherit_origin_remote_returns_none_without_an_origin(tmp_path: Path, repo: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    vcs.init_repo(upstream)
    vcs.init_repo(repo)

    assert vcs.inherit_origin_remote(repo, upstream) is None


def test_push_branch_succeeds_against_a_local_bare_remote(tmp_path: Path, repo: Path) -> None:
    bare = tmp_path / "origin.git"
    bare.mkdir()
    _run(bare, ["init", "-q", "--bare"])
    vcs.init_repo(repo)
    vcs.create_branch(repo, "xeno/add-widget-abc123")
    _run(repo, ["remote", "add", "origin", str(bare)])

    assert vcs.push_branch(repo, "xeno/add-widget-abc123") is True


def test_push_branch_returns_false_without_a_remote(repo: Path) -> None:
    vcs.init_repo(repo)
    vcs.create_branch(repo, "xeno/add-widget-abc123")
    assert vcs.push_branch(repo, "xeno/add-widget-abc123") is False


def test_open_pr_returns_none_when_gh_is_missing(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    vcs.init_repo(repo)
    monkeypatch.setattr(vcs.shutil, "which", lambda name: None)
    result = vcs.open_pr(repo, branch="xeno/x", title="t", body="b")
    assert result is None


def test_open_pr_returns_none_on_gh_failure(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    vcs.init_repo(repo)
    monkeypatch.setattr(vcs.shutil, "which", lambda name: "/usr/bin/gh")

    class _FailedResult:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(vcs.subprocess, "run", lambda *a, **k: _FailedResult())
    result = vcs.open_pr(repo, branch="xeno/x", title="t", body="b")
    assert result is None


def test_open_pr_returns_url_on_success(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    vcs.init_repo(repo)
    monkeypatch.setattr(vcs.shutil, "which", lambda name: "/usr/bin/gh")

    class _OkResult:
        returncode = 0
        stdout = "https://github.com/example/repo/pull/1\n"

    monkeypatch.setattr(vcs.subprocess, "run", lambda *a, **k: _OkResult())
    result = vcs.open_pr(repo, branch="xeno/x", title="t", body="b")
    assert result == "https://github.com/example/repo/pull/1"


def _current_branch(repo: Path) -> str:
    return _run(repo, ["rev-parse", "--abbrev-ref", "HEAD"])


def _run(repo: Path, args: list[str]) -> str:
    import subprocess

    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()
