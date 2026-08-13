"""Reaching the container engine, and the message when we cannot.

The bug these cover: a stopped Docker Desktop surfaced as a thirty-frame
urllib3 traceback ending in `FileNotFoundError: [Errno 2] No such file or
directory`, with no path and no mention of Docker anywhere in it. Run inside a
directory you are actively working in and that reads as "xeno cannot find my
files" — which is precisely how it was reported. The diagnosis is only half
the fix; the other half is that the check now runs before a run spends a token
on work no gate could ever evaluate.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import typer
from docker.errors import DockerException

from xeno import cli
from xeno.core.config import XenoConfig
from xeno.sandbox import engine


@pytest.fixture(autouse=True)
def _no_ambient_docker_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer with DOCKER_HOST set would otherwise see different
    endpoints than CI, and half of these assertions turn on the endpoint."""
    monkeypatch.delenv("DOCKER_HOST", raising=False)


def _refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise DockerException(
            "Error while fetching server API version: "
            "('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))"
        )

    monkeypatch.setattr(engine.docker, "from_env", boom)


# ---- which endpoint we actually tried --------------------------------------


def test_docker_host_wins_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://192.168.64.2:2375")
    assert engine.endpoint() == "tcp://192.168.64.2:2375"


def test_the_endpoint_falls_back_to_the_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason the default `/var/run/docker.sock` is the wrong thing to
    quote at a Mac user: Docker Desktop's context points somewhere else
    entirely, under ~/.docker."""
    monkeypatch.setattr(
        engine.ContextAPI, "get_current_context", lambda: type("C", (), {"Host": "unix:///ctx.sock"})
    )
    assert engine.endpoint() == "unix:///ctx.sock"


def test_an_unreadable_context_still_yields_an_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """`endpoint()` exists to make an error message concrete. An error message
    that raises while being built is strictly worse than a vague one."""

    def boom() -> None:
        raise RuntimeError("no context store")

    monkeypatch.setattr(engine.ContextAPI, "get_current_context", boom)
    assert engine.endpoint() == engine._DEFAULT_ENDPOINT


# ---- what the message says -------------------------------------------------


def test_a_missing_socket_is_named_as_such(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path}/absent.sock")
    _refuse(monkeypatch)

    with pytest.raises(engine.EngineUnavailable) as excinfo:
        engine.connect()

    message = str(excinfo.value)
    assert "container engine" in message, "the word the traceback never said"
    assert "no socket at" in message
    assert str(tmp_path / "absent.sock") in message, "the path it actually tried"


def test_a_dangling_symlink_is_called_out_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The case a user cannot see with `ls`: Docker Desktop leaves
    /var/run/docker.sock behind as a symlink to a socket under ~/.docker that
    only exists while its VM is up. The path looks present and resolves to
    nothing, so 'no such file' is true of a file that appears to be there."""
    link = tmp_path / "docker.sock"
    link.symlink_to(tmp_path / "gone.sock")
    monkeypatch.setenv("DOCKER_HOST", f"unix://{link}")
    _refuse(monkeypatch)

    with pytest.raises(engine.EngineUnavailable) as excinfo:
        engine.connect()

    message = str(excinfo.value)
    assert "symlink" in message
    assert str(tmp_path / "gone.sock") in message, "where the link points, not just the link"


def test_a_tcp_endpoint_reports_the_underlying_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no socket to stat on a remote engine, so the message falls
    back to what docker-py said rather than inventing a filesystem finding."""
    monkeypatch.setenv("DOCKER_HOST", "tcp://10.0.0.9:2375")
    _refuse(monkeypatch)

    with pytest.raises(engine.EngineUnavailable, match=re.escape("tcp://10.0.0.9:2375")):
        engine.connect()


def test_the_message_says_what_to_do(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A diagnosis with no remedy still leaves the user searching."""
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path}/absent.sock")
    _refuse(monkeypatch)

    with pytest.raises(engine.EngineUnavailable) as excinfo:
        engine.connect()

    message = str(excinfo.value)
    assert "Docker Desktop" in message
    assert "xeno doctor" in message


def test_the_message_says_why_it_is_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Not a nicety: 'no container engine' does not obviously mean 'this run
    cannot happen' unless you already know every gate runs in one."""
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path}/absent.sock")
    _refuse(monkeypatch)

    with pytest.raises(engine.EngineUnavailable, match="gate"):
        engine.connect()


def test_a_socket_that_exists_but_refuses_is_distinguished(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live socket that refuses is a different problem — a daemon starting
    up, or a permissions issue — and telling the user to start Docker when
    Docker is already running sends them the wrong way."""
    import socket
    import tempfile

    del tmp_path  # AF_UNIX paths cap at ~104 bytes; pytest's tmp_path exceeds it
    with tempfile.TemporaryDirectory(dir="/tmp") as short:
        sock_path = Path(short) / "live.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(sock_path))
        try:
            monkeypatch.setenv("DOCKER_HOST", f"unix://{sock_path}")
            _refuse(monkeypatch)
            with pytest.raises(engine.EngineUnavailable, match="refused the connection"):
                engine.connect()
        finally:
            server.close()


# ---- probe -----------------------------------------------------------------


def test_probe_reports_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_preflight` and `doctor` both want a verdict, not an exception."""
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path}/absent.sock")
    _refuse(monkeypatch)

    reachable, detail = engine.probe()
    assert reachable is False
    assert "no socket at" in detail


def test_probe_reports_the_engine_version_when_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Client:
        def __init__(self) -> None:
            self.closed = False

        def version(self) -> dict[str, str]:
            return {"Version": "27.1.1"}

        def close(self) -> None:
            self.closed = True

    client = _Client()
    monkeypatch.setattr(engine.docker, "from_env", lambda: client)

    reachable, detail = engine.probe()
    assert reachable is True
    assert "27.1.1" in detail
    assert client.closed, "probe holds no connection open — a run opens its own"


def test_probe_does_not_leak_a_client_when_the_daemon_half_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connected, then failed on the first real call. The client is live and
    must still be closed, or `doctor` leaks a socket per invocation."""

    class _Client:
        def __init__(self) -> None:
            self.closed = False

        def version(self) -> dict[str, str]:
            raise DockerException("daemon is shutting down")

        def close(self) -> None:
            self.closed = True

    client = _Client()
    monkeypatch.setattr(engine.docker, "from_env", lambda: client)

    reachable, detail = engine.probe()
    assert reachable is False
    assert "did not answer" in detail
    assert client.closed


# ---- the timing, which is the other half of the fix ------------------------


def test_preflight_refuses_before_the_run_spends_anything(
    monkeypatch: pytest.MonkeyPatch, config: XenoConfig
) -> None:
    """The original failure happened inside `ToolchainSession.start`, which is
    downstream of the worktree copy, the whole spec conversation, and the
    discovery call. Everything before it was spent on a run that could never
    have gated anything."""
    monkeypatch.setattr(cli, "engine_probe", lambda: (False, "cannot reach the container engine"))
    monkeypatch.setattr(
        cli, "_print_capability_warnings", lambda config: pytest.fail("reached too far")
    )

    with pytest.raises(typer.Exit) as excinfo:
        cli._preflight(config)

    assert excinfo.value.exit_code == 2


def test_preflight_passes_when_the_engine_is_up(
    monkeypatch: pytest.MonkeyPatch, config: XenoConfig
) -> None:
    monkeypatch.setattr(cli, "engine_probe", lambda: (True, "unix:///x.sock (engine 27.1.1)"))
    cli._preflight(config)


def test_the_symlink_reading_needs_no_daemon(tmp_path: Path) -> None:
    """`_socket_finding` is pure filesystem inspection, deliberately: it has to
    work in exactly the situation where nothing about Docker is answering."""
    missing = tmp_path / "absent.sock"
    assert "no socket at" in engine._socket_finding(str(missing))

    plain = tmp_path / "regular"
    plain.write_text("")
    assert "not a socket" in engine._socket_finding(str(plain))


def test_the_relative_paths_of_os_readlink_are_not_mangled(tmp_path: Path) -> None:
    """A relative symlink target is reported as written. Resolving it would be
    friendlier but would also silently invent a path the user never typed."""
    link = tmp_path / "docker.sock"
    link.symlink_to("../elsewhere.sock")
    finding = engine._socket_finding(str(link))
    assert "../elsewhere.sock" in finding
    assert os.readlink(link) == "../elsewhere.sock"
