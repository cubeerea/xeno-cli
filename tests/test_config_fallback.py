"""What a run says when it is not using the config you think it is.

`find_config` walks up from the cwd, so running xeno in a directory outside
any repo that carries an xeno.yaml swaps the entire routing table for the
built-in defaults — silently. The defaults route medium and light to Ollama,
which is the right thing to ship and the wrong thing to do without saying so:
the observable effect of running in an unconfigured directory was several GB
of model weights becoming resident on a machine that had not been asked to
hold them, and staying there long enough for the OS to give up.

Local weights are the one configuration choice the harness cannot meter,
throttle, or roll back. A remote model that is too big for the budget is a
number in cost.json. A local one that is too big for the machine is swap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xeno import cli
from xeno.core.config import (
    CONFIG_FILENAME,
    DEFAULT_NODE_TIERS,
    XenoConfig,
    default_config,
    find_config,
    load_config,
    user_config_path,
)
from xeno.core.types import Tier

_REMOTE_TIERS = """\
tiers:
  flagship:
    - {provider: openrouter, model: remote/a, usd_per_1m_input: 1.0, usd_per_1m_output: 1.0}
  medium:
    - {provider: openrouter, model: remote/b, usd_per_1m_input: 1.0, usd_per_1m_output: 1.0}
  light:
    - {provider: openrouter, model: remote/c, usd_per_1m_input: 1.0, usd_per_1m_output: 1.0}
"""


def _written(tmp_path: Path, tiers: str) -> XenoConfig:
    """A config that came from a file, so `source_path` is set the way a real
    run sets it — the whole point of these tests is that provenance is what
    the warning keys on."""
    nodes = "\n".join(f"  {r.value}: {{tier: {t.value}}}" for r, t in DEFAULT_NODE_TIERS.items())
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / CONFIG_FILENAME
    path.write_text(f"{tiers}nodes:\n{nodes}\n")
    return load_config(path)


@pytest.fixture(autouse=True)
def _isolated_user_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Point the user-level layer at an empty directory for every test here.

    Without this the suite reads whatever is in the developer's real home, so
    the same code passes on a machine with no `~/.config/xeno/xeno.yaml` and
    fails on one that has it — and the second machine is every machine where
    someone has taken this feature's advice.
    """
    home = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    return home


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    collected: list[str] = []
    monkeypatch.setattr(cli, "_warn", collected.append)
    return collected


# ---- the silent swap itself ------------------------------------------------


def test_an_unconfigured_directory_finds_nothing_to_load(tmp_path: Path) -> None:
    """Not a bug on its own — this is the documented behaviour, and it is what
    lets `xeno run` work in a fresh directory at all. `tmp_path` is under
    /private/var on macOS, so the upward walk terminates without a hit."""
    assert find_config(tmp_path) is None
    assert default_config().source_path is None, "the marker the warning keys on"


def test_the_fallback_routes_to_local_weights() -> None:
    """The fact that makes the silence expensive rather than merely untidy."""
    local = default_config().local_models()
    assert local, "if the defaults ever stop being local, the warning's urgency changes"
    assert {tier for tier, _ in local} == {Tier.MEDIUM, Tier.LIGHT}


# ---- what the user is told -------------------------------------------------


def test_a_defaulted_config_warns_that_it_defaulted(warnings: list[str]) -> None:
    cli._print_capability_warnings(default_config())
    assert any(CONFIG_FILENAME in w and "defaults are in effect" in w for w in warnings)


def test_the_warning_names_the_models_that_will_load(warnings: list[str]) -> None:
    """"Using defaults" does not tell you what is about to be resident in RAM,
    and the models are the only part of the message that is actionable."""
    cli._print_capability_warnings(default_config())
    text = " ".join(warnings)
    assert "qwen2.5-coder:14b" in text
    assert "qwen2.5-coder:7b" in text


def test_the_warning_says_how_it_fails(warnings: list[str]) -> None:
    """Swapping rather than failing is the specific thing to warn about: a run
    that dies is obvious, and a run that quietly takes the machine down with
    it is what actually happened."""
    cli._print_capability_warnings(default_config())
    assert any("swap" in w for w in warnings)


def test_the_warning_says_how_to_stop_it_permanently(warnings: list[str]) -> None:
    """Pointing at the tier config would fix this directory. `--user` fixes
    every directory, which is the shape the problem actually has: you do not
    hit this once, you hit it in each new place you try to work."""
    cli._print_capability_warnings(default_config())
    text = " ".join(warnings)
    assert "xeno init --user" in text
    assert CONFIG_FILENAME in text


def test_a_loaded_config_does_not_warn_about_provenance(
    tmp_path: Path, warnings: list[str]
) -> None:
    """The warning is about NOT KNOWING which config is in effect. Someone who
    wrote the file knows, and a warning they cannot act on is noise that
    teaches them to skim past the ones they can."""
    cli._print_capability_warnings(_written(tmp_path, _REMOTE_TIERS))
    assert not any("defaults are in effect" in w for w in warnings)


def test_a_defaulted_config_with_no_local_models_still_warns(warnings: list[str]) -> None:
    """Provenance is worth reporting even when nothing will load locally —
    the routing table is still not the one the user thinks it is — but the
    RAM paragraph would be a lie, so it is omitted rather than softened."""
    config = default_config()
    remote_only = config.model_copy(
        update={"tiers": {Tier.FLAGSHIP: config.tiers[Tier.FLAGSHIP]}}
    )
    assert not remote_only.local_models()

    cli._print_capability_warnings(remote_only)
    text = " ".join(w for w in warnings if "defaults are in effect" in w)
    assert text, "still warned"
    assert "swap" not in text


# ---- the user-level layer --------------------------------------------------


def _user_file(home: Path, tiers: str = _REMOTE_TIERS) -> Path:
    nodes = "\n".join(f"  {r.value}: {{tier: {t.value}}}" for r, t in DEFAULT_NODE_TIERS.items())
    path = home / "xeno" / CONFIG_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text(f"{tiers}nodes:\n{nodes}\n")
    return path


def test_the_user_config_is_found_from_an_unconfigured_directory(
    tmp_path: Path, _isolated_user_config: Path
) -> None:
    """The whole feature: your routing follows you out of your repo."""
    expected = _user_file(_isolated_user_config)
    assert find_config(tmp_path) == expected


def test_a_project_config_still_wins(tmp_path: Path, _isolated_user_config: Path) -> None:
    """A repository that ships a config has said something specific about how
    it wants to be built, and a personal default must not overrule it."""
    _user_file(_isolated_user_config)
    project = tmp_path / CONFIG_FILENAME
    project.write_text("tiers: {}\n")  # never parsed; find_config only stats
    assert find_config(tmp_path) == project


def test_a_parent_project_config_outranks_the_user_config(
    tmp_path: Path, _isolated_user_config: Path
) -> None:
    """Precedence is by specificity, not by proximity in the lookup code: a
    config three directories up is still about the project you are inside."""
    _user_file(_isolated_user_config)
    (tmp_path / CONFIG_FILENAME).write_text("tiers: {}\n")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert find_config(nested) == tmp_path / CONFIG_FILENAME


def test_xdg_config_home_is_honoured(_isolated_user_config: Path) -> None:
    assert user_config_path() == _isolated_user_config / "xeno" / CONFIG_FILENAME


def test_the_default_location_is_dot_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert user_config_path() == Path.home() / ".config" / "xeno" / CONFIG_FILENAME


def test_a_user_config_suppresses_the_defaults_warning(
    tmp_path: Path, _isolated_user_config: Path, warnings: list[str]
) -> None:
    """The point of setting one up. If it loaded but the run still warned
    about built-in defaults, the advice the warning gives would be wrong."""
    _user_file(_isolated_user_config)
    cli._print_capability_warnings(load_config(find_config(tmp_path)))
    assert not any("defaults are in effect" in w for w in warnings)


def test_a_user_config_with_local_weights_is_not_silently_blessed(
    tmp_path: Path, _isolated_user_config: Path, warnings: list[str]
) -> None:
    """Provenance suppresses the PROVENANCE warning, not the RAM one. Someone
    who wrote a local tier chose it; someone who wrote it and then ran on a
    smaller machine still has a 14B loading into it."""
    _user_file(
        _isolated_user_config,
        _REMOTE_TIERS.replace(
            "    - {provider: openrouter, model: remote/b, "
            "usd_per_1m_input: 1.0, usd_per_1m_output: 1.0}",
            "    - {provider: ollama, model: qwen2.5-coder:14b}",
        ),
    )
    config = load_config(find_config(tmp_path))
    assert config.local_models() == [(Tier.MEDIUM, "qwen2.5-coder:14b")]
    del warnings  # nothing asserted about output; the model list is the contract


def test_an_explicit_config_flag_beats_both(
    tmp_path: Path, _isolated_user_config: Path
) -> None:
    """`--config` is someone naming a file. Nothing should outrank that."""
    _user_file(_isolated_user_config)
    (tmp_path / CONFIG_FILENAME).write_text("tiers: {}\n")
    explicit = _written(tmp_path / "elsewhere", _REMOTE_TIERS)
    assert explicit.source_path == tmp_path / "elsewhere" / CONFIG_FILENAME


# ---- local_models ----------------------------------------------------------


def test_local_models_reports_every_tier_not_just_flagship() -> None:
    """`flagship_is_local` answers a quality question and deliberately looks
    at one tier. The RAM question is about every tier, because medium is where
    the 14B lives and medium is what Daedalus calls on every single task."""
    local = dict(default_config().local_models())
    assert Tier.MEDIUM in local


def test_local_models_sees_past_the_head_of_a_chain(tmp_path: Path) -> None:
    """A local escalation entry loads weights the moment the chain walks to
    it, which is precisely the moment the run is already going badly."""
    config = _written(
        tmp_path,
        _REMOTE_TIERS.replace(
            "  light:",
            "    - {provider: ollama, model: local/fallback, escalation: true}\n  light:",
        ),
    )
    # medium's chain is now [remote/b, ollama local/fallback] — the local
    # entry is reachable but is not what `tiers[medium][0]` reports.
    assert ("medium", "local/fallback") in [
        (tier.value, model) for tier, model in config.local_models()
    ]
