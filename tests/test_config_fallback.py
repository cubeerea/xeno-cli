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
    path = tmp_path / CONFIG_FILENAME
    path.write_text(f"{tiers}nodes:\n{nodes}\n")
    return load_config(path)


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


def test_the_warning_says_how_to_stop_it(warnings: list[str]) -> None:
    cli._print_capability_warnings(default_config())
    text = " ".join(warnings)
    assert "API provider" in text
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
