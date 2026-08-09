"""Baseline behaviour of the environment overlay."""

from __future__ import annotations

import pytest

from configkit.env import env_overrides, split_var_name


def test_env_overrides_nests_on_the_double_underscore_separator() -> None:
    environ = {"APP__DB__PORT": "5432", "APP__NAME": "svc", "OTHER__X": "ignored"}
    assert env_overrides("APP", environ) == {"db": {"port": "5432"}, "name": "svc"}


def test_env_overrides_keeps_every_value_as_a_string() -> None:
    environ = {"APP__DB__PORT": "5432", "APP__DEBUG": "true", "APP__RATIO": "0.5"}
    result = env_overrides("APP", environ)
    assert result["db"]["port"] == "5432"
    assert result["debug"] == "true"
    assert result["ratio"] == "0.5"


def test_env_overrides_ignores_the_bare_prefix_and_foreign_names() -> None:
    environ = {"APP": "x", "APPLE__PIE": "y", "APP__OK": "z"}
    assert env_overrides("APP", environ) == {"ok": "z"}


def test_env_overrides_reads_the_process_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CFGKIT__DB__HOST", "from-process")
    assert env_overrides("CFGKIT") == {"db": {"host": "from-process"}}


def test_env_overrides_rejects_an_empty_prefix() -> None:
    with pytest.raises(ValueError):
        env_overrides("", {})


def test_split_var_name_lowercases_components() -> None:
    assert split_var_name("APP__DB__POOL__SIZE", "APP") == ["db", "pool", "size"]
    assert split_var_name("NOPE__X", "APP") is None
