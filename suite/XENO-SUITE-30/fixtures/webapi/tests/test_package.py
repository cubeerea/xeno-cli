"""The package re-exports its public surface."""

from __future__ import annotations

import webapi


def test_public_names_are_importable_from_the_package() -> None:
    for name in webapi.__all__:
        assert hasattr(webapi, name), name


def test_core_objects_are_re_exported() -> None:
    assert webapi.App is not None
    assert webapi.Router is not None
    assert webapi.Request is not None
    assert webapi.Response is not None


def test_all_has_no_duplicates() -> None:
    assert len(set(webapi.__all__)) == len(webapi.__all__)


def test_version_is_exported() -> None:
    assert webapi.__version__ == "0.1.0"
