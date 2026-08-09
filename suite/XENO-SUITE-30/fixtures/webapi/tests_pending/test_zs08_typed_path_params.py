"""ZS-08 acceptance spec: typed path converters in route patterns.

Patterns may use ``<int:name>`` and ``<str:name>``.  An ``<int:...>`` segment
matches digits only and yields a real ``int``; ``<str:...>`` and the bare
``<name>`` form keep yielding strings.
"""

from __future__ import annotations

import pytest

from webapi.request import Request
from webapi.response import Response
from webapi.routing import Router


def _ok(request: Request) -> Response:
    return Response.text(200, "ok")


def _other(request: Request) -> Response:
    return Response.text(200, "other")


def test_int_converter_yields_a_real_int() -> None:
    router = Router()
    router.add("GET", "/users/<int:user_id>", _ok)
    match = router.match("GET", "/users/42")
    assert match is not None
    assert match.params == {"user_id": 42}
    assert isinstance(match.params["user_id"], int)
    assert not isinstance(match.params["user_id"], str)


def test_int_converter_rejects_a_non_numeric_segment() -> None:
    router = Router()
    router.add("GET", "/users/<int:user_id>", _ok)
    assert router.match("GET", "/users/abc") is None
    assert router.allowed_methods("/users/abc") == frozenset()


def test_int_converter_still_matches_a_numeric_segment_for_allowed_methods() -> None:
    router = Router()
    router.add("GET", "/users/<int:user_id>", _ok)
    router.add("DELETE", "/users/<int:user_id>", _other)
    assert router.allowed_methods("/users/7") == frozenset({"GET", "DELETE"})


def test_str_converter_yields_a_string() -> None:
    router = Router()
    router.add("GET", "/tags/<str:name>", _ok)
    match = router.match("GET", "/tags/123")
    assert match is not None
    assert match.params == {"name": "123"}
    assert isinstance(match.params["name"], str)


def test_bare_placeholder_still_yields_a_string() -> None:
    router = Router()
    router.add("GET", "/legacy/<name>", _ok)
    router.add("GET", "/typed/<int:number>", _other)
    match = router.match("GET", "/legacy/42")
    assert match is not None
    assert match.params == {"name": "42"}
    assert isinstance(match.params["name"], str)


def test_typed_and_untyped_routes_coexist_in_registration_order() -> None:
    router = Router()
    router.add("GET", "/items/<int:item_id>", _ok)
    router.add("GET", "/items/<str:slug>", _other)
    numeric = router.match("GET", "/items/7")
    textual = router.match("GET", "/items/blue")
    assert numeric is not None
    assert numeric.handler is _ok
    assert numeric.params == {"item_id": 7}
    assert textual is not None
    assert textual.handler is _other
    assert textual.params == {"slug": "blue"}


def test_unknown_converter_is_rejected() -> None:
    router = Router()
    router.add("GET", "/users/<int:user_id>", _ok)
    with pytest.raises(ValueError):
        router.add("GET", "/tokens/<uuid:token_id>", _other)
