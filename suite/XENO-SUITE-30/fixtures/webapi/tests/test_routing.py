"""Baseline behaviour of pattern compilation and the router."""

from __future__ import annotations

import pytest

from webapi.request import Request
from webapi.response import Response
from webapi.routing import Router, split_path


def _ok(request: Request) -> Response:
    return Response.text(200, "ok")


def _created(request: Request) -> Response:
    return Response.text(201, "created")


def test_split_path_drops_empty_segments() -> None:
    assert split_path("/users/42/") == ("users", "42")
    assert split_path("/") == ()


def test_static_route_matches_exactly() -> None:
    router = Router()
    router.add("GET", "/health", _ok)
    match = router.match("GET", "/health")
    assert match is not None
    assert match.params == {}
    assert match.handler is _ok
    assert router.match("GET", "/health/extra") is None


def test_placeholders_capture_strings() -> None:
    router = Router()
    router.add("GET", "/users/<user_id>/posts/<post_id>", _ok)
    match = router.match("GET", "/users/42/posts/abc")
    assert match is not None
    assert match.params == {"user_id": "42", "post_id": "abc"}


def test_method_is_normalised_on_registration_and_lookup() -> None:
    router = Router()
    route = router.add("post", "/users", _created)
    assert route.method == "POST"
    assert router.match("post", "/users") is not None


def test_unknown_method_does_not_match_but_is_reported_as_allowed() -> None:
    router = Router()
    router.add("GET", "/users/<user_id>", _ok)
    router.add("DELETE", "/users/<user_id>", _ok)
    assert router.match("PATCH", "/users/1") is None
    assert router.allowed_methods("/users/1") == frozenset({"GET", "DELETE"})


def test_allowed_methods_is_empty_for_an_unknown_path() -> None:
    router = Router()
    router.add("GET", "/users", _ok)
    assert router.allowed_methods("/widgets") == frozenset()


def test_first_registered_route_wins() -> None:
    router = Router()
    router.add("GET", "/items/<item_id>", _ok)
    router.add("GET", "/items/<slug>", _created)
    match = router.match("GET", "/items/7")
    assert match is not None
    assert match.handler is _ok
    assert match.params == {"item_id": "7"}


def test_routes_are_exposed_in_registration_order() -> None:
    router = Router()
    router.add("GET", "/a", _ok)
    router.add("POST", "/b", _created)
    assert [(route.method, route.pattern) for route in router.routes] == [
        ("GET", "/a"),
        ("POST", "/b"),
    ]


def test_relative_pattern_is_rejected() -> None:
    router = Router()
    with pytest.raises(ValueError, match="absolute"):
        router.add("GET", "users/<user_id>", _ok)


def test_duplicate_placeholder_is_rejected() -> None:
    router = Router()
    with pytest.raises(ValueError, match="duplicate placeholder"):
        router.add("GET", "/users/<user_id>/pets/<user_id>", _ok)
