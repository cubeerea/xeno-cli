"""ZS-14 acceptance spec: query-string parsing.

``Request`` learns to split its target into ``path_only`` plus a parsed
``query`` mapping, the router matches on the path portion alone, and the
application exposes the parsed query to handlers under ``context["query"]``.
"""

from __future__ import annotations

from typing import cast

from webapi.app import App
from webapi.request import Request
from webapi.response import Response
from webapi.routing import Router


def _ok(request: Request) -> Response:
    return Response.text(200, "ok")


def test_request_splits_path_and_query() -> None:
    request = Request("GET", "/search?q=cat&q=dog&page=2")
    assert request.path == "/search?q=cat&q=dog&page=2"
    assert request.path_only == "/search"
    assert request.query == {"q": ["cat", "dog"], "page": ["2"]}


def test_request_without_a_query_string() -> None:
    request = Request("GET", "/search")
    assert request.path_only == "/search"
    assert request.query == {}


def test_request_with_an_empty_query_string() -> None:
    request = Request("GET", "/search?")
    assert request.path_only == "/search"
    assert request.query == {}


def test_query_survives_with_context() -> None:
    request = Request("GET", "/search?q=cat").with_context(trace="t1")
    assert request.query == {"q": ["cat"]}
    assert request.path_only == "/search"
    assert request.context == {"trace": "t1"}


def test_router_matches_on_the_path_portion_only() -> None:
    router = Router()
    router.add("GET", "/users/<user_id>", _ok)
    match = router.match("GET", "/users/42?verbose=1")
    assert match is not None
    assert match.params == {"user_id": "42"}


def test_router_allowed_methods_ignores_the_query_string() -> None:
    router = Router()
    router.add("GET", "/users/me", _ok)
    router.add("DELETE", "/users/me", _ok)
    assert router.allowed_methods("/users/me?verbose=1") == frozenset({"GET", "DELETE"})
    assert router.match("PATCH", "/users/me?verbose=1") is None
    assert router.match("GET", "/users/me?verbose=1") is not None


def test_app_routes_a_path_carrying_a_query_and_exposes_it() -> None:
    app = App()
    seen: list[Request] = []

    @app.get("/users/<user_id>")
    def show_user(request: Request) -> Response:
        seen.append(request)
        return Response.text(200, "ok")

    response = app.handle(Request("GET", "/users/7?fields=name&fields=email"))
    assert response.status == 200
    handler_request = seen[-1]
    assert cast(dict[str, object], handler_request.context["params"]) == {"user_id": "7"}
    assert handler_request.context["query"] == {"fields": ["name", "email"]}


def test_app_still_404s_on_an_unknown_path_with_a_query() -> None:
    app = App()
    app.route("GET", "/users/<user_id>")(_ok)
    request = Request("GET", "/nowhere?x=1")
    assert request.path_only == "/nowhere"
    assert app.handle(request).status == 404
