"""Baseline end-to-end behaviour of the App object."""

from __future__ import annotations

from typing import cast

from webapi.app import App
from webapi.middleware import request_id_middleware
from webapi.request import Request
from webapi.response import Response
from webapi.routing import Handler


def _build_app(seen: list[Request] | None = None) -> App:
    app = App()

    @app.get("/health")
    def health(request: Request) -> Response:
        if seen is not None:
            seen.append(request)
        return Response.text(200, "ok")

    @app.get("/users/<user_id>")
    def show_user(request: Request) -> Response:
        if seen is not None:
            seen.append(request)
        params = cast(dict[str, object], request.context["params"])
        return Response.json(200, {"user_id": params["user_id"]})

    @app.post("/users")
    def create_user(request: Request) -> Response:
        return Response.text(201, f"created:{request.body}")

    return app


def test_route_decorator_returns_the_handler_unchanged() -> None:
    app = App()

    def handler(request: Request) -> Response:
        return Response.text(200, "ok")

    decorated: Handler = app.route("GET", "/thing")(handler)
    assert decorated is handler
    assert [route.pattern for route in app.router.routes] == ["/thing"]


def test_static_route_is_dispatched() -> None:
    response = _build_app().handle(Request("GET", "/health"))
    assert response.status == 200
    assert response.body == "ok"


def test_path_parameters_reach_the_handler_context() -> None:
    seen: list[Request] = []
    response = _build_app(seen).handle(Request("GET", "/users/42"))
    assert response.status == 200
    assert response.body == '{"user_id":"42"}'
    assert seen[-1].context["params"] == {"user_id": "42"}


def test_body_is_available_to_the_handler() -> None:
    response = _build_app().handle(Request("POST", "/users", body="alice"))
    assert response.status == 201
    assert response.body == "created:alice"


def test_unknown_path_is_a_404() -> None:
    response = _build_app().handle(Request("GET", "/nowhere"))
    assert response.status == 404
    assert response.header("Allow") is None


def test_known_path_with_wrong_method_is_a_405_with_allow() -> None:
    response = _build_app().handle(Request("DELETE", "/users/42"))
    assert response.status == 405
    assert response.header("Allow") == "GET"


def test_405_allow_header_lists_every_registered_method() -> None:
    app = App()
    app.route("GET", "/things")(lambda request: Response.text(200, "ok"))
    app.route("POST", "/things")(lambda request: Response.text(201, "ok"))
    response = app.handle(Request("PATCH", "/things"))
    assert response.status == 405
    assert response.header("Allow") == "GET, POST"


def test_middleware_wraps_dispatch_including_misses() -> None:
    app = _build_app()
    app.use(request_id_middleware(lambda: "fixed-id"))
    hit = app.handle(Request("GET", "/health"))
    miss = app.handle(Request("GET", "/nowhere"))
    assert hit.header("X-Request-ID") == "fixed-id"
    assert miss.header("X-Request-ID") == "fixed-id"
    assert miss.status == 404


def test_middleware_registration_order_is_preserved() -> None:
    app = App()
    first = request_id_middleware(lambda: "a", header="X-First")
    second = request_id_middleware(lambda: "b", header="X-Second")
    app.use(first)
    app.use(second)
    assert app.middlewares == (first, second)
