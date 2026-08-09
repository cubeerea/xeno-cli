"""ZS-21 acceptance spec: cross-origin support.

Adds ``cors_middleware`` (exported from the package), a ``Response.no_content``
helper, and native ``OPTIONS`` handling in the application so a preflight is
answered without registering an OPTIONS route.
"""

from __future__ import annotations

import webapi
from webapi import App, Request, Response, cors_middleware

ALLOWED = "https://allowed.example"
BLOCKED = "https://blocked.example"


def _build_app(origins: list[str] | None = None) -> App:
    app = App()

    @app.get("/widgets")
    def list_widgets(request: Request) -> Response:
        return Response.text(200, "widgets")

    @app.post("/widgets")
    def create_widget(request: Request) -> Response:
        return Response.text(201, "created")

    if origins is not None:
        app.use(cors_middleware(origins))
    return app


def test_no_content_helper_exists() -> None:
    response = Response.no_content()
    assert response.status == 204
    assert response.body == ""


def test_cors_middleware_is_exported_from_the_package() -> None:
    assert "cors_middleware" in webapi.__all__
    assert webapi.cors_middleware is cors_middleware


def test_allowed_origin_receives_the_allow_origin_header() -> None:
    app = _build_app([ALLOWED])
    response = app.handle(Request("GET", "/widgets", headers={"Origin": ALLOWED}))
    assert response.status == 200
    assert response.body == "widgets"
    assert response.header("Access-Control-Allow-Origin") == ALLOWED


def test_disallowed_origin_receives_no_allow_origin_header() -> None:
    app = _build_app([ALLOWED])
    response = app.handle(Request("GET", "/widgets", headers={"Origin": BLOCKED}))
    assert response.status == 200
    assert response.header("Access-Control-Allow-Origin") is None


def test_request_without_an_origin_is_untouched() -> None:
    app = _build_app([ALLOWED])
    response = app.handle(Request("GET", "/widgets"))
    assert response.status == 200
    assert response.header("Access-Control-Allow-Origin") is None


def test_preflight_is_answered_with_204_and_cors_headers() -> None:
    app = _build_app([ALLOWED])
    response = app.handle(
        Request(
            "OPTIONS",
            "/widgets",
            headers={
                "Origin": ALLOWED,
                "Access-Control-Request-Method": "POST",
            },
        )
    )
    assert response.status == 204
    assert response.body == ""
    assert response.header("Access-Control-Allow-Origin") == ALLOWED
    methods = response.header("Access-Control-Allow-Methods") or ""
    assert "GET" in methods
    assert "POST" in methods
    assert (response.header("Access-Control-Allow-Headers") or "") != ""


def test_app_answers_options_natively_without_any_middleware() -> None:
    app = _build_app()
    response = app.handle(Request("OPTIONS", "/widgets"))
    assert response.status == 204
    allow = response.header("Allow") or ""
    assert "GET" in allow
    assert "POST" in allow


def test_options_on_an_unknown_path_is_still_a_404() -> None:
    app = _build_app([ALLOWED])
    response = app.handle(Request("OPTIONS", "/nowhere", headers={"Origin": ALLOWED}))
    assert response.status == 404
