"""ZS-13 acceptance spec: structured JSON error responses.

``Response.error`` builds an ``application/json`` envelope, and the application
uses it for its own 404 and 405 answers instead of plain text.
"""

from __future__ import annotations

import json

from webapi.app import App
from webapi.request import Request
from webapi.response import Response


def _build_app() -> App:
    app = App()

    @app.get("/users/<user_id>")
    def show_user(request: Request) -> Response:
        return Response.text(200, "ok")

    @app.post("/users")
    def create_user(request: Request) -> Response:
        return Response.text(201, "created")

    return app


def test_error_builds_a_json_envelope() -> None:
    response = Response.error(404, "no such thing")
    assert response.status == 404
    assert response.header("Content-Type") == "application/json"
    assert json.loads(response.body) == {
        "error": {"status": 404, "message": "no such thing", "code": None}
    }


def test_error_accepts_a_machine_readable_code() -> None:
    response = Response.error(429, "slow down", code="rate_limited")
    assert response.status == 429
    assert json.loads(response.body) == {
        "error": {"status": 429, "message": "slow down", "code": "rate_limited"}
    }


def test_app_renders_404_as_json() -> None:
    response = _build_app().handle(Request("GET", "/nowhere"))
    assert response.status == 404
    assert response.header("Content-Type") == "application/json"
    payload = json.loads(response.body)
    assert payload["error"]["status"] == 404
    assert isinstance(payload["error"]["message"], str)
    assert payload["error"]["message"] != ""


def test_app_renders_405_as_json_and_keeps_the_allow_header() -> None:
    response = _build_app().handle(Request("DELETE", "/users/42"))
    assert response.status == 405
    assert response.header("Content-Type") == "application/json"
    assert response.header("Allow") == "GET"
    payload = json.loads(response.body)
    assert payload["error"]["status"] == 405
    assert isinstance(payload["error"]["message"], str)
    assert payload["error"]["message"] != ""


def test_successful_responses_are_untouched() -> None:
    app = _build_app()
    ok = app.handle(Request("GET", "/users/42"))
    created = app.handle(Request("POST", "/users"))
    assert (ok.status, ok.body) == (200, "ok")
    assert (created.status, created.body) == (201, "created")
    assert Response.error(400, "bad").status == 400
