"""Baseline behaviour of middleware composition and the built-in middleware."""

from __future__ import annotations

from collections.abc import Iterator

from webapi.middleware import Middleware, chain, logging_middleware, request_id_middleware
from webapi.request import Request
from webapi.response import Response
from webapi.routing import Handler


def _handler(request: Request) -> Response:
    return Response.text(200, f"body:{request.body}")


def _counter(values: list[str]) -> Iterator[str]:
    return iter(values)


def _tag(name: str, order: list[str]) -> Middleware:
    def middleware(request: Request, next_handler: Handler) -> Response:
        order.append(f"enter:{name}")
        response = next_handler(request)
        order.append(f"exit:{name}")
        return response

    return middleware


def test_chain_without_middleware_returns_an_equivalent_handler() -> None:
    composed = chain([], _handler)
    assert composed(Request("GET", "/", body="x")).body == "body:x"


def test_first_middleware_is_outermost() -> None:
    order: list[str] = []
    composed = chain([_tag("a", order), _tag("b", order)], _handler)
    assert composed(Request("GET", "/")).status == 200
    assert order == ["enter:a", "enter:b", "exit:b", "exit:a"]


def test_middleware_can_short_circuit() -> None:
    reached: list[str] = []

    def deny(request: Request, next_handler: Handler) -> Response:
        return Response.text(403, "denied")

    def inner(request: Request) -> Response:
        reached.append("inner")
        return Response.text(200, "ok")

    response = chain([deny], inner)(Request("GET", "/"))
    assert response.status == 403
    assert reached == []


def test_middleware_can_rewrite_the_request_before_delegating() -> None:
    def rewrite(request: Request, next_handler: Handler) -> Response:
        return next_handler(request.with_body("rewritten"))

    assert chain([rewrite], _handler)(Request("GET", "/", body="orig")).body == "body:rewritten"


def test_request_id_middleware_generates_an_id_when_absent() -> None:
    ids = _counter(["id-1", "id-2"])
    composed = chain([request_id_middleware(lambda: next(ids))], _handler)
    assert composed(Request("GET", "/")).header("X-Request-ID") == "id-1"
    assert composed(Request("GET", "/")).header("X-Request-ID") == "id-2"


def test_request_id_middleware_echoes_an_incoming_id() -> None:
    ids = _counter(["generated"])
    composed = chain([request_id_middleware(lambda: next(ids))], _handler)
    request = Request("GET", "/", headers={"x-request-id": "caller-supplied"})
    assert composed(request).header("X-Request-ID") == "caller-supplied"


def test_logging_middleware_records_one_line_per_request() -> None:
    lines: list[str] = []
    composed = chain([logging_middleware(lines.append)], _handler)
    composed(Request("GET", "/health"))
    composed(Request("post", "/users"))
    assert lines == ["GET /health -> 200", "POST /users -> 200"]
