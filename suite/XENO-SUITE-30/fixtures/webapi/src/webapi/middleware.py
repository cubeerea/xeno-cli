"""Middleware composition.

A middleware is any callable taking ``(request, next_handler)`` and returning a
:class:`~webapi.response.Response`.  Middleware may short-circuit by never
calling ``next_handler``, may derive a new request before delegating, and may
post-process whatever the inner handler returned.

Only transport-neutral helpers live here; there is no cross-origin support.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from webapi.request import Request
from webapi.response import Response
from webapi.routing import Handler

Middleware = Callable[[Request, Handler], Response]

REQUEST_ID_HEADER = "X-Request-ID"


def chain(middlewares: Sequence[Middleware], handler: Handler) -> Handler:
    """Compose ``middlewares`` around ``handler``.

    The first middleware of the sequence is the outermost one, so it sees the
    request first and the response last.
    """
    composed = handler
    for middleware in reversed(tuple(middlewares)):
        composed = _bind(middleware, composed)
    return composed


def _bind(middleware: Middleware, next_handler: Handler) -> Handler:
    def invoke(request: Request) -> Response:
        return middleware(request, next_handler)

    return invoke


def request_id_middleware(
    next_id: Callable[[], str], header: str = REQUEST_ID_HEADER
) -> Middleware:
    """Build a middleware stamping every response with a request id.

    An id supplied by the caller is echoed back untouched; otherwise ``next_id``
    is consulted.  Injecting the generator keeps the middleware deterministic
    under test.
    """

    def middleware(request: Request, next_handler: Handler) -> Response:
        incoming = request.header(header)
        value = incoming if incoming is not None else next_id()
        return next_handler(request).with_header(header, value)

    return middleware


def logging_middleware(sink: Callable[[str], None]) -> Middleware:
    """Build a middleware writing one ``METHOD path -> status`` line per request."""

    def middleware(request: Request, next_handler: Handler) -> Response:
        response = next_handler(request)
        sink(f"{request.method} {request.path} -> {response.status}")
        return response

    return middleware
