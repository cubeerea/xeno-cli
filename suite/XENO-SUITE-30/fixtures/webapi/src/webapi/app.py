"""The application object.

:class:`App` ties a :class:`~webapi.routing.Router` to a middleware stack.  It
owns exactly two pieces of policy: how a handler is invoked, and what an
unroutable request looks like.  Both answers are deliberately minimal today --
handlers receive the matched path parameters and nothing else, and misses are
rendered as plain text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from webapi.middleware import Middleware, chain
from webapi.request import Request
from webapi.response import Response
from webapi.routing import Handler, Router


class App:
    """A routed application driven directly, in process, by :meth:`handle`."""

    def __init__(self) -> None:
        self._router = Router()
        self._middlewares: list[Middleware] = []

    @property
    def router(self) -> Router:
        """The router backing this application."""
        return self._router

    @property
    def middlewares(self) -> tuple[Middleware, ...]:
        """The installed middleware stack, outermost first."""
        return tuple(self._middlewares)

    def use(self, middleware: Middleware) -> None:
        """Append ``middleware`` to the stack.

        Middleware installed first is the outermost layer of the pipeline.
        """
        self._middlewares.append(middleware)

    def route(self, method: str, pattern: str) -> Callable[[Handler], Handler]:
        """Return a decorator registering a handler for ``method`` and ``pattern``."""

        def decorator(handler: Handler) -> Handler:
            self._router.add(method, pattern, handler)
            return handler

        return decorator

    def get(self, pattern: str) -> Callable[[Handler], Handler]:
        """Shorthand for ``route("GET", pattern)``."""
        return self.route("GET", pattern)

    def post(self, pattern: str) -> Callable[[Handler], Handler]:
        """Shorthand for ``route("POST", pattern)``."""
        return self.route("POST", pattern)

    def handle(self, request: Request) -> Response:
        """Run ``request`` through the middleware stack and the router."""
        pipeline = chain(self._middlewares, self._dispatch)
        return pipeline(request)

    def _dispatch(self, request: Request) -> Response:
        match = self._router.match(request.method, request.path)
        if match is not None:
            return match.handler(self._handler_request(request, match.params))
        allowed = self._router.allowed_methods(request.path)
        if allowed:
            return Response.text(
                405, f"405 Method Not Allowed: {request.method} {request.path}"
            ).with_header("Allow", ", ".join(sorted(allowed)))
        return Response.text(404, f"404 Not Found: {request.path}")

    @staticmethod
    def _handler_request(request: Request, params: Mapping[str, object]) -> Request:
        """Build the request a handler sees.

        The handler is given a request whose ``context`` carries the path
        parameters the router captured, under the ``"params"`` key.
        """
        return Request(
            method=request.method,
            path=request.path,
            headers=dict(request.headers),
            body=request.body,
            context={"params": dict(params)},
        )
