"""webapi: a tiny in-process request-routing micro-framework.

The package is transport free.  Build a :class:`~webapi.request.Request`,
hand it to :meth:`~webapi.app.App.handle`, and inspect the returned
:class:`~webapi.response.Response`.  There are no sockets and no server.
"""

from webapi.app import App
from webapi.limits import RateLimiter, RateLimitPolicy
from webapi.middleware import Middleware, chain, logging_middleware, request_id_middleware
from webapi.request import Request
from webapi.response import Response
from webapi.routing import Handler, Match, Route, Router

__version__ = "0.1.0"

__all__ = [
    "App",
    "Handler",
    "Match",
    "Middleware",
    "RateLimitPolicy",
    "RateLimiter",
    "Request",
    "Response",
    "Route",
    "Router",
    "__version__",
    "chain",
    "logging_middleware",
    "request_id_middleware",
]
