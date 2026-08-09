"""Pattern based routing.

Route patterns are ``/`` separated.  A segment written ``<name>`` is a
placeholder that matches exactly one segment and captures it as a string.
There is no converter syntax: every captured value is a ``str``, and matching
is performed against the request path exactly as it was supplied.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from webapi.request import Request
from webapi.response import Response

Handler = Callable[[Request], Response]

_PLACEHOLDER = re.compile(r"^<([A-Za-z_][A-Za-z0-9_]*)>$")


@dataclass(frozen=True)
class _Placeholder:
    """A single ``<name>`` segment of a compiled route pattern."""

    name: str


_Segment = str | _Placeholder


def split_path(path: str) -> tuple[str, ...]:
    """Split ``path`` into its non-empty segments.

    Leading, trailing and repeated separators are ignored, so ``/a/b`` and
    ``/a/b/`` describe the same resource.
    """
    return tuple(segment for segment in path.split("/") if segment)


def compile_pattern(pattern: str) -> tuple[_Segment, ...]:
    """Compile a route pattern into a tuple of literal and placeholder segments.

    Raises:
        ValueError: If the pattern is not absolute, contains a malformed
            placeholder, or repeats a placeholder name.
    """
    if not pattern.startswith("/"):
        raise ValueError(f"route pattern must be absolute: {pattern!r}")
    segments: list[_Segment] = []
    seen: set[str] = set()
    for raw in split_path(pattern):
        matched = _PLACEHOLDER.match(raw)
        if matched is None:
            if "<" in raw or ">" in raw:
                raise ValueError(f"malformed placeholder {raw!r} in pattern {pattern!r}")
            segments.append(raw)
            continue
        name = matched.group(1)
        if name in seen:
            raise ValueError(f"duplicate placeholder {name!r} in pattern {pattern!r}")
        seen.add(name)
        segments.append(_Placeholder(name))
    return tuple(segments)


@dataclass(frozen=True)
class Route:
    """A single registered route."""

    method: str
    pattern: str
    handler: Handler
    segments: tuple[_Segment, ...] = field(init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", self.method.upper())
        object.__setattr__(self, "segments", compile_pattern(self.pattern))

    def match_segments(self, segments: Sequence[str]) -> dict[str, str] | None:
        """Match already split path segments, returning captured parameters."""
        if len(segments) != len(self.segments):
            return None
        params: dict[str, str] = {}
        for expected, actual in zip(self.segments, segments, strict=True):
            if isinstance(expected, _Placeholder):
                params[expected.name] = actual
            elif expected != actual:
                return None
        return params


@dataclass(frozen=True)
class Match:
    """A successful resolution of a request onto a route."""

    route: Route
    params: dict[str, str]

    @property
    def handler(self) -> Handler:
        """The handler registered for the matched route."""
        return self.route.handler


class Router:
    """An ordered collection of routes.

    Routes are tried in registration order, so the first pattern that matches
    a path wins.
    """

    def __init__(self) -> None:
        self._routes: list[Route] = []

    @property
    def routes(self) -> tuple[Route, ...]:
        """All registered routes, in registration order."""
        return tuple(self._routes)

    def add(self, method: str, pattern: str, handler: Handler) -> Route:
        """Register ``handler`` for ``method`` requests matching ``pattern``."""
        route = Route(method=method, pattern=pattern, handler=handler)
        self._routes.append(route)
        return route

    def match(self, method: str, path: str) -> Match | None:
        """Resolve ``method`` and ``path`` onto a route.

        Returns ``None`` when nothing matches.  Callers distinguish "no such
        path" from "wrong method" by consulting :meth:`allowed_methods`.
        """
        wanted = method.upper()
        for route, params in self._candidates(path):
            if route.method == wanted:
                return Match(route=route, params=params)
        return None

    def allowed_methods(self, path: str) -> frozenset[str]:
        """Return every method registered for ``path``.

        An empty set means the path itself is unknown, which is what separates
        a 404 from a 405.
        """
        return frozenset(route.method for route, _ in self._candidates(path))

    def _candidates(self, path: str) -> Iterator[tuple[Route, dict[str, str]]]:
        segments = split_path(path)
        for route in self._routes:
            params = route.match_segments(segments)
            if params is not None:
                yield route, params
