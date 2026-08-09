"""Inbound request objects.

A :class:`Request` is a frozen value object: middleware never mutates one in
place, it derives a new request with :meth:`Request.with_context`.  The
``path`` attribute is stored verbatim exactly as the caller supplied it; this
module performs no query-string parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Request:
    """An HTTP-shaped request handed to middleware and route handlers.

    Attributes:
        method: The request method.  Normalised to upper case.
        path: The request target, stored verbatim.
        headers: Request headers.  Names are normalised to lower case.
        body: The raw request body as text.
        context: Free-form per-request state.  The application populates it
            with the path parameters extracted by the router; middleware may
            add its own entries via :meth:`with_context`.
    """

    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""
    context: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", self.method.upper())
        object.__setattr__(
            self,
            "headers",
            {name.lower(): value for name, value in self.headers.items()},
        )

    def header(self, name: str, default: str | None = None) -> str | None:
        """Return the header called ``name``, case-insensitively."""
        return self.headers.get(name.lower(), default)

    def with_context(self, **values: object) -> Request:
        """Return a copy of this request with extra ``context`` entries."""
        merged = dict(self.context)
        merged.update(values)
        return replace(self, context=merged)

    def with_body(self, body: str) -> Request:
        """Return a copy of this request carrying a different body."""
        return replace(self, body=body)
