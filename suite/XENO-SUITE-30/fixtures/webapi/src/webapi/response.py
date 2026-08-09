"""Outbound response objects.

Only two constructors ship today: :meth:`Response.text` and
:meth:`Response.json`.  Error payloads are therefore whatever the caller
decides to build, and the application renders its own 404/405 answers as plain
text.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field, replace

TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"
JSON_CONTENT_TYPE = "application/json"


@dataclass
class Response:
    """An HTTP-shaped response produced by a handler or by the application.

    Attributes:
        status: The numeric status code.
        headers: Response headers, stored with the casing they were given.
        body: The rendered response body as text.
    """

    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""

    @classmethod
    def text(cls, status: int, body: str) -> Response:
        """Build a ``text/plain`` response."""
        return cls(status=status, headers={"Content-Type": TEXT_CONTENT_TYPE}, body=body)

    @classmethod
    def json(cls, status: int, payload: object) -> Response:
        """Build an ``application/json`` response from a JSON-encodable value."""
        body = _json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return cls(status=status, headers={"Content-Type": JSON_CONTENT_TYPE}, body=body)

    def header(self, name: str) -> str | None:
        """Return the response header called ``name``, case-insensitively."""
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None

    def with_header(self, name: str, value: str) -> Response:
        """Return a copy of this response carrying an extra header."""
        lowered = name.lower()
        headers = {key: old for key, old in self.headers.items() if key.lower() != lowered}
        headers[name] = value
        return replace(self, headers=headers)
