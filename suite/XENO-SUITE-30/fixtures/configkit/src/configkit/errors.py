"""Exception hierarchy used across :mod:`configkit`.

Every error raised by this package derives from :class:`ConfigError`, so callers
can catch a single type when they do not care about the precise failure mode.
"""

from __future__ import annotations


class ConfigError(Exception):
    """Base class for every error raised by configkit."""


class ParseError(ConfigError):
    """Raised when a configuration document cannot be parsed.

    ``source`` is the origin of the document (usually a file path) and ``line``
    is the 1-based line number the parser was looking at, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        line: int | None = None,
    ) -> None:
        self.message = message
        self.source = source
        self.line = line
        super().__init__(self._render())

    def _render(self) -> str:
        location = self.source or "<config>"
        if self.line is not None:
            location = f"{location}:{self.line}"
        return f"{location}: {self.message}"


class ValidationError(ConfigError):
    """Raised when configuration data does not satisfy a :class:`~configkit.schema.Schema`.

    ``field`` holds the dotted path of the offending field, when known.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.message = message
        self.field = field
        super().__init__(message if field is None else f"{field}: {message}")
