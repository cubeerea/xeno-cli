"""configkit: a small layered configuration loader.

Configuration is assembled from ordered layers -- files first, then process
environment variables -- and finally checked against a :class:`Schema`.
"""

from __future__ import annotations

from configkit.env import env_overrides
from configkit.errors import ConfigError, ParseError, ValidationError
from configkit.loader import load_file, load_layers, parse_ini, parse_json
from configkit.merge import deep_merge, get_path, has_path, merge_all, set_path
from configkit.schema import Field, Schema

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "Field",
    "ParseError",
    "Schema",
    "ValidationError",
    "__version__",
    "deep_merge",
    "env_overrides",
    "get_path",
    "has_path",
    "load_file",
    "load_layers",
    "merge_all",
    "parse_ini",
    "parse_json",
    "set_path",
]
