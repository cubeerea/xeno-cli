"""Filesystem-level exclusion for anything copied into a worktree or a
sandbox mount (PRD S11.3: "Never mounted into the sandbox. The denylist is
explicit and user-extensible.").

Distinct from `xeno.security.scanner`, which redacts secret-SHAPED text
already inside an assembled prompt. This module is the earlier, coarser
gate: entire files matching the secrets denylist are never copied into a
worktree in the first place, so they cannot be read by Daedalus, Chiron, a
codebase-map dump, or a sandboxed test run — redaction never gets a chance
to fail on something that was never there.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable

from xeno.core.config import SecretsConfig
from xeno.prompt.keys import DEFAULT_IGNORES

_IgnoreFn = Callable[[str, list[str]], set[str]]


def mount_ignore(secrets: SecretsConfig) -> _IgnoreFn:
    """A `shutil.copytree(ignore=...)` callable dropping both the standard
    worktree-copy exclusions (.git, caches, ...) and every path matching the
    secrets denylist."""
    patterns = (*DEFAULT_IGNORES, *secrets.effective_denylist)
    return shutil.ignore_patterns(*patterns)
