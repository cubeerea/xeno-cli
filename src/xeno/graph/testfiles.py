"""Shared test-file detection.

Three rules in the harness are enforced against this one definition, and they
point in opposite directions, which is exactly why it lives in one place:

* Lachesis may write ONLY test files (`xeno.graph.lachesis`) — it owns the
  specification.
* Chiron may NEVER write a test file (PRD S10's hard rule: "MUST NOT modify
  test files to make tests pass") — a debugger that can edit the check can
  always make the check pass.
* Daedalus may never write one either, for the same reason at one remove.

Talos also flags `EvalReport.touched_test_files` from it, so Cerberus sees
any diff that reached a test by some other route. Two of these drifting apart
would leave a gap between the files one node is confined to and the files
another is barred from.

Matching is PATH-STRUCTURAL rather than substring-based. The substring
version could not see `foo_test.go` at a repository root, and read the `src/
contest/` directory as a test tree — one under-match and one over-match, both
of which now decide whether a write is allowed rather than merely how a diff
is labelled.
"""

from __future__ import annotations

from pathlib import PurePosixPath

#: A directory anywhere in the path whose name is one of these makes
#: everything under it a test file. Spans conventions rather than one
#: ecosystem's (PRD S12 revised: gates are no longer Python-only).
TEST_DIR_NAMES = frozenset({"test", "tests", "__tests__", "spec", "specs", "testing"})

#: Whole filenames that are test infrastructure regardless of where they sit.
TEST_FILENAMES = frozenset({"conftest.py"})

#: `test_foo.py`, `test-foo.js`.
_TEST_PREFIXES = ("test_", "test-")

#: `foo_test.go`, `foo-spec.js` — matched against the stem, so the extension
#: does not have to be enumerated per language.
_TEST_STEM_SUFFIXES = ("_test", "-test", "_spec", "-spec")

#: `foo.test.ts`, `foo.spec.tsx` — an infix component of a dotted filename.
_TEST_INFIXES = (".test.", ".spec.")


def is_test_file(rel_path: str) -> bool:
    """Whether this repository-relative path is a test file."""
    path = PurePosixPath(rel_path)
    if any(part in TEST_DIR_NAMES for part in path.parts[:-1]):
        return True

    name = path.name
    if name in TEST_FILENAMES:
        return True
    if name.startswith(_TEST_PREFIXES):
        return True
    if any(infix in name for infix in _TEST_INFIXES):
        return True
    return PurePosixPath(name).stem.endswith(_TEST_STEM_SUFFIXES)
