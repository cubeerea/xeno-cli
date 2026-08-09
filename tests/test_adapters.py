"""PythonAdapter (PRD S12): argv shape and log parsing. No subprocess or
Docker here — that's covered live against the real sandbox; this module is
about the pure parsing/argv-building logic being correct on its own."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from xeno.adapters.python import PythonAdapter

adapter = PythonAdapter()


def test_argv_never_uses_the_host_interpreter_path() -> None:
    """Regression: every *_cmd must be resolvable INSIDE the sandbox
    container, where the harness's own sys.executable does not exist."""
    import sys

    for cmd in (
        adapter.parse_cmd(["a.py"]),
        adapter.lint_cmd(["a.py"]),
        adapter.typecheck_cmd(["a.py"]),
        adapter.test_cmd(),
        adapter.coverage_cmd(),
    ):
        assert cmd is not None
        assert cmd[0] != sys.executable
        assert cmd[0] == "python3"


def test_parse_cmd_is_none_with_no_paths() -> None:
    assert adapter.parse_cmd([]) is None


def test_parse_lint_counts_ruff_json_findings() -> None:
    findings = '[{"code": "F401"}, {"code": "E501"}]'
    assert adapter.parse_lint(findings) == 2


def test_parse_lint_empty_output_is_zero_findings() -> None:
    assert adapter.parse_lint("") == 0


def test_parse_lint_raises_on_unparseable_output() -> None:
    with pytest.raises(ValueError):
        adapter.parse_lint("ruff: command not found")


def test_lint_signature_distinguishes_different_rules_at_same_count() -> None:
    """Regression: CB-4 was blind to lint-only failures because the old
    signature had no lint-derived input at all — two DIFFERENT single-finding
    lint failures must not collide."""
    f401 = '[{"code": "F401", "filename": "a.py", "location": {"row": 1, "column": 1}}]'
    e501 = '[{"code": "E501", "filename": "a.py", "location": {"row": 1, "column": 1}}]'
    assert adapter.lint_signature(f401) != adapter.lint_signature(e501)


def test_lint_signature_stable_under_reordering() -> None:
    a = (
        '[{"code": "F401", "filename": "a.py", "location": {"row": 1, "column": 1}},'
        '{"code": "E501", "filename": "b.py", "location": {"row": 2, "column": 1}}]'
    )
    b = (
        '[{"code": "E501", "filename": "b.py", "location": {"row": 2, "column": 1}},'
        '{"code": "F401", "filename": "a.py", "location": {"row": 1, "column": 1}}]'
    )
    assert adapter.lint_signature(a) == adapter.lint_signature(b)


def test_lint_signature_ignores_column_shift_same_row() -> None:
    """A reflow that nudges the same violation a few columns right is still
    the same defect — only row identity should matter, not column."""
    a = '[{"code": "E501", "filename": "a.py", "location": {"row": 38, "column": 101}}]'
    b = '[{"code": "E501", "filename": "a.py", "location": {"row": 38, "column": 130}}]'
    assert adapter.lint_signature(a) == adapter.lint_signature(b)


def test_lint_signature_empty_when_no_findings() -> None:
    assert adapter.lint_signature("") == ""
    assert adapter.lint_signature("[]") == ""


def test_lint_signature_empty_on_unparseable_output() -> None:
    assert adapter.lint_signature("ruff: command not found") == ""


def test_parse_typecheck_counts_error_lines() -> None:
    output = "a.py:3: error: bad\nb.py:5: note: fyi\nc.py:9: error: also bad\n"
    assert adapter.parse_typecheck(output) == 2


def test_type_signature_distinguishes_different_errors() -> None:
    a = "a.py:3: error: bad [arg-type]\n"
    b = "a.py:3: error: bad [assignment]\n"
    assert adapter.type_signature(a) != adapter.type_signature(b)


def test_type_signature_ignores_notes() -> None:
    with_note = "a.py:3: error: bad [arg-type]\nb.py:5: note: fyi\n"
    without_note = "a.py:3: error: bad [arg-type]\n"
    assert adapter.type_signature(with_note) == adapter.type_signature(without_note)


def test_type_signature_empty_when_no_errors() -> None:
    assert adapter.type_signature("") == ""
    assert adapter.type_signature("b.py:5: note: fyi\n") == ""


def test_parse_test_failures_bare_assert_defaults_to_assertion_error() -> None:
    """Regression: pytest 8.x omits the 'AssertionError:' prefix for bare
    `assert` statements — found by actually running a failing test through
    the sandbox, not by unit-testing a synthetic log string."""
    output = (
        "FAILED tests/test_mod.py::test_add - assert 4 == 5\n1 failed in 0.01s\n"
    )
    summary = adapter.parse_test_failures(output)
    assert summary.tests_failed == 1
    assert summary.failing_test_ids == ("tests/test_mod.py::test_add",)
    assert summary.exception_type == "AssertionError"
    assert summary.failing_location == "tests/test_mod.py::test_add"


def test_parse_test_failures_explicit_exception_type() -> None:
    output = (
        "FAILED tests/test_x.py::test_a - ValueError: invalid literal: 'x'\n1 failed in 0.01s\n"
    )
    summary = adapter.parse_test_failures(output)
    assert summary.exception_type == "ValueError"


def test_parse_test_failures_counts_passed_and_failed() -> None:
    output = "3 passed, 2 failed in 0.10s\n"
    summary = adapter.parse_test_failures(output)
    assert summary.tests_run == 5
    assert summary.tests_failed == 2


def test_parse_coverage_reads_total_percent() -> None:
    output = "TOTAL              120     12    90%\n"
    assert adapter.parse_coverage(output) == 90.0


def test_parse_coverage_none_when_absent() -> None:
    assert adapter.parse_coverage("no coverage section here") is None


def test_install_cmd_none_with_no_dependency_manifest(tmp_path: Path) -> None:
    assert adapter.install_cmd(tmp_path) is None


def test_install_cmd_uses_requirements_txt(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("six==1.16.0\n")
    cmd = adapter.install_cmd(tmp_path)
    assert cmd is not None
    assert cmd[-2:] == ["-r", "requirements.txt"]


def test_install_cmd_none_for_empty_requirements_txt(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("   \n")
    assert adapter.install_cmd(tmp_path) is None


def test_install_cmd_none_when_pyproject_has_no_dependencies(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = []\n'
    )
    assert adapter.install_cmd(tmp_path) is None


def test_install_cmd_pyproject_with_dependencies(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["six"]\n'
    )
    cmd = adapter.install_cmd(tmp_path)
    assert cmd is not None
    assert cmd[-2:] == ["-e", "."]


def test_dockerfile_matches_the_image_tags_python_version() -> None:
    dockerfile = adapter.dockerfile()
    assert dockerfile is not None
    tag_version = adapter.image().rsplit(":", 1)[1]
    assert f"python:{tag_version}" in dockerfile


def test_pyproject_toml_fixture_parses(tmp_path: Path) -> None:
    # Sanity check the tomllib-based branch against a real TOML document.
    text = '[project]\nname = "x"\ndependencies = ["a", "b"]\n'
    (tmp_path / "pyproject.toml").write_text(text)
    assert tomllib.loads(text)["project"]["dependencies"] == ["a", "b"]
