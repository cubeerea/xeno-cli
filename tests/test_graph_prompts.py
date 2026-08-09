"""Wire-format parsing (PRD S10): Daedalus's <xeno-file>/<xeno-objection>,
Chiron's <xeno-file>/<xeno-decline>, Odysseus's <xeno-task>/<xeno-objection>,
and Argus's <xeno-file>/<xeno-no-files>. Direct unit coverage rather than
only exercising these through the full graph, since local-model drift
toward prose/fences is a real failure mode discovered against a live model
(xeno.graph.daedalus's format-retry, xeno.graph.context's focus scoping)."""

from __future__ import annotations

from xeno.graph.prompts import (
    parse_argus_research_output,
    parse_chiron_output,
    parse_daedalus_output,
    parse_odysseus_output,
)


def test_daedalus_parses_a_single_file_block() -> None:
    out = parse_daedalus_output('<xeno-file path="pkg/mod.py">x = 1\n</xeno-file>')
    assert not out.is_objection
    assert not out.malformed
    assert len(out.files) == 1
    assert out.files[0].path == "pkg/mod.py"
    assert out.files[0].content == "x = 1"


def test_daedalus_parses_multiple_file_blocks() -> None:
    text = (
        '<xeno-file path="a.py">A\n</xeno-file>\n'
        '<xeno-file path="b.py">B\n</xeno-file>'
    )
    out = parse_daedalus_output(text)
    assert [b.path for b in out.files] == ["a.py", "b.py"]
    assert [b.content for b in out.files] == ["A", "B"]


def test_daedalus_accepts_single_quoted_path() -> None:
    out = parse_daedalus_output("<xeno-file path='pkg/mod.py'>x = 1\n</xeno-file>")
    assert out.files[0].path == "pkg/mod.py"


def test_daedalus_strips_a_markdown_fence() -> None:
    out = parse_daedalus_output('<xeno-file path="a.py">```python\nx = 1\n```</xeno-file>')
    assert out.files[0].content == "x = 1"


def test_daedalus_genuine_objection_is_not_malformed() -> None:
    out = parse_daedalus_output("<xeno-objection>underspecified</xeno-objection>")
    assert out.is_objection
    assert out.objection == "underspecified"
    assert not out.malformed


def test_daedalus_prose_with_no_tags_is_malformed_objection() -> None:
    out = parse_daedalus_output("Sure! Here's what I'll do:\n1. First...\n2. Then...")
    assert out.is_objection
    assert out.malformed


def test_chiron_parses_a_patch() -> None:
    out = parse_chiron_output('<xeno-file path="pkg/mod.py">x = 2\n</xeno-file>')
    assert not out.declined
    assert not out.malformed
    assert out.files[0].content == "x = 2"


def test_chiron_genuine_decline_is_not_malformed() -> None:
    out = parse_chiron_output("<xeno-decline>no hypothesis fits the failure</xeno-decline>")
    assert out.declined
    assert not out.malformed
    assert out.decline_reason == "no hypothesis fits the failure"


def test_chiron_prose_with_no_tags_is_malformed_decline() -> None:
    out = parse_chiron_output("I think the issue is probably in the validator...")
    assert out.declined
    assert out.malformed


def test_odysseus_parses_a_single_task() -> None:
    out = parse_odysseus_output(
        '<xeno-task acceptance="tests pass">write pkg/mod.py</xeno-task>'
    )
    assert not out.is_objection
    assert not out.malformed
    assert len(out.tasks) == 1
    assert out.tasks[0].description == "write pkg/mod.py"
    assert out.tasks[0].acceptance == "tests pass"


def test_odysseus_parses_multiple_tasks_in_order() -> None:
    text = (
        '<xeno-task acceptance="a exists">write a</xeno-task>'
        '<xeno-task acceptance="b exists">write b</xeno-task>'
    )
    out = parse_odysseus_output(text)
    assert [t.description for t in out.tasks] == ["write a", "write b"]
    assert [t.acceptance for t in out.tasks] == ["a exists", "b exists"]


def test_odysseus_accepts_single_quoted_acceptance() -> None:
    out = parse_odysseus_output("<xeno-task acceptance='tests pass'>write pkg/mod.py</xeno-task>")
    assert out.tasks[0].acceptance == "tests pass"


def test_odysseus_genuine_objection_is_not_malformed() -> None:
    out = parse_odysseus_output("<xeno-objection>the goal is impossible</xeno-objection>")
    assert out.is_objection
    assert out.objection == "the goal is impossible"
    assert not out.malformed


def test_odysseus_prose_with_no_tags_is_malformed_objection() -> None:
    out = parse_odysseus_output("Sure, here's my plan:\n1. First...\n2. Then...")
    assert out.is_objection
    assert out.malformed


def test_odysseus_truncates_an_overlong_task_field() -> None:
    out = parse_odysseus_output(f'<xeno-task acceptance="ok">{"x" * 3000}</xeno-task>')
    assert len(out.tasks[0].description) == 2000  # PlanTask.description's max_length


def test_argus_parses_a_single_file() -> None:
    out = parse_argus_research_output(
        '<xeno-file path="pkg/mod.py" reason="existing convention"/>'
    )
    assert not out.malformed
    assert out.no_files_reason is None
    assert len(out.files) == 1
    assert out.files[0].path == "pkg/mod.py"
    assert out.files[0].reason == "existing convention"


def test_argus_parses_multiple_files() -> None:
    text = (
        '<xeno-file path="a.py" reason="r1"/>\n<xeno-file path="b.py" reason="r2"/>'
    )
    out = parse_argus_research_output(text)
    assert [f.path for f in out.files] == ["a.py", "b.py"]


def test_argus_accepts_a_non_self_closing_tag() -> None:
    out = parse_argus_research_output('<xeno-file path="a.py" reason="r1">')
    assert out.files[0].path == "a.py"


def test_argus_no_files_is_not_malformed() -> None:
    out = parse_argus_research_output("<xeno-no-files>pure net-new code</xeno-no-files>")
    assert not out.malformed
    assert out.files == ()
    assert out.no_files_reason == "pure net-new code"


def test_argus_prose_with_no_tags_is_malformed_no_files() -> None:
    out = parse_argus_research_output("I looked around and didn't find much...")
    assert out.malformed
    assert out.files == ()
    assert out.no_files_reason is not None
