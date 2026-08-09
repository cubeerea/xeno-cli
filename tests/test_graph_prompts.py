"""Wire-format parsing (PRD S10): Daedalus's <xeno-file>/<xeno-objection> and
Chiron's <xeno-file>/<xeno-decline>. Direct unit coverage rather than only
exercising these through the full graph, since local-model drift toward
prose/fences is a real failure mode discovered against a live model
(xeno.graph.daedalus's format-retry, xeno.graph.context's focus scoping)."""

from __future__ import annotations

from xeno.graph.prompts import parse_chiron_output, parse_daedalus_output


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
