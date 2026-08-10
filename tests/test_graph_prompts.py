"""Wire-format parsing (PRD S10): Daedalus's <xeno-file>/<xeno-objection>,
Chiron's <xeno-file>/<xeno-decline>, Odysseus's <xeno-task>/<xeno-objection>,
and Argus's <xeno-file>/<xeno-no-files>. Direct unit coverage rather than
only exercising these through the full graph, since local-model drift
toward prose/fences is a real failure mode discovered against a live model
(xeno.graph.daedalus's format-retry, xeno.graph.context's focus scoping)."""

from __future__ import annotations

from xeno.core.types import NodeRole, Verdict
from xeno.graph.prompts import (
    parse_argus_research_output,
    parse_cerberus_output,
    parse_chiron_output,
    parse_daedalus_output,
    parse_discovery_output,
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


def test_cerberus_parses_approve() -> None:
    text = (
        "<xeno-verdict>approve</xeno-verdict>\n"
        "<xeno-commit-message>\nfeat: add widget\n\n"
        "Why: users asked for it.\n</xeno-commit-message>\n"
        "<xeno-notes>looks good</xeno-notes>"
    )
    out = parse_cerberus_output(text)
    assert not out.malformed
    assert out.verdict is Verdict.APPROVE
    assert out.commit_message == "feat: add widget\n\nWhy: users asked for it."
    assert out.notes == "looks good"


def test_cerberus_approve_without_notes() -> None:
    text = (
        "<xeno-verdict>approve</xeno-verdict>\n<xeno-commit-message>fix: bug</xeno-commit-message>"
    )
    out = parse_cerberus_output(text)
    assert not out.malformed
    assert out.verdict is Verdict.APPROVE
    assert out.commit_message == "fix: bug"
    assert out.notes is None


def test_cerberus_approve_missing_commit_message_is_malformed() -> None:
    out = parse_cerberus_output("<xeno-verdict>approve</xeno-verdict>")
    assert out.malformed
    assert out.verdict is None


def test_cerberus_parses_reject_and_return_to_daedalus() -> None:
    text = (
        "<xeno-verdict>reject_and_return</xeno-verdict>\n"
        "<xeno-destination>daedalus</xeno-destination>\n"
        "<xeno-objections>\nthe error is swallowed silently\n</xeno-objections>"
    )
    out = parse_cerberus_output(text)
    assert not out.malformed
    assert out.verdict is Verdict.REJECT_AND_RETURN
    assert out.destination is NodeRole.CODER
    assert out.objections == "the error is swallowed silently"


def test_cerberus_parses_reject_and_return_to_odysseus() -> None:
    text = (
        "<xeno-verdict>reject_and_return</xeno-verdict>\n"
        "<xeno-destination>odysseus</xeno-destination>\n"
        "<xeno-objections>the plan never covers the rate-limit reset</xeno-objections>"
    )
    out = parse_cerberus_output(text)
    assert out.destination is NodeRole.PLANNER


def test_cerberus_reject_unknown_destination_is_malformed() -> None:
    text = (
        "<xeno-verdict>reject_and_return</xeno-verdict>\n"
        "<xeno-destination>talos</xeno-destination>\n"
        "<xeno-objections>bad</xeno-objections>"
    )
    out = parse_cerberus_output(text)
    assert out.malformed


def test_cerberus_reject_missing_objections_is_malformed() -> None:
    text = (
        "<xeno-verdict>reject_and_return</xeno-verdict>\n"
        "<xeno-destination>daedalus</xeno-destination>"
    )
    out = parse_cerberus_output(text)
    assert out.malformed


def test_cerberus_parses_escalate() -> None:
    text = (
        "<xeno-verdict>escalate</xeno-verdict>\n"
        "<xeno-report>\nthis needs a human call\n</xeno-report>"
    )
    out = parse_cerberus_output(text)
    assert not out.malformed
    assert out.verdict is Verdict.ESCALATE
    assert out.report == "this needs a human call"


def test_cerberus_escalate_missing_report_is_malformed() -> None:
    out = parse_cerberus_output("<xeno-verdict>escalate</xeno-verdict>")
    assert out.malformed


def test_cerberus_unknown_verdict_is_malformed() -> None:
    out = parse_cerberus_output("<xeno-verdict>maybe</xeno-verdict>")
    assert out.malformed
    assert out.verdict is None


def test_cerberus_prose_with_no_tags_is_malformed() -> None:
    out = parse_cerberus_output("I think this looks fine overall...")
    assert out.malformed
    assert out.verdict is None


def test_discovery_parses_install_and_required_and_advisory() -> None:
    text = (
        "<xeno-install>pip install -e .</xeno-install>\n"
        '<xeno-required-command name="lint">ruff check .</xeno-required-command>\n'
        '<xeno-required-command name="test">pytest -q</xeno-required-command>\n'
        '<xeno-advisory-command name="coverage">pytest -q --cov=.</xeno-advisory-command>'
    )
    out = parse_discovery_output(text)
    assert not out.malformed
    assert out.install == "pip install -e ."
    assert out.required == (("lint", "ruff check ."), ("test", "pytest -q"))
    assert out.advisory == (("coverage", "pytest -q --cov=."),)


def test_discovery_install_is_optional() -> None:
    text = '<xeno-required-command name="test">pytest -q</xeno-required-command>'
    out = parse_discovery_output(text)
    assert not out.malformed
    assert out.install is None
    assert out.required == (("test", "pytest -q"),)


def test_discovery_advisory_is_optional() -> None:
    text = '<xeno-required-command name="test">pytest -q</xeno-required-command>'
    out = parse_discovery_output(text)
    assert out.advisory == ()


def test_discovery_no_required_commands_is_malformed() -> None:
    out = parse_discovery_output("<xeno-install>pip install -e .</xeno-install>")
    assert out.malformed
    assert out.required == ()


def test_discovery_prose_with_no_tags_is_malformed() -> None:
    out = parse_discovery_output("This repo uses pytest for testing.")
    assert out.malformed
