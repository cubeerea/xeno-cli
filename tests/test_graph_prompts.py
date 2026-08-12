"""Wire-format parsing (PRD S10): Daedalus's <xeno-file>/<xeno-objection>,
Chiron's <xeno-file>/<xeno-decline>, Odysseus's
<xeno-milestone>/<xeno-objection>, Lachesis's <xeno-task> and test-file
blocks, and Argus's <xeno-file>/<xeno-no-files>. Direct unit coverage rather
than only exercising these through the full graph, since local-model drift
toward prose/fences is a real failure mode discovered against a live model
(xeno.graph.daedalus's format-retry, xeno.graph.context's focus scoping)."""

from __future__ import annotations

import pytest

from xeno.core.types import NodeRole, Verdict
from xeno.graph.prompts import (
    parse_argus_research_output,
    parse_cerberus_output,
    parse_chiron_output,
    parse_daedalus_output,
    parse_discovery_output,
    parse_roadmap_output,
    parse_task_list_output,
    parse_test_files_output,
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


def test_lachesis_parses_a_single_task() -> None:
    out = parse_task_list_output(
        '<xeno-task acceptance="tests pass">write pkg/mod.py</xeno-task>'
    )
    assert not out.is_objection
    assert not out.malformed
    assert len(out.tasks) == 1
    assert out.tasks[0].description == "write pkg/mod.py"
    assert out.tasks[0].acceptance == "tests pass"


def test_lachesis_parses_multiple_tasks_in_order() -> None:
    text = (
        '<xeno-task acceptance="a exists">write a</xeno-task>'
        '<xeno-task acceptance="b exists">write b</xeno-task>'
    )
    out = parse_task_list_output(text)
    assert [t.description for t in out.tasks] == ["write a", "write b"]
    assert [t.acceptance for t in out.tasks] == ["a exists", "b exists"]


def test_lachesis_accepts_single_quoted_acceptance() -> None:
    out = parse_task_list_output("<xeno-task acceptance='tests pass'>write pkg/mod.py</xeno-task>")
    assert out.tasks[0].acceptance == "tests pass"


def test_lachesis_genuine_objection_is_not_malformed() -> None:
    out = parse_task_list_output("<xeno-objection>the goal is impossible</xeno-objection>")
    assert out.is_objection
    assert out.objection == "the goal is impossible"
    assert not out.malformed


def test_lachesis_prose_with_no_tags_is_malformed_objection() -> None:
    out = parse_task_list_output("Sure, here's my plan:\n1. First...\n2. Then...")
    assert out.is_objection
    assert out.malformed


def test_lachesis_truncates_an_overlong_task_field() -> None:
    out = parse_task_list_output(f'<xeno-task acceptance="ok">{"x" * 3000}</xeno-task>')
    assert len(out.tasks[0].description) == 2000  # PlanTask.description's max_length


def test_lachesis_test_files_parse_as_writes() -> None:
    out = parse_test_files_output(
        '<xeno-file path="tests/test_convert.py">def test_x(): pass\n</xeno-file>'
    )
    assert not out.is_objection
    assert out.files[0].path == "tests/test_convert.py"


def test_lachesis_prose_instead_of_tests_is_a_malformed_objection() -> None:
    out = parse_test_files_output("I would test the conversion function thoroughly.")
    assert out.is_objection
    assert out.malformed


def test_odysseus_parses_a_single_milestone() -> None:
    out = parse_roadmap_output(
        '<xeno-milestone outcome="converts C to F">build the conversion core</xeno-milestone>'
    )
    assert not out.is_objection
    assert not out.malformed
    assert len(out.milestones) == 1
    assert out.milestones[0].description == "build the conversion core"
    assert out.milestones[0].outcome == "converts C to F"


def test_odysseus_parses_multiple_milestones_in_order() -> None:
    text = (
        '<xeno-milestone outcome="a works">build a</xeno-milestone>'
        "<xeno-milestone outcome='b works'>build b</xeno-milestone>"
    )
    out = parse_roadmap_output(text)
    assert [m.description for m in out.milestones] == ["build a", "build b"]
    assert [m.outcome for m in out.milestones] == ["a works", "b works"]


def test_odysseus_genuine_objection_is_not_malformed() -> None:
    out = parse_roadmap_output("<xeno-objection>the goal is impossible</xeno-objection>")
    assert out.is_objection
    assert out.objection == "the goal is impossible"
    assert not out.malformed


def test_odysseus_prose_with_no_tags_is_a_malformed_objection() -> None:
    out = parse_roadmap_output("Sure, here's my roadmap:\n1. First...\n2. Then...")
    assert out.is_objection
    assert out.malformed


def test_a_roadmap_of_tasks_does_not_parse_as_a_roadmap() -> None:
    """The two jobs use different tags on purpose. Odysseus emitting the
    task format is Odysseus doing Lachesis's job a milestone too early, and
    it has to fail loudly rather than half-parse."""
    out = parse_roadmap_output('<xeno-task acceptance="ok">write mod.py</xeno-task>')
    assert out.is_objection
    assert out.malformed


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


# ---------------------------------------------------------------------------
# Attribute tolerance (regression: run 20260811T124116)
# ---------------------------------------------------------------------------
#
# The three block regexes used to exclude BOTH quote characters from an
# attribute value, so an apostrophe in an acceptance criterion made the whole
# block unmatchable. The failure is per-block, which is what makes it
# dangerous: four tasks out of five could vanish leaving a plan of one, no
# objection, and nothing in the run log. These pin the tolerances rather than
# the implementation, since the point is which model outputs survive.


@pytest.mark.parametrize(
    ("label", "acceptance"),
    [
        ("apostrophe", "the parser's output is a tree"),
        ("possessive plus contraction", "parse() doesn't touch the caller's input"),
        ("named after a person", "handles L'Hopital's rule"),
        ("nested double quotes", 'parse("2+2") returns an Add node'),
        ("angle bracket in the value", "limit(x->0, sin(x)/x) evaluates to 1"),
        ("both quote styles", "parse('x') and parse(\"x\") agree"),
    ],
)
def test_an_acceptance_criterion_may_contain_quotes(label: str, acceptance: str) -> None:
    """XML's own rule: a value may carry the other delimiter. A symbolic-maths
    project writes criteria like these constantly, and every one of them used
    to silently discard its task."""
    out = parse_task_list_output(f'<xeno-task acceptance="{acceptance}">do it</xeno-task>')
    assert len(out.tasks) == 1, label
    assert out.tasks[0].acceptance == acceptance


def test_one_unreadable_task_no_longer_takes_its_neighbours_with_it() -> None:
    """The silent partial loss, which had no symptom at all: the plan looked
    complete and was simply short, so the milestone was built one-third
    implemented and everything downstream agreed it was fine."""
    text = (
        '<xeno-task acceptance="a exists">one</xeno-task>'
        "<xeno-task acceptance=\"it's b\">two</xeno-task>"
        '<xeno-task acceptance="c exists">three</xeno-task>'
    )
    out = parse_task_list_output(text)
    assert [t.description for t in out.tasks] == ["one", "two", "three"]
    assert out.dropped == 0


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("attribute after", '<xeno-task acceptance="a" id="1">do it</xeno-task>'),
        ("attribute before", '<xeno-task id="1" acceptance="a">do it</xeno-task>'),
        ("spaces around =", '<xeno-task acceptance = "a">do it</xeno-task>'),
        ("underscore in the tag", '<xeno_task acceptance="a">do it</xeno_task>'),
        ("single quotes", "<xeno-task acceptance='a'>do it</xeno-task>"),
    ],
)
def test_ordinary_model_drift_on_tag_syntax_is_tolerated(label: str, text: str) -> None:
    out = parse_task_list_output(text)
    assert len(out.tasks) == 1, label


def test_a_roadmap_outcome_may_contain_an_apostrophe_too() -> None:
    """Odysseus survived the same bug by one character position: its roadmap
    said "L'Hopital's rule" in a milestone BODY, where it was harmless. One
    field over and the run would have died before Lachesis ever ran."""
    out = parse_roadmap_output(
        "<xeno-milestone outcome=\"the user's expression parses\">m1</xeno-milestone>"
    )
    assert len(out.milestones) == 1


def test_a_file_path_is_still_required_to_be_non_empty() -> None:
    out = parse_daedalus_output('<xeno-file path="">x = 1</xeno-file>')
    assert out.malformed


# ---- what an unusable response says --------------------------------------


def test_tags_that_could_not_be_read_are_reported_as_such() -> None:
    """Not the same failure as prose, and not the same correction. Telling a
    model that emitted well-formed tags "no tag was found" wastes the one
    corrective retry on a fault that is not there — which is how the run that
    prompted this work burned its second call."""
    out = parse_task_list_output("<xeno-task>no attribute here</xeno-task>")
    assert out.malformed
    assert "1 <xeno-task> tag(s) the harness could not read" in (out.objection or "")
    assert "acceptance=" in out.format_hint


def test_a_response_with_no_tags_at_all_says_that_instead() -> None:
    out = parse_task_list_output("Sure! Here is the plan:\n1. Build it\n2. Ship it")
    assert out.malformed
    assert "contained no <xeno-task> block" in (out.objection or "")
    assert out.format_hint == ""


def test_an_unterminated_block_is_reported_as_a_possible_truncation() -> None:
    out = parse_task_list_output('<xeno-task acceptance="a">do it')
    assert out.malformed
    assert "cut off" in out.format_hint


# ---- objection precedence -------------------------------------------------


def test_a_caveat_does_not_discard_the_work_it_is_a_caveat_about() -> None:
    """The objection tag used to be searched first and unconditionally, so a
    response that did the work and added one note about it had all of the
    work thrown away in favour of the note."""
    text = (
        '<xeno-task acceptance="a">one</xeno-task>'
        '<xeno-task acceptance="b">two</xeno-task>'
        "<xeno-objection>note: this milestone is on the large side</xeno-objection>"
    )
    out = parse_task_list_output(text)
    assert len(out.tasks) == 2
    assert not out.is_objection


def test_the_same_holds_for_writes() -> None:
    text = (
        '<xeno-file path="a.py">x = 1</xeno-file>'
        "<xeno-objection>I could not also do b.py</xeno-objection>"
    )
    assert len(parse_daedalus_output(text).files) == 1
    assert len(parse_test_files_output(text).files) == 1


def test_an_objection_on_its_own_is_still_honoured() -> None:
    """The whole point of the tag: a deliberate refusal must still stop the
    run rather than being read as an empty task list."""
    out = parse_task_list_output(
        "<xeno-objection>this repository is not the project</xeno-objection>"
    )
    assert out.is_objection
    assert not out.malformed
