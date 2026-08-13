"""History condensation and the map's tree cap (item 3 of the context work).

The measurement that motivated this: Daedalus's accumulated history reached
~52,000 tokens by task 20 on this repo's own source, against a codebase map
hard-capped near 6,000 — because every assistant turn carried each file it
wrote verbatim, and history is append-only. The same bytes were already in the
map one block up, so the model was reading each file twice and paying for the
larger copy forever.
"""

from __future__ import annotations

from pathlib import Path

from xeno.graph.context import DEFAULT_MAX_TREE_BYTES, build_codebase_map, focus_paths
from xeno.graph.nodeops import condense_file_blocks
from xeno.graph.prompts import parse_daedalus_output

BIG = "x = 1\n" * 400


def _response(path: str, body: str) -> str:
    return f'Here is the change.\n<xeno-file path="{path}">\n{body}\n</xeno-file>\nDone.'


# ---- condensation ---------------------------------------------------------


def test_a_large_file_body_is_replaced_by_a_reference() -> None:
    condensed = condense_file_blocks(_response("src/a.py", BIG))
    assert BIG not in condensed
    assert "src/a.py" in condensed
    assert "CODEBASE MAP" in condensed
    assert len(condensed) < 400


def test_the_decision_survives_even_though_the_content_does_not() -> None:
    """History's job here is to record that this node wrote this path on
    purpose. That is what a later turn reasons from."""
    condensed = condense_file_blocks(_response("src/a.py", BIG))
    assert '<xeno-file path="src/a.py">' in condensed
    assert "Here is the change." in condensed
    assert "Done." in condensed


def test_a_small_body_is_left_alone() -> None:
    """Below the threshold the reference costs more than the body."""
    small = "x = 1\n"
    assert condense_file_blocks(_response("src/a.py", small)) == _response("src/a.py", small)


def test_every_block_in_a_multi_file_response_is_condensed() -> None:
    text = _response("a.py", BIG) + _response("b.py", BIG)
    condensed = condense_file_blocks(text)
    assert BIG not in condensed
    assert "a.py" in condensed and "b.py" in condensed


def test_the_marker_does_not_claim_the_write_landed() -> None:
    """A response enters history BEFORE its write is attempted — a test-file
    refusal rejects the whole thing afterwards. A marker asserting the bytes
    reached disk would be a lie the model then reasons from."""
    condensed = condense_file_blocks(_response("tests/test_a.py", BIG))
    assert "emitted" in condensed
    assert "if the write landed" in condensed


def test_condensing_does_not_change_what_the_node_acted_on() -> None:
    """`parse` sees the verbatim response; condensation applies only on the
    way into history. If these ever swapped, the harness would write the
    reference text to disk as the file's contents."""
    raw = _response("src/a.py", BIG)
    parsed = parse_daedalus_output(raw)
    assert parsed.files[0].content.strip() == BIG.strip()
    assert parse_daedalus_output(condense_file_blocks(raw)).files[0].content != BIG


def test_a_response_with_no_file_blocks_is_untouched() -> None:
    objection = "<xeno-objection>the task is underspecified</xeno-objection>"
    assert condense_file_blocks(objection) == objection


def test_a_malformed_block_is_left_intact() -> None:
    """An unclosed block is exactly what the corrective retry needs to see."""
    broken = '<xeno-file path="src/a.py">\n' + BIG
    assert condense_file_blocks(broken) == broken


def test_condensation_is_what_keeps_history_flat(tmp_path: Path) -> None:
    """The point of the whole exercise, as a ratio rather than an anecdote."""
    raw = sum(len(_response(f"src/m{i}.py", BIG)) for i in range(20))
    condensed = sum(len(condense_file_blocks(_response(f"src/m{i}.py", BIG))) for i in range(20))
    assert condensed < raw // 10


# ---- the precondition -----------------------------------------------------


def test_written_files_join_the_map_focus() -> None:
    """What makes condensation lossless. `focus` is a FILTER, and Argus
    selects before the write exists, so without this a file Daedalus just
    wrote is narrowed out of the map — and with its body gone from history
    too, no copy would remain."""
    handles = [Path("/w/picked.py")]
    written = [Path("/w/just_wrote.py")]
    assert set(focus_paths(handles, written)) == {Path("/w/picked.py"), Path("/w/just_wrote.py")}


def test_focus_does_not_duplicate_a_file_that_is_both() -> None:
    same = Path("/w/a.py")
    assert focus_paths([same], [same]) == [same]


def test_a_written_file_survives_into_the_rendered_map(tmp_path: Path) -> None:
    """End to end: Argus picks one file, Daedalus writes another, and both
    are visible in the map the next call assembles."""
    (tmp_path / "picked.py").write_text("PICKED = 1\n")
    (tmp_path / "written.py").write_text("WRITTEN = 2\n")

    narrowed = build_codebase_map(tmp_path, focus=[tmp_path / "picked.py"])
    assert "WRITTEN = 2" not in narrowed, "the regression this guards against"

    widened = build_codebase_map(
        tmp_path, focus=focus_paths([tmp_path / "picked.py"], [tmp_path / "written.py"])
    )
    assert "PICKED = 1" in widened
    assert "WRITTEN = 2" in widened


# ---- the tree cap ---------------------------------------------------------


def test_the_tree_listing_is_capped(tmp_path: Path) -> None:
    """The one part of the map with no bound: content was budgeted, but the
    listing printed every path, so its size was set by file count."""
    for i in range(2_000):
        (tmp_path / f"module_with_a_longish_name_{i:04d}.py").write_text("x = 1\n")

    tree = build_codebase_map(tmp_path).split("File contents:")[0]
    assert len(tree) < DEFAULT_MAX_TREE_BYTES + 200


def test_a_truncated_tree_says_how_much_it_dropped(tmp_path: Path) -> None:
    """A tree that trails off silently reads like a project that genuinely
    does not contain those files, and a node will act on that."""
    for i in range(2_000):
        (tmp_path / f"module_with_a_longish_name_{i:04d}.py").write_text("x = 1\n")

    tree = build_codebase_map(tmp_path).split("File contents:")[0]
    assert "more path(s) not listed" in tree


def test_a_small_tree_is_listed_in_full(tmp_path: Path) -> None:
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("x = 1\n")

    tree = build_codebase_map(tmp_path).split("File contents:")[0]
    assert "not listed" not in tree
    for name in ("a.py", "b.py", "c.py"):
        assert name in tree
