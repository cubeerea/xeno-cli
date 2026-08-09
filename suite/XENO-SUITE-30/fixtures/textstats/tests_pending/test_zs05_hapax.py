"""ZS-05 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from textstats.stats import analyse, hapax_legomena

SAMPLE = "The cat sat on the mat. The mat was flat."


def test_hapax_legomena_is_alphabetically_sorted() -> None:
    assert hapax_legomena("the cat sat on the mat") == ["cat", "mat", "on", "sat"]
    assert hapax_legomena(SAMPLE) == ["cat", "flat", "on", "sat", "was"]


def test_hapax_legomena_excludes_repeats_and_handles_empty_text() -> None:
    assert hapax_legomena("a a b b") == []
    assert hapax_legomena("") == []
    assert hapax_legomena("Alpha ALPHA beta") == ["beta"]


def test_analyse_reports_hapax_count() -> None:
    assert analyse(SAMPLE).hapax_count == 5
    assert analyse("the cat sat on the mat").hapax_count == 4
    assert analyse("").hapax_count == 0


def test_hapax_count_matches_hapax_legomena() -> None:
    for text in (SAMPLE, "one two two three three three", "solo"):
        assert analyse(text).hapax_count == len(hapax_legomena(text))


def test_existing_word_stats_fields_are_untouched() -> None:
    stats = analyse(SAMPLE)
    assert stats.word_count == 10
    assert stats.unique_words == 7
    assert stats.sentence_count == 2
    assert stats.avg_word_length == 3.0
    assert stats.hapax_count == 5
