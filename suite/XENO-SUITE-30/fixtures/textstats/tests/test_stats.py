from __future__ import annotations

from textstats.stats import analyse, average_word_length, top_n, word_frequencies

SAMPLE = "The cat sat on the mat. The mat was flat."


def test_word_frequencies_counts_and_orders() -> None:
    assert word_frequencies("the cat sat on the mat") == {
        "the": 2,
        "cat": 1,
        "mat": 1,
        "on": 1,
        "sat": 1,
    }
    assert list(word_frequencies(SAMPLE))[0] == "the"


def test_word_frequencies_of_empty_text() -> None:
    assert word_frequencies("") == {}


def test_top_n_breaks_ties_alphabetically() -> None:
    assert top_n(SAMPLE, 3) == [("the", 3), ("mat", 2), ("cat", 1)]


def test_top_n_clamps_and_rejects_non_positive() -> None:
    assert top_n("only one", 10) == [("one", 1), ("only", 1)]
    assert top_n(SAMPLE, 0) == []
    assert top_n(SAMPLE, -2) == []


def test_average_word_length_rounds_to_two_places() -> None:
    assert average_word_length("ab abcd") == 3.0
    assert average_word_length("a bb cccc") == 2.33
    assert average_word_length("") == 0.0


def test_analyse_reports_each_field() -> None:
    stats = analyse(SAMPLE)
    assert stats.word_count == 10
    assert stats.unique_words == 7
    assert stats.sentence_count == 2
    assert stats.avg_word_length == 3.0


def test_analyse_of_empty_text_is_all_zero() -> None:
    stats = analyse("")
    assert stats.word_count == 0
    assert stats.unique_words == 0
    assert stats.sentence_count == 0
    assert stats.avg_word_length == 0.0
