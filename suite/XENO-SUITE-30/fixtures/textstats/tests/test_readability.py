from __future__ import annotations

import pytest

from textstats.readability import flesch_reading_ease, syllables_per_word, words_per_sentence

EASY = "The cat sat on the mat."
HARDER = "Readable prose uses short sentences and common words."


def test_words_per_sentence() -> None:
    assert words_per_sentence(EASY) == 6.0
    assert words_per_sentence("One two. Three four.") == 2.0
    assert words_per_sentence("") == 0.0


def test_syllables_per_word() -> None:
    assert syllables_per_word(EASY) == 1.0
    assert syllables_per_word("banana") == 3.0
    assert syllables_per_word("") == 0.0


def test_flesch_reading_ease_on_known_inputs() -> None:
    assert flesch_reading_ease(EASY) == pytest.approx(116.15)
    assert flesch_reading_ease(HARDER) == pytest.approx(50.67)


def test_flesch_reading_ease_rewards_simpler_text() -> None:
    assert flesch_reading_ease(EASY) > flesch_reading_ease(HARDER)


def test_flesch_reading_ease_of_empty_text_is_zero() -> None:
    assert flesch_reading_ease("") == 0.0
    assert flesch_reading_ease("!!!") == 0.0
