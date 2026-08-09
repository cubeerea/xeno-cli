"""Readability scoring.

Only the Flesch reading-ease score is implemented; the helper terms it needs
are exposed separately because they are useful on their own.
"""

from __future__ import annotations

from textstats.tokenizer import sentence_count, tokenize, total_syllables

#: Coefficients of the Flesch reading-ease formula.
FLESCH_BASE = 206.835
FLESCH_SENTENCE_WEIGHT = 1.015
FLESCH_SYLLABLE_WEIGHT = 84.6


def words_per_sentence(text: str) -> float:
    """Mean number of word tokens per sentence.

    Returns ``0.0`` for text with no sentences, which keeps the readability
    formulas defined for empty input.
    """
    count = sentence_count(text)
    if count == 0:
        return 0.0
    return len(tokenize(text)) / count


def syllables_per_word(text: str) -> float:
    """Mean number of syllables per word token.

    Returns ``0.0`` when ``text`` contains no words.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return total_syllables(text) / len(tokens)


def flesch_reading_ease(text: str) -> float:
    """Flesch reading-ease score for ``text``, rounded to 2 decimal places.

    Higher is easier: roughly 90-100 for very simple prose and below 30 for
    dense academic writing. Empty text scores ``0.0`` rather than the
    degenerate maximum.
    """
    if not tokenize(text):
        return 0.0
    score = (
        FLESCH_BASE
        - FLESCH_SENTENCE_WEIGHT * words_per_sentence(text)
        - FLESCH_SYLLABLE_WEIGHT * syllables_per_word(text)
    )
    return round(score, 2)
