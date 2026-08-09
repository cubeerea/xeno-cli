"""Aggregate word statistics built on top of :mod:`textstats.tokenizer`."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from textstats.tokenizer import sentence_count, tokenize


@dataclass(frozen=True)
class WordStats:
    """A summary of one block of text.

    Attributes:
        word_count: Total number of word tokens.
        unique_words: Number of distinct word tokens.
        sentence_count: Number of sentences detected.
        avg_word_length: Mean token length in characters, rounded to 2 places.
    """

    word_count: int
    unique_words: int
    sentence_count: int
    avg_word_length: float


def word_frequencies(text: str) -> dict[str, int]:
    """Map each token in ``text`` to the number of times it occurs.

    The mapping is ordered by descending frequency and then alphabetically, so
    iteration order is stable for a given input.
    """
    counts = Counter(tokenize(text))
    return {word: count for word, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}


def top_n(text: str, n: int) -> list[tuple[str, int]]:
    """Return the ``n`` most frequent ``(word, count)`` pairs in ``text``.

    Ties are broken alphabetically. Asking for more words than exist simply
    returns everything; a non-positive ``n`` returns an empty list.
    """
    if n <= 0:
        return []
    return list(word_frequencies(text).items())[:n]


def average_word_length(text: str) -> float:
    """Mean token length in ``text``, rounded to 2 decimal places."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return round(sum(len(token) for token in tokens) / len(tokens), 2)


def analyse(text: str) -> WordStats:
    """Compute a :class:`WordStats` summary for ``text``."""
    tokens = tokenize(text)
    return WordStats(
        word_count=len(tokens),
        unique_words=len(set(tokens)),
        sentence_count=sentence_count(text),
        avg_word_length=average_word_length(text),
    )
