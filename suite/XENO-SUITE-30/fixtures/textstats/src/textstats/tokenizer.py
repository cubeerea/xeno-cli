"""Low-level text splitting: words, sentences and syllable estimates.

Everything in this module is deliberately deterministic and dependency-free so
that the numbers produced by :mod:`textstats.stats` and
:mod:`textstats.readability` are reproducible across machines.
"""

from __future__ import annotations

import re

#: Matches a run of ASCII letters. Every other character acts as a separator,
#: so ``"state-of-the-art"`` currently yields four separate tokens.
WORD_RE = re.compile(r"[a-z]+")

#: Characters that terminate a sentence.
SENTENCE_ENDINGS = ".!?"

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

_VOWELS = frozenset("aeiouy")

#: Word endings whose trailing ``e`` is *not* treated as silent.
_KEEP_FINAL_E = ("le", "ee", "ie", "oe", "ye")


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercase word tokens.

    Any character outside ``a-z`` acts as a separator, so punctuation --
    including hyphens and apostrophes that sit inside a word -- breaks the
    token apart.

    >>> tokenize("Hello, world!")
    ['hello', 'world']
    """
    return WORD_RE.findall(text.lower())


def sentences(text: str) -> list[str]:
    """Split ``text`` into stripped sentences.

    Runs of terminators collapse into a single boundary and empty fragments
    are discarded, so trailing punctuation never produces a phantom sentence.
    """
    parts = (part.strip() for part in _SENTENCE_SPLIT_RE.split(text))
    return [part for part in parts if part]


def sentence_count(text: str) -> int:
    """Number of sentences in ``text``.

    Text that contains words but no terminator counts as one sentence.
    """
    return len(sentences(text))


def syllable_count(word: str) -> int:
    """Estimate the number of syllables in ``word``.

    The heuristic counts vowel groups and drops a silent trailing ``e``. Any
    word containing at least one letter reports at least one syllable.
    """
    letters = [char for char in word.lower() if char.isascii() and char.isalpha()]
    if not letters:
        return 0

    normalised = "".join(letters)
    count = 0
    previous_was_vowel = False
    for char in normalised:
        is_vowel = char in _VOWELS
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel

    if normalised.endswith("e") and not normalised.endswith(_KEEP_FINAL_E) and count > 1:
        count -= 1

    return max(count, 1)


def total_syllables(text: str) -> int:
    """Sum of :func:`syllable_count` over every token in ``text``."""
    return sum(syllable_count(token) for token in tokenize(text))
