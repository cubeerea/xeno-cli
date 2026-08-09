"""Human-readable rendering of a text analysis.

Only a plain-text renderer exists today; the labels are module constants so
callers can look for a line without hard-coding its spelling.
"""

from __future__ import annotations

from textstats.readability import flesch_reading_ease
from textstats.stats import analyse, top_n

TITLE = "Text statistics"
SEPARATOR = "-" * len(TITLE)

LABEL_WORDS = "Words"
LABEL_UNIQUE = "Unique words"
LABEL_SENTENCES = "Sentences"
LABEL_AVG_LENGTH = "Average word length"
LABEL_READING_EASE = "Reading ease"
LABEL_TOP_WORDS = "Top words"

#: How many entries the ``Top words`` line shows.
TOP_WORDS_SHOWN = 3


def format_top_words(text: str, n: int = TOP_WORDS_SHOWN) -> str:
    """Render the ``n`` most frequent words as ``word (count)`` pairs.

    Empty text renders as ``"(none)"`` so the line is never blank.
    """
    pairs = top_n(text, n)
    if not pairs:
        return "(none)"
    return ", ".join(f"{word} ({count})" for word, count in pairs)


def render_text_report(text: str) -> str:
    """Render a labelled plain-text report for ``text``.

    The first two lines are a title and an underline; every remaining line has
    the form ``"<label>: <value>"``.
    """
    stats = analyse(text)
    lines = [
        TITLE,
        SEPARATOR,
        f"{LABEL_WORDS}: {stats.word_count}",
        f"{LABEL_UNIQUE}: {stats.unique_words}",
        f"{LABEL_SENTENCES}: {stats.sentence_count}",
        f"{LABEL_AVG_LENGTH}: {stats.avg_word_length:.2f}",
        f"{LABEL_READING_EASE}: {flesch_reading_ease(text):.2f}",
        f"{LABEL_TOP_WORDS}: {format_top_words(text)}",
    ]
    return "\n".join(lines)
