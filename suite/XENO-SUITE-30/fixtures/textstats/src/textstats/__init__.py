"""A minimal text-statistics library.

Fixture for XENO-SUITE-30. Deliberately small, dependency-free and green on
ruff / mypy / pytest before any suite task is applied.
"""

from __future__ import annotations

from textstats.readability import flesch_reading_ease
from textstats.report import render_text_report
from textstats.stats import WordStats, analyse, top_n, word_frequencies
from textstats.tokenizer import sentences, syllable_count, tokenize

__all__ = [
    "WordStats",
    "analyse",
    "flesch_reading_ease",
    "render_text_report",
    "sentences",
    "syllable_count",
    "tokenize",
    "top_n",
    "word_frequencies",
]
__version__ = "0.1.0"
