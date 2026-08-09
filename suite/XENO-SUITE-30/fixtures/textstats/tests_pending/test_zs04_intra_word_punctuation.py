"""ZS-04 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from textstats.tokenizer import tokenize


def test_intra_word_hyphens_are_preserved() -> None:
    assert tokenize("state-of-the-art") == ["state-of-the-art"]
    assert tokenize("A state-of-the-art design.") == ["a", "state-of-the-art", "design"]
    assert tokenize("up-to-date well-known") == ["up-to-date", "well-known"]


def test_intra_word_apostrophes_are_preserved() -> None:
    assert tokenize("don't") == ["don't"]
    assert tokenize("Don't stop.") == ["don't", "stop"]
    assert tokenize("O'Brien said it wasn't over.") == ["o'brien", "said", "it", "wasn't", "over"]


def test_edge_punctuation_is_still_stripped() -> None:
    assert tokenize("-well-known-") == ["well-known"]
    assert tokenize("'quoted'") == ["quoted"]
    assert tokenize("well-known, up-to-date!") == ["well-known", "up-to-date"]
    assert tokenize("--start end--") == ["start", "end"]


def test_bare_hyphen_or_quote_is_never_a_token() -> None:
    assert tokenize("a - b") == ["a", "b"]
    assert tokenize("--- ' -- ''' ---") == []
    assert tokenize("one -- two") == ["one", "two"]
    assert tokenize("' - '") == []
    # A separator run next to a word with inner punctuation must not join them.
    assert tokenize("well-known -- fine") == ["well-known", "fine"]
    assert tokenize("it's - fine") == ["it's", "fine"]


def test_simple_tokenisation_is_unchanged() -> None:
    assert tokenize("Hello, world!") == ["hello", "world"]
    assert tokenize("The quick brown fox jumps over the lazy dog.") == [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "the",
        "lazy",
        "dog",
    ]
    assert tokenize("abc 123 def") == ["abc", "def"]
    assert tokenize("") == []
    assert tokenize("!!! ??? ...") == []
    # The new behaviour must apply to words that do contain inner punctuation.
    assert tokenize("mid-word") == ["mid-word"]
