from __future__ import annotations

from textstats.tokenizer import (
    sentence_count,
    sentences,
    syllable_count,
    tokenize,
    total_syllables,
)


def test_tokenize_lowercases_and_drops_surrounding_punctuation() -> None:
    assert tokenize("Hello, world!") == ["hello", "world"]
    assert tokenize("  Spaced   OUT  ") == ["spaced", "out"]


def test_tokenize_ignores_digits_and_symbols() -> None:
    assert tokenize("abc 123 def") == ["abc", "def"]
    assert tokenize("42 + 7 = 49") == []


def test_tokenize_empty_and_punctuation_only() -> None:
    assert tokenize("") == []
    assert tokenize("!!! ??? ...") == []


def test_sentences_are_stripped_and_non_empty() -> None:
    assert sentences("One. Two! Three?  ") == ["One", "Two", "Three"]
    assert sentences("Wait... really?!") == ["Wait", "really"]


def test_sentence_count_without_terminator_is_one() -> None:
    assert sentence_count("no terminator here") == 1
    assert sentence_count("") == 0


def test_syllable_count_uses_vowel_groups() -> None:
    assert syllable_count("cat") == 1
    assert syllable_count("banana") == 3
    assert syllable_count("beautiful") == 3


def test_syllable_count_handles_silent_and_kept_final_e() -> None:
    assert syllable_count("coding") == 2
    assert syllable_count("table") == 2
    assert syllable_count("see") == 1


def test_syllable_count_of_wordless_input_is_zero() -> None:
    assert syllable_count("123") == 0
    assert syllable_count("") == 0


def test_total_syllables_sums_over_tokens() -> None:
    assert total_syllables("The cat sat on the mat.") == 6
    assert total_syllables("") == 0
