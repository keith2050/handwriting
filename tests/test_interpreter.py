"""Tests for the shorthand interpreter."""

from __future__ import annotations

from pathlib import Path

import pytest

from handwriting.interpreter.apply import interpret, load_dict

DICT_PATH = Path(__file__).resolve().parents[1] / "handwriting" / "interpreter" / "shorthand.yml"


@pytest.fixture()
def mapping():
    return load_dict(DICT_PATH)


def test_arrow_replacement(mapping):
    result = interpret("pt -> f/u", mapping)
    assert "to" in result
    assert "->" not in result


def test_updown_arrows(mapping):
    result = interpret("↑ dose ↓ pain", mapping)
    assert "increase" in result
    assert "↑" not in result
    assert "decrease" in result
    assert "↓" not in result


def test_ampersand(mapping):
    result = interpret("bread & butter", mapping)
    assert "and" in result
    assert "&" not in result


def test_whitespace_normalization(mapping):
    result = interpret("  too   many   spaces  ", mapping)
    assert "  " not in result
    assert result == result.strip()


def test_punctuation_spacing(mapping):
    result = interpret("hello , world", mapping)
    assert "hello," in result


def test_empty_string(mapping):
    assert interpret("", mapping) == ""


def test_no_replacement_needed(mapping):
    text = "plain text with no shortcuts"
    assert interpret(text, mapping) == text


def test_longer_key_wins(mapping):
    """'-->' should expand before '->' (longer key first)."""
    result = interpret("--> result", mapping)
    # Should expand to "leads to result", not "- to result"
    assert "leads to" in result
