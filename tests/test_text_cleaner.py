# tests/test_text_cleaner.py
from src.preprocessing.text_cleaner import clean_text


def test_removes_excessive_whitespace():
    assert clean_text("wound  is   healing") == "wound is healing"


def test_strips_leading_trailing():
    assert clean_text("  wound healing  ") == "wound healing"


def test_normalises_carriage_returns():
    result = clean_text("wound\r\nhealing\rwell")
    assert result == "wound\nhealing\nwell"


def test_replaces_null_bytes_with_space():
    assert clean_text("wound\x00healing") == "wound healing"


def test_replaces_form_feed_with_space():
    assert clean_text("wound\x0chealing") == "wound healing"


def test_handles_empty_string():
    assert clean_text("") == ""


def test_handles_none():
    assert clean_text(None) == ""


def test_preserves_clinical_punctuation():
    result = clean_text("T 38.5. No signs of infection.")
    assert "38.5" in result and "No signs" in result
