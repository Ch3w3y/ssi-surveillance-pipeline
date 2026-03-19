import pandas as pd
from src.output.summary import generate_summary


def make_linelist():
    return pd.DataFrame([
        {"ssi_classification": "none", "processing_mode": "text_only", "review_required": False},
        {"ssi_classification": "superficial", "processing_mode": "text_only", "review_required": False},
        {"ssi_classification": "deep", "processing_mode": "text_only", "review_required": True},
        {"ssi_classification": "out_of_scope", "processing_mode": "text_only", "review_required": False},
    ])


def test_summary_contains_mode():
    assert "text_only" in generate_summary(make_linelist(), "2026-03-19", {})


def test_summary_contains_percentage():
    assert "%" in generate_summary(make_linelist(), "2026-03-19", {})


def test_summary_is_non_empty_string():
    result = generate_summary(make_linelist(), "2026-03-19", {})
    assert isinstance(result, str) and len(result) > 0
