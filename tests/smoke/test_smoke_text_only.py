"""Smoke tests for text_only mode.

Validates pipeline structure and correctness — NOT classification accuracy.
Requires HuggingFace model download on first run.
Add @pytest.mark.requires_model to skip in CI.
"""
import pandas as pd
import pytest

FIXTURE = "tests/smoke/fixtures/synthetic_notes.csv"
VALID_LABELS = {
    "none", "superficial", "deep", "organ_space", "out_of_scope",
    "outside_window", "missing_operation_date", "missing_note_date",
    "invalid_dates", "insufficient_data",
}


@pytest.fixture(scope="module")
@pytest.mark.requires_model
def results():
    from src.pipeline.run import SSIPipeline
    pipeline = SSIPipeline({
        "model": "Simonlee711/Clinical_ModernBERT",
        "processing_mode": "text_only",
        "text_columns": [],
        "thresholds": {"auto_negative": 0.85, "auto_positive": 0.85},
    })
    return pipeline.run(pd.read_csv(FIXTURE, dtype=str))


@pytest.mark.requires_model
def test_all_rows_in_output(results):
    assert len(results) == len(pd.read_csv(FIXTURE))

@pytest.mark.requires_model
def test_required_columns(results):
    for col in ["ssi_classification", "p_none", "confidence_zone", "review_required"]:
        assert col in results.columns

@pytest.mark.requires_model
def test_valid_labels(results):
    for label in results["ssi_classification"].dropna():
        assert label in VALID_LABELS, f"Unexpected: {label}"

@pytest.mark.requires_model
def test_probs_sum_to_one(results):
    text_rows = results[results["processing_mode"] == "text_only"]
    for _, row in text_rows.iterrows():
        if pd.notna(row["p_none"]):
            total = row["p_none"] + row["p_superficial"] + row["p_deep"] + row["p_organ_space"]
            assert abs(total - 1.0) < 1e-4

@pytest.mark.requires_model
def test_out_of_scope_flagged(results):
    oos = results[results["episode_id"] == "E007"]
    assert oos.iloc[0]["ssi_classification"] == "out_of_scope"

@pytest.mark.requires_model
def test_missing_date_flagged(results):
    row = results[results["episode_id"] == "E008"]
    assert row.iloc[0]["ssi_classification"] == "missing_operation_date"

@pytest.mark.requires_model
def test_outside_window_flagged(results):
    row = results[results["episode_id"] == "E010"]
    assert row.iloc[0]["ssi_classification"] == "outside_window"
