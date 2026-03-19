"""Smoke tests for structured_only mode (ICD-10 codes, no text)."""
import pandas as pd
import pytest

FIXTURE = "tests/smoke/fixtures/synthetic_notes.csv"


@pytest.fixture(scope="module")
def results():
    from src.pipeline.run import SSIPipeline
    df = pd.read_csv(FIXTURE, dtype=str).drop(columns=["note_text"])
    pipeline = SSIPipeline({
        "processing_mode": "structured_only",
        "text_columns": [],
        "thresholds": {"auto_negative": 0.85, "auto_positive": 0.85},
    })
    return pipeline.run(df)


def test_pipeline_completes(results):
    assert results is not None

def test_all_rows_present(results):
    assert len(results) == len(pd.read_csv(FIXTURE))

def test_mode_is_structured_only(results):
    assert "structured_only" in results["processing_mode"].values

def test_classified_rows_are_rule_based(results):
    flag_vals = {
        "out_of_scope", "outside_window", "missing_operation_date",
        "missing_note_date", "invalid_dates", "insufficient_data",
    }
    classified = results[~results["ssi_classification"].isin(flag_vals)]
    for zone in classified["confidence_zone"].dropna():
        assert zone == "rule_based"

def test_t84_5_detected_as_ssi(results):
    row = results[results["episode_id"] == "E004"]
    assert row.iloc[0]["ssi_classification"] in ("deep", "organ_space")
