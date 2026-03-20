# tests/test_validator.py
import pandas as pd
import pytest
from src.preprocessing.validator import validate_input


def make_row(**kwargs):
    defaults = {
        "patient_id": "P001",
        "episode_id": "E001",
        "operation_date": "2025-01-15",
        "note_date": "2025-01-25",
        "procedure_code": "W38",
        "note_text": "Wound healing well.",
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def test_valid_row_passes():
    result = validate_input(make_row())
    assert result.loc[0, "ssi_classification"] == ""


def test_missing_episode_id_raises():
    df = make_row().drop(columns=["episode_id"])
    with pytest.raises(ValueError, match="episode_id"):
        validate_input(df)


def test_missing_operation_date_flagged():
    result = validate_input(make_row(operation_date=None))
    assert result.loc[0, "ssi_classification"] == "missing_operation_date"


def test_missing_note_date_flagged():
    result = validate_input(make_row(note_date=None))
    assert result.loc[0, "ssi_classification"] == "missing_note_date"


def test_note_before_operation_flagged():
    result = validate_input(
        make_row(operation_date="2025-02-01", note_date="2025-01-01")
    )
    assert result.loc[0, "ssi_classification"] == "invalid_dates"


def test_out_of_scope_code_flagged():
    result = validate_input(make_row(procedure_code="W99"))
    assert result.loc[0, "ssi_classification"] == "out_of_scope"


def test_null_text_flagged():
    df = make_row()
    df["note_text"] = None
    result = validate_input(df)
    assert result.loc[0, "ssi_classification"] == "insufficient_data"


def test_same_day_note_valid():
    result = validate_input(
        make_row(operation_date="2025-01-15", note_date="2025-01-15")
    )
    assert result.loc[0, "ssi_classification"] != "invalid_dates"


def test_validation_flag_matches_ssi_classification_on_invalid_row():
    result = validate_input(make_row(procedure_code="W99"))
    assert result.loc[0, "validation_flag"] == "out_of_scope"


def test_no_text_columns_not_flagged_as_insufficient():
    """Structured-only mode: no text columns present should not flag insufficient_data."""
    df = pd.DataFrame(
        [
            {
                "patient_id": "P001",
                "episode_id": "E001",
                "operation_date": "2025-01-15",
                "note_date": "2025-01-25",
                "procedure_code": "W38",
                "icd10_codes": "T81.4",
            }
        ]
    )
    result = validate_input(df)
    assert result.loc[0, "ssi_classification"] == ""
