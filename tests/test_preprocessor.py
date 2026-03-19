# tests/test_preprocessor.py
import pandas as pd
from src.preprocessing.preprocessor import Preprocessor

CONFIG = {"text_columns": [{"field": "clinical_findings", "header": "CLINICAL FINDINGS"}]}


def make_row(**kwargs):
    defaults = {
        "patient_id": "P001", "episode_id": "E001",
        "operation_date": "2025-01-15", "note_date": "2025-01-25",
        "procedure_code": "W38", "note_text": "Wound erythema noted.",
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def test_days_post_op_computed():
    result = Preprocessor(CONFIG).run(make_row())
    assert result.loc[0, "days_post_op"] == 10


def test_ecdc_window_added():
    result = Preprocessor(CONFIG).run(make_row())
    assert result.loc[0, "ecdc_window_flag"] == "within_30d"


def test_outside_window_flagged():
    result = Preprocessor(CONFIG).run(make_row(note_date="2026-05-01"))
    assert result.loc[0, "ssi_classification"] == "outside_window"


def test_text_cleaned():
    result = Preprocessor(CONFIG).run(make_row(note_text="wound\x00 healing"))
    assert "\x00" not in result.loc[0, "note_text"]


def test_procedure_type_added():
    result = Preprocessor(CONFIG).run(make_row(procedure_code="W38"))
    assert result.loc[0, "procedure_type"] == "hip_total"
