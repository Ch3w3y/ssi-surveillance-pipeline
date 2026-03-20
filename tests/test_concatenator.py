# tests/test_concatenator.py
import pandas as pd
from src.preprocessing.concatenator import concatenate_text_columns, detect_input_format

CONFIG = [
    {"field": "presenting_complaint", "header": "PRESENTING COMPLAINT"},
    {"field": "clinical_findings", "header": "CLINICAL FINDINGS"},
]


def test_format_a_detected():
    df = pd.DataFrame([{"note_text": "some text"}])
    assert detect_input_format(df) == "A"


def test_format_b_detected():
    df = pd.DataFrame([{"clinical_findings": "wound ok"}])
    assert detect_input_format(df) == "B"


def test_format_a_unchanged():
    df = pd.DataFrame([{"note_text": "wound healing well"}])
    result = concatenate_text_columns(df, CONFIG)
    assert result.loc[0, "note_text"] == "wound healing well"


def test_format_b_concatenates_with_headers():
    df = pd.DataFrame(
        [
            {
                "presenting_complaint": "wound pain",
                "clinical_findings": "erythema",
                "management_plan": None,
            }
        ]
    )
    result = concatenate_text_columns(df, CONFIG)
    text = result.loc[0, "note_text"]
    assert "PRESENTING COMPLAINT" in text and "wound pain" in text
    assert "erythema" in text


def test_all_null_produces_empty():
    df = pd.DataFrame([{"presenting_complaint": None, "clinical_findings": None}])
    result = concatenate_text_columns(df, CONFIG)
    assert result.loc[0, "note_text"] == ""
