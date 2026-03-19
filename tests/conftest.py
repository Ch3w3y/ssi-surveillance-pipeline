# tests/conftest.py
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df():
    return pd.DataFrame([
        {
            "patient_id": "P001", "episode_id": "E001",
            "operation_date": "2025-01-15", "note_date": "2025-01-25",
            "procedure_code": "W38",
            "note_text": "Wound healing satisfactorily. No signs of infection.",
        },
        {
            "patient_id": "P002", "episode_id": "E002",
            "operation_date": "2025-02-01", "note_date": "2025-02-12",
            "procedure_code": "W43",
            "note_text": "Purulent discharge noted at incision site. Erythema present.",
        },
        {
            "patient_id": "P003", "episode_id": "E003",
            "operation_date": "2025-03-01", "note_date": "2025-03-10",
            "procedure_code": "W99",
            "note_text": "Wound clean and dry.",
        },
    ])
