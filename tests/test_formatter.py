import pandas as pd
from src.output.formatter import format_linelist, filter_mdt_review

REQUIRED_COLS = [
    "patient_id",
    "episode_id",
    "ssi_classification",
    "p_none",
    "p_superficial",
    "p_deep",
    "p_organ_space",
    "confidence_zone",
    "review_required",
    "extracted_entities",
    "entity_snippets",
]


def make_row(**kwargs):
    d = {
        "patient_id": "P001",
        "episode_id": "E001",
        "operation_date": "2025-01-15",
        "note_date": "2025-01-25",
        "days_post_op": 10,
        "procedure_code": "W38",
        "procedure_description": "Hip replacement",
        "procedure_type": "hip_total",
        "hospital_site": "SITE_A",
        "processing_mode": "text_only",
        "ssi_classification": "superficial",
        "p_none": 0.05,
        "p_superficial": 0.88,
        "p_deep": 0.05,
        "p_organ_space": 0.02,
        "confidence_zone": "auto_positive",
        "review_required": False,
        "extracted_entities": "DISCHARGE:affirmed",
        "entity_snippets": '"pus"',
        "ecdc_window_flag": "within_30d",
        "icd10_codes": "Z96.6",
        "validation_flag": "",
    }
    d.update(kwargs)
    return pd.DataFrame([d])


def test_required_columns_present():
    result = format_linelist(make_row())
    for col in REQUIRED_COLS:
        assert col in result.columns


def test_mdt_filters_to_review_required_only():
    df = pd.concat(
        [
            make_row(review_required=True, episode_id="E001"),
            make_row(review_required=False, episode_id="E002"),
        ],
        ignore_index=True,
    )
    result = filter_mdt_review(df)
    assert len(result) == 1 and result.loc[0, "episode_id"] == "E001"


def test_mdt_has_reviewer_notes_column():
    df = make_row(review_required=True)
    assert "reviewer_notes" in filter_mdt_review(df).columns


def test_mdt_sorted_by_max_ssi_prob():
    df = pd.concat(
        [
            make_row(review_required=True, episode_id="E001", p_superficial=0.60),
            make_row(review_required=True, episode_id="E002", p_superficial=0.75),
        ],
        ignore_index=True,
    )
    result = filter_mdt_review(df)
    assert result.iloc[0]["episode_id"] == "E002"
