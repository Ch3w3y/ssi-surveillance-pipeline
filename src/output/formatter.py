"""Output line list and MDT review list formatting.

Merges NER and classifier results into the final tabular output.
No classification logic here — only column arrangement and filtering.
"""

from __future__ import annotations
import pandas as pd

OUTPUT_COLUMNS = [
    "patient_id",
    "episode_id",
    "operation_date",
    "note_date",
    "days_post_op",
    "procedure_code",
    "procedure_description",
    "procedure_type",
    "hospital_site",
    "processing_mode",
    "ssi_classification",
    "p_none",
    "p_superficial",
    "p_deep",
    "p_organ_space",
    "confidence_zone",
    "review_required",
    "extracted_entities",
    "entity_snippets",
    "ecdc_window_flag",
    "icd10_codes",
]


def format_linelist(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order columns for the full line list output."""
    df = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[OUTPUT_COLUMNS]


def filter_mdt_review(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to review-required episodes, sorted highest probability first.

    Adds blank reviewer_notes column for clinical team annotation.
    """
    review_df = df[df["review_required"].eq(True)].copy()
    prob_cols = [
        c
        for c in ["p_superficial", "p_deep", "p_organ_space"]
        if c in review_df.columns
    ]
    if prob_cols:
        review_df["_max_ssi"] = review_df[prob_cols].max(axis=1)
        review_df = review_df.sort_values("_max_ssi", ascending=False).drop(
            columns=["_max_ssi"]
        )
    review_df["reviewer_notes"] = ""
    return review_df.reset_index(drop=True)
