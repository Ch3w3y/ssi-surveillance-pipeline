"""Preprocessing orchestrator.

Runs: validation → concatenation → text cleaning → temporal
features → procedure metadata → ECDC window gating.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validator import validate_input
from .text_cleaner import clean_text
from .concatenator import concatenate_text_columns
from .temporal import compute_days_post_op, get_ecdc_window

_OPCS4_REF = None

# Resolve reference CSV relative to this file so tests run from any cwd.
_OPCS4_CSV = Path(__file__).resolve().parents[2] / "data" / "reference" / "opcs4_orthopaedic.csv"


def _load_opcs4() -> pd.DataFrame:
    global _OPCS4_REF
    if _OPCS4_REF is None:
        _OPCS4_REF = pd.read_csv(_OPCS4_CSV, dtype=str)
    return _OPCS4_REF


class Preprocessor:
    """Orchestrates all preprocessing steps for the SSI pipeline.

    Args:
        config: Pipeline configuration dict.
    """

    def __init__(self, config: dict):
        self.config = config
        self.text_column_config = config.get("text_columns", [])

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all preprocessing steps.

        Args:
            df: Raw input DataFrame.

        Returns:
            Preprocessed DataFrame with temporal features, procedure
            metadata, and bad-row flags applied.
        """
        df = validate_input(df)
        df = concatenate_text_columns(df, self.text_column_config)
        df = df.copy()
        df["note_text"] = df["note_text"].apply(clean_text)
        df = self._add_temporal(df)
        df = self._add_procedure_metadata(df)
        df = self._flag_outside_window(df)
        return df

    def _add_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["days_post_op"] = df.apply(
            lambda r: compute_days_post_op(r["operation_date"], r["note_date"]), axis=1
        )
        df["ecdc_window_flag"] = df["days_post_op"].apply(get_ecdc_window)
        return df

    def _add_procedure_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        ref = _load_opcs4().set_index("code")
        df["procedure_description"] = df["procedure_code"].map(
            lambda c: ref.loc[c, "description"] if c in ref.index else ""
        )
        df["procedure_type"] = df["procedure_code"].map(
            lambda c: ref.loc[c, "procedure_type"] if c in ref.index else ""
        )
        return df

    def _flag_outside_window(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df["ssi_classification"] == "") & (df["ecdc_window_flag"] == "outside_window")
        df.loc[mask, "ssi_classification"] = "outside_window"
        return df
