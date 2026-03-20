"""Text column concatenation for Format B (multi-column) inputs.

Format A: single note_text column — passed through unchanged.
Format B: multiple named columns — concatenated in configured order
          with section headers between non-null blocks.
"""

from __future__ import annotations
import pandas as pd


def detect_input_format(df: pd.DataFrame) -> str:
    """Return 'A' if note_text column is present, 'B' otherwise."""
    return "A" if "note_text" in df.columns else "B"


def concatenate_text_columns(df: pd.DataFrame, config: list[dict]) -> pd.DataFrame:
    """Assemble note_text from Format B multi-column inputs.

    For Format A inputs, returns df unchanged.

    Args:
        df: Input DataFrame.
        config: List of dicts with 'field' and 'header' keys.

    Returns:
        DataFrame with note_text column populated.
    """
    if "note_text" in df.columns:
        return df
    df = df.copy()
    df["note_text"] = df.apply(lambda row: _build_text(row, config), axis=1)
    return df


def _build_text(row: pd.Series, config: list[dict]) -> str:
    parts = []
    for col_conf in config:
        value = row.get(col_conf["field"])
        if value is not None and not pd.isna(value) and str(value).strip():
            parts.append(f"{col_conf['header']}\n{str(value).strip()}")
    return "\n\n".join(parts)
