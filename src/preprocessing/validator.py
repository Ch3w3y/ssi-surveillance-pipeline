"""Input schema validation and bad-row flagging for the SSI pipeline.

Validates required columns, date logic, procedure scope, and text
availability. Flags invalid rows in-place rather than dropping them.
"""
import pandas as pd

REQUIRED_COLUMNS = {
    "patient_id", "episode_id", "operation_date", "note_date", "procedure_code",
}

IN_SCOPE_CODES = {
    "W37", "W38", "W39", "W40", "W41",
    "W42", "W43", "W44", "W45", "W46", "W47",
}

TEXT_COLUMNS = [
    "note_text", "presenting_complaint", "clinical_findings",
    "diagnosis", "management_plan", "discharge_summary",
]


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input DataFrame and flag invalid rows.

    Raises ValueError if episode_id is absent — MDT review list is
    clinically unusable without episode identifiers.

    Args:
        df: Raw input DataFrame.

    Returns:
        DataFrame with added ssi_classification and validation_flag columns.
    """
    if "episode_id" not in df.columns:
        raise ValueError(
            "episode_id column is required but absent. "
            "MDT review list cannot be used without episode identifiers."
        )

    missing = (REQUIRED_COLUMNS - {"episode_id"}) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["ssi_classification"] = ""
    df["validation_flag"] = ""

    for idx in df.index:
        flag = _check_row(df.loc[idx])
        if flag:
            df.at[idx, "ssi_classification"] = flag
            df.at[idx, "validation_flag"] = flag

    return df


def _check_row(row: pd.Series) -> str:
    """Return flag string if row is invalid, empty string otherwise."""
    if pd.isna(row.get("operation_date")):
        return "missing_operation_date"
    if pd.isna(row.get("note_date")):
        return "missing_note_date"

    try:
        op_date = pd.to_datetime(row["operation_date"])
        note_date = pd.to_datetime(row["note_date"])
    except (ValueError, TypeError):
        return "invalid_dates"

    if note_date < op_date:
        return "invalid_dates"

    code = str(row.get("procedure_code", "")).strip().upper()
    if code not in IN_SCOPE_CODES:
        return "out_of_scope"

    present_text_cols = [c for c in TEXT_COLUMNS if c in row.index]
    # Only flag insufficient_data when at least one text column is present but all are null.
    # If zero text columns exist (structured_only mode), do not flag — ICD-10 engine handles it.
    if present_text_cols and all(pd.isna(row.get(c)) for c in present_text_cols):
        return "insufficient_data"

    return ""
