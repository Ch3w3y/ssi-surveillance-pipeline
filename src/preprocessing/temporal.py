"""Temporal feature computation for ECDC SSI surveillance windows.

All in-scope procedures (W37-W47) involve prosthetic implants so the ECDC
surveillance window for deep incisional and organ/space SSI extends to
1 year (365 days) post-operatively.
"""
from __future__ import annotations
import pandas as pd


def compute_days_post_op(operation_date, note_date) -> int | None:
    """Compute integer calendar days from operation to note.

    Args:
        operation_date: Operation date as string (YYYY-MM-DD) or None.
        note_date: Note date as string (YYYY-MM-DD) or None.

    Returns:
        Non-negative integer days, or None if dates are invalid or
        note_date precedes operation_date.
    """
    if operation_date is None or note_date is None:
        return None
    if pd.isna(operation_date) or pd.isna(note_date):
        return None
    try:
        op = pd.to_datetime(operation_date)
        note = pd.to_datetime(note_date)
    except (ValueError, TypeError):
        return None
    days = (note - op).days
    return days if days >= 0 else None


def get_ecdc_window(days_post_op: int | None) -> str:
    """Map days_post_op to ECDC surveillance window label.

    Args:
        days_post_op: Integer days post-op, or None.

    Returns:
        One of: 'within_30d', 'within_1yr', 'outside_window', 'unknown'.
    """
    if days_post_op is None:
        return "unknown"
    if days_post_op <= 30:
        return "within_30d"
    if days_post_op <= 365:
        return "within_1yr"
    return "outside_window"
