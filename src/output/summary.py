"""Run summary text generation for surveillance reporting."""
from __future__ import annotations
import math
import pandas as pd

SSI_TYPES = ["superficial", "deep", "organ_space"]
FLAG_TYPES = [
    "out_of_scope", "missing_operation_date", "missing_note_date",
    "invalid_dates", "insufficient_data", "outside_window",
]


def generate_summary(df: pd.DataFrame, run_date: str, thresholds: dict) -> str:
    """Generate plain-text surveillance run summary.

    Args:
        df: Full line list DataFrame.
        run_date: ISO date string.
        thresholds: Threshold config dict.

    Returns:
        Multi-line string for a .txt report file.
    """
    total = len(df)
    mode = df["processing_mode"].mode()[0] if total > 0 else "unknown"
    valid = df[~df["ssi_classification"].isin(FLAG_TYPES)]
    n_valid = len(valid)
    counts = {t: int((valid["ssi_classification"] == t).sum()) for t in ["none"] + SSI_TYPES}
    n_ssi = sum(counts[t] for t in SSI_TYPES)
    n_review = int((df["review_required"] == True).sum())
    rate = n_ssi / n_valid if n_valid > 0 else 0
    lo, hi = _wilson_ci(n_ssi, n_valid)

    lines = [
        "SSI Surveillance Pipeline — Run Summary",
        "=" * 40,
        f"Run date           : {run_date}",
        f"Processing mode    : {mode}",
        f"Episodes processed : {total:,}",
    ]
    for flag in FLAG_TYPES:
        n = int((df["ssi_classification"] == flag).sum())
        if n:
            lines.append(f"  {flag:<26}: {n:>5,}")

    lines += [
        "",
        "Classifications (ECDC — valid episodes only):",
        f"  None             : {counts['none']:>5,}  ({100*counts['none']/n_valid:.1f}%)" if n_valid else "  None             :     0",
        f"  Superficial SSI  : {counts['superficial']:>5,}  ({100*counts['superficial']/n_valid:.2f}%)" if n_valid else "  Superficial SSI  :     0",
        f"  Deep SSI         : {counts['deep']:>5,}  ({100*counts['deep']/n_valid:.2f}%)" if n_valid else "  Deep SSI         :     0",
        f"  Organ/Space SSI  : {counts['organ_space']:>5,}  ({100*counts['organ_space']/n_valid:.2f}%)" if n_valid else "  Organ/Space SSI  :     0",
        f"  Overall SSI rate :  {100*rate:.2f}% (95% CI: {100*lo:.2f}-{100*hi:.2f}%)" if n_valid else "  Overall SSI rate :  N/A (no valid episodes)",
        "",
        f"Review-required (borderline): {n_review:,} episodes",
        "",
        "Thresholds applied:",
        f"  auto_negative : P(none) >= {thresholds.get('auto_negative', 0.85)}",
        f"  auto_positive : P(SSI)  >= {thresholds.get('auto_positive', 0.85)}",
    ]
    return "\n".join(lines)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)
