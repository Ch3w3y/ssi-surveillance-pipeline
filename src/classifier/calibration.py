"""Temperature scaling calibration and confidence zone assignment.

Temperature scaling divides logits by scalar T before softmax.
T is fit on a held-out calibration split (10% of training data).

Zone assignment uses priority-order catch-all logic:
  Priority 1: auto_negative  — P(none) >= threshold
  Priority 2: auto_positive  — max(SSI classes) >= threshold
  Priority 3: review_required — catch-all for all other rows

The catch-all ensures every row is assigned a zone regardless of
probability distribution shape, including low-confidence rows.
"""
from __future__ import annotations
import numpy as np

DEFAULT_THRESHOLDS = {"auto_negative": 0.85, "auto_positive": 0.85}
CLASS_ORDER = ["none", "superficial", "deep", "organ_space"]


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling and softmax to raw model logits.

    Args:
        logits: 1D array of 4 raw model output values.
        temperature: Scalar divisor. Must be > 0.

    Returns:
        1D array of calibrated probabilities summing to 1.0.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = logits / temperature
    exp = np.exp(scaled - np.max(scaled))
    return exp / exp.sum()


def assign_confidence_zone(probs: dict, thresholds: dict) -> tuple[str, bool]:
    """Assign triage zone using priority-order catch-all logic.

    Args:
        probs: Dict mapping class names to calibrated probabilities.
        thresholds: Dict with 'auto_negative' and 'auto_positive' keys.

    Returns:
        (confidence_zone string, review_required boolean).
    """
    if probs["none"] >= thresholds["auto_negative"]:
        return "auto_negative", False
    ssi_max = max(v for k, v in probs.items() if k != "none")
    if ssi_max >= thresholds["auto_positive"]:
        return "auto_positive", False
    return "review_required", True
