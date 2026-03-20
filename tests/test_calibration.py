import numpy as np
from src.classifier.calibration import (
    apply_temperature,
    assign_confidence_zone,
    DEFAULT_THRESHOLDS,
)


def test_temperature_1_unchanged():
    logits = np.array([2.0, 1.0, 0.5, 0.3])
    assert abs(sum(apply_temperature(logits, 1.0)) - 1.0) < 1e-6


def test_auto_negative_assigned():
    p = {"none": 0.90, "superficial": 0.05, "deep": 0.03, "organ_space": 0.02}
    zone, review = assign_confidence_zone(p, DEFAULT_THRESHOLDS)
    assert zone == "auto_negative" and review is False


def test_auto_positive_assigned():
    p = {"none": 0.05, "superficial": 0.88, "deep": 0.05, "organ_space": 0.02}
    zone, review = assign_confidence_zone(p, DEFAULT_THRESHOLDS)
    assert zone == "auto_positive" and review is False


def test_catch_all_review_required():
    # P(none)=0.70 < 0.85, max(SSI)=0.20 < 0.85
    p = {"none": 0.70, "superficial": 0.20, "deep": 0.07, "organ_space": 0.03}
    zone, review = assign_confidence_zone(p, DEFAULT_THRESHOLDS)
    assert zone == "review_required" and review is True


def test_mid_range_review_required():
    p = {"none": 0.30, "superficial": 0.50, "deep": 0.15, "organ_space": 0.05}
    zone, review = assign_confidence_zone(p, DEFAULT_THRESHOLDS)
    assert zone == "review_required" and review is True
