# tests/test_temporal.py
from src.preprocessing.temporal import compute_days_post_op, get_ecdc_window


def test_standard_days():
    assert compute_days_post_op("2025-01-15", "2025-01-25") == 10


def test_same_day_is_zero():
    assert compute_days_post_op("2025-01-15", "2025-01-15") == 0


def test_negative_returns_none():
    assert compute_days_post_op("2025-02-01", "2025-01-01") is None


def test_none_dates_return_none():
    assert compute_days_post_op(None, "2025-01-25") is None
    assert compute_days_post_op("2025-01-15", None) is None
    assert compute_days_post_op(None, None) is None


def test_ecdc_window_within_30d():
    assert get_ecdc_window(10) == "within_30d"
    assert get_ecdc_window(30) == "within_30d"
    assert get_ecdc_window(0) == "within_30d"


def test_ecdc_window_within_1yr():
    assert get_ecdc_window(31) == "within_1yr"
    assert get_ecdc_window(365) == "within_1yr"


def test_ecdc_window_outside():
    assert get_ecdc_window(366) == "outside_window"


def test_ecdc_window_none():
    assert get_ecdc_window(None) == "unknown"
