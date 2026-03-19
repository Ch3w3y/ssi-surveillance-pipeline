from src.classifier.ecdc_gating import apply_ecdc_gating


def test_within_30d_all_classes_available():
    p = {"none": 0.1, "superficial": 0.6, "deep": 0.2, "organ_space": 0.1}
    r = apply_ecdc_gating(p, days_post_op=10)
    assert r["deep"] > 0 and r["organ_space"] > 0


def test_over_365d_zeroes_deep_and_organ_space():
    p = {"none": 0.1, "superficial": 0.2, "deep": 0.4, "organ_space": 0.3}
    r = apply_ecdc_gating(p, days_post_op=400)
    assert r["deep"] == 0.0 and r["organ_space"] == 0.0


def test_over_365d_renormalises():
    p = {"none": 0.1, "superficial": 0.2, "deep": 0.4, "organ_space": 0.3}
    r = apply_ecdc_gating(p, days_post_op=400)
    assert abs(sum(r.values()) - 1.0) < 1e-6


def test_none_days_unchanged():
    p = {"none": 0.4, "superficial": 0.3, "deep": 0.2, "organ_space": 0.1}
    assert apply_ecdc_gating(p, days_post_op=None) == p


def test_day_365_within_window():
    p = {"none": 0.1, "superficial": 0.2, "deep": 0.4, "organ_space": 0.3}
    r = apply_ecdc_gating(p, days_post_op=365)
    assert r["deep"] > 0 and r["organ_space"] > 0
