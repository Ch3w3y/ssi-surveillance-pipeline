# tests/test_entity_rules.py
from src.ner.entity_rules import ENTITY_PATTERNS, ENTITY_TYPES


def test_all_entity_types_defined():
    expected = {
        "WOUND_SIGN",
        "DISCHARGE",
        "WOUND_DISRUPTION",
        "ABSCESS",
        "FEVER",
        "ANTIBIOTIC",
        "WOUND_TREATMENT",
        "MICROBIOLOGY",
        "ANATOMICAL_DEPTH",
        "TEMPORAL",
    }
    assert set(ENTITY_TYPES) == expected


def test_purulent_in_discharge_patterns():
    discharge = [p for p in ENTITY_PATTERNS if p["label"] == "DISCHARGE"]
    flat = [t.get("LOWER", "") for p in discharge for t in p["pattern"]]
    assert "purulent" in flat


def test_antibiotic_patterns_exist():
    assert len([p for p in ENTITY_PATTERNS if p["label"] == "ANTIBIOTIC"]) >= 3


def test_dair_in_wound_treatment():
    wt = [p for p in ENTITY_PATTERNS if p["label"] == "WOUND_TREATMENT"]
    flat = [t.get("LOWER", "") for p in wt for t in p["pattern"]]
    assert "dair" in flat or "debridement" in flat


def test_periprosthetic_in_anatomical_depth():
    ad = [p for p in ENTITY_PATTERNS if p["label"] == "ANATOMICAL_DEPTH"]
    flat = [t.get("LOWER", "") for p in ad for t in p["pattern"]]
    assert "periprosthetic" in flat
