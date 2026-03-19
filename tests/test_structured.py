from src.classifier.structured import ICD10RuleEngine

engine = ICD10RuleEngine()


def test_t81_4_is_ssi():
    assert engine.classify("T81.4|Z96.6")["ssi_classification"] != "none"


def test_t84_5_is_deep_or_organ_space():
    assert engine.classify("T84.5")["ssi_classification"] in ("deep", "organ_space")


def test_l02_is_superficial():
    assert engine.classify("L02|Z96.6")["ssi_classification"] == "superficial"


def test_no_ssi_codes_is_none():
    assert engine.classify("Z96.6|M16.1")["ssi_classification"] == "none"


def test_empty_is_none():
    assert engine.classify("")["ssi_classification"] == "none"


def test_none_is_none():
    assert engine.classify(None)["ssi_classification"] == "none"


def test_confidence_zone_is_rule_based():
    assert engine.classify("T81.4")["confidence_zone"] == "rule_based"


def test_hierarchy_deep_over_superficial():
    result = engine.classify("T84.5|L02")
    assert result["ssi_classification"] in ("deep", "organ_space")
