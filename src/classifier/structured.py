"""ICD-10 rule engine for structured-only SSI classification.

Uses NHS ICD-10 (WHO fifth edition) codes as in HES, PEDW, and SMR.
These differ from ICD-10-CM (US): NHS uses T84.5 (four chars) where
ICD-10-CM uses T84.50-T84.54 (five chars). This module uses NHS codes.

Classification follows ECDC hierarchy: report only the deepest level.
"""
from __future__ import annotations

# NHS ICD-10 code → (ssi_type, depth_level)
# Higher depth = deeper SSI type (used for hierarchy enforcement)
_CODE_DEPTH_MAP = {
    "L02": ("superficial", 1),
    "L03": ("superficial", 1),
    "T81.4": ("deep", 2),
    "T84.6": ("deep", 2),
    "T84.5": ("organ_space", 3),
    "M00.8": ("organ_space", 3),
    "M00.9": ("organ_space", 3),
}

_DEPTH_TO_TYPE = {1: "superficial", 2: "deep", 3: "organ_space"}


class ICD10RuleEngine:
    """Deterministic SSI classifier using NHS ICD-10 codes.

    Implements ECDC reporting hierarchy: when multiple SSI types are
    signalled, only the deepest level is reported.
    """

    def classify(self, icd10_codes) -> dict:
        """Classify using pipe-separated NHS ICD-10 codes.

        Args:
            icd10_codes: Pipe-separated string of codes, or None.

        Returns:
            Dict with ssi_classification, confidence_zone, and null
            probability fields (not applicable for rule-based mode).
        """
        codes = _parse_codes(icd10_codes)
        max_depth = max(
            (_CODE_DEPTH_MAP[c][1] for c in codes if c in _CODE_DEPTH_MAP),
            default=0,
        )
        return {
            "ssi_classification": _DEPTH_TO_TYPE.get(max_depth, "none"),
            "confidence_zone": "rule_based",
            "p_none": None,
            "p_superficial": None,
            "p_deep": None,
            "p_organ_space": None,
        }


def _parse_codes(icd10_codes) -> list[str]:
    if not icd10_codes or isinstance(icd10_codes, float):
        return []
    return [c.strip() for c in str(icd10_codes).split("|") if c.strip()]
