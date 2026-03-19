"""Post-softmax ECDC surveillance window enforcement.

All in-scope procedures (W37-W47) involve implants, so implant=True always.
The 1-year window applies to deep incisional and organ/space SSI.
Gating is unconditional — cannot be overridden by model confidence.
"""
from __future__ import annotations


def apply_ecdc_gating(probs: dict, days_post_op: int | None) -> dict:
    """Zero and renormalise probabilities outside the ECDC window.

    Args:
        probs: Dict mapping 'none','superficial','deep','organ_space' to floats.
        days_post_op: Integer days since operation, or None.

    Returns:
        Gated and renormalised probability dict. Unchanged if days_post_op is None.
    """
    probs = dict(probs)
    if days_post_op is None:
        return probs
    if days_post_op > 365:
        probs["deep"] = 0.0
        probs["organ_space"] = 0.0
    total = sum(probs.values())
    if total == 0:
        probs["none"] = 1.0
        return probs
    return {k: v / total for k, v in probs.items()}
