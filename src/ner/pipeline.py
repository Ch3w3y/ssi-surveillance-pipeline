"""MedSpaCy NER pipeline with ConText assertion detection.

Builds a pipeline from scispaCy en_core_sci_sm base model, adds the
MedSpaCy ConText component, then applies custom entity patterns.

Architecture: scispaCy (en_core_sci_sm) → base tokenisation + NER →
EntityRuler patterns (before ner, overwriting) → MedSpaCy ConText →
assertion status → entity list filtered to custom labels.

The PyRuSH sentencizer is disabled because it conflicts with the
en_core_sci_sm parser component (both set sentence boundaries).

Instantiating NERPipeline loads the language model (~2-5 seconds).
Instantiate once and reuse for batch processing.
"""

from __future__ import annotations
import medspacy
from .entity_rules import ENTITY_PATTERNS, ENTITY_TYPES

_CUSTOM_LABELS = set(ENTITY_TYPES)


class NERPipeline:
    """MedSpaCy pipeline for SSI entity extraction with assertion detection."""

    def __init__(self):
        # Disable medspacy_pyrush: it conflicts with en_core_sci_sm's parser
        # (both attempt to set sentence boundaries, raising ValueError [E043]).
        self.nlp = medspacy.load("en_core_sci_sm", disable=["medspacy_pyrush"])
        # Place EntityRuler before the scispaCy NER so our domain patterns
        # take precedence; overwrite_ents=True drops any overlapping scispaCy
        # ENTITY spans in favour of our labelled spans.
        ruler = self.nlp.add_pipe(
            "entity_ruler", before="ner", config={"overwrite_ents": True}
        )
        ruler.add_patterns(ENTITY_PATTERNS)

    def run(self, text: str) -> list[dict]:
        """Extract entities and assertion status from a clinical note.

        Only entities matching our custom SSI label set are returned;
        generic scispaCy ENTITY spans are excluded.

        Args:
            text: Cleaned note text.

        Returns:
            List of dicts with keys: text, label, assertion,
            start_char, end_char.
        """
        if not text or not text.strip():
            return []
        doc = self.nlp(text)
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "assertion": _get_assertion(ent),
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            }
            for ent in doc.ents
            if ent.label_ in _CUSTOM_LABELS
        ]


def _get_assertion(ent) -> str:
    """Extract ConText assertion from a spaCy span.

    MedSpaCy ConText sets is_negated, is_uncertain, is_historical,
    is_hypothetical as span extension attributes.
    """
    if getattr(ent._, "is_negated", False):
        return "negated"
    if getattr(ent._, "is_uncertain", False):
        return "uncertain"
    if getattr(ent._, "is_historical", False):
        return "historical"
    if getattr(ent._, "is_hypothetical", False):
        return "hypothetical"
    return "affirmed"
