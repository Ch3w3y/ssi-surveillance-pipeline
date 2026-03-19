"""NER entity type definitions and spaCy EntityRuler patterns for SSI surveillance.

Patterns use the spaCy Matcher format: each entry has a 'label' and 'pattern'
key. The 'pattern' is a list of token attribute dicts (LOWER for case-insensitive
matching, IS_ALPHA for word tokens).
"""

ENTITY_TYPES = [
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
]

ENTITY_PATTERNS = [
    # WOUND_SIGN
    {"label": "WOUND_SIGN", "pattern": [{"LOWER": "redness"}]},
    {"label": "WOUND_SIGN", "pattern": [{"LOWER": "erythema"}]},
    {"label": "WOUND_SIGN", "pattern": [{"LOWER": "swelling"}]},
    {"label": "WOUND_SIGN", "pattern": [{"LOWER": "warmth"}]},
    {"label": "WOUND_SIGN", "pattern": [{"LOWER": "tenderness"}]},
    {"label": "WOUND_SIGN", "pattern": [{"LOWER": "heat"}]},
    {"label": "WOUND_SIGN", "pattern": [{"LOWER": "induration"}]},
    {"label": "WOUND_SIGN", "pattern": [{"LOWER": "cellulitis"}]},
    # DISCHARGE
    {"label": "DISCHARGE", "pattern": [{"LOWER": "pus"}]},
    {"label": "DISCHARGE", "pattern": [{"LOWER": "purulent"}]},
    {"label": "DISCHARGE", "pattern": [{"LOWER": "seropurulent"}]},
    {"label": "DISCHARGE", "pattern": [{"LOWER": "exudate"}]},
    {"label": "DISCHARGE", "pattern": [{"LOWER": "purulent"}, {"LOWER": "discharge"}]},
    {"label": "DISCHARGE", "pattern": [{"LOWER": "wound"}, {"LOWER": "discharge"}]},
    {"label": "DISCHARGE", "pattern": [{"LOWER": "offensive"}, {"LOWER": "discharge"}]},
    # WOUND_DISRUPTION
    {"label": "WOUND_DISRUPTION", "pattern": [{"LOWER": "dehiscence"}]},
    {"label": "WOUND_DISRUPTION", "pattern": [{"LOWER": "dehisced"}]},
    {"label": "WOUND_DISRUPTION", "pattern": [{"LOWER": "gaping"}]},
    {"label": "WOUND_DISRUPTION", "pattern": [{"LOWER": "wound"}, {"LOWER": "breakdown"}]},
    {"label": "WOUND_DISRUPTION", "pattern": [{"LOWER": "wound"}, {"LOWER": "opened"}]},
    {"label": "WOUND_DISRUPTION", "pattern": [{"LOWER": "wound"}, {"LOWER": "separation"}]},
    # ABSCESS
    {"label": "ABSCESS", "pattern": [{"LOWER": "abscess"}]},
    {"label": "ABSCESS", "pattern": [{"LOWER": "collection"}]},
    {"label": "ABSCESS", "pattern": [{"LOWER": "haematoma"}]},
    {"label": "ABSCESS", "pattern": [{"LOWER": "hematoma"}]},
    {"label": "ABSCESS", "pattern": [{"LOWER": "fluid"}, {"LOWER": "collection"}]},
    # FEVER
    {"label": "FEVER", "pattern": [{"LOWER": "fever"}]},
    {"label": "FEVER", "pattern": [{"LOWER": "pyrexia"}]},
    {"label": "FEVER", "pattern": [{"LOWER": "febrile"}]},
    {"label": "FEVER", "pattern": [{"LOWER": "high"}, {"LOWER": "temperature"}]},
    {"label": "FEVER", "pattern": [{"LOWER": "temperature"}, {"IS_DIGIT": True}]},
    # ANTIBIOTIC
    {"label": "ANTIBIOTIC", "pattern": [{"LOWER": "antibiotic"}]},
    {"label": "ANTIBIOTIC", "pattern": [{"LOWER": "antibiotics"}]},
    {"label": "ANTIBIOTIC", "pattern": [{"LOWER": "antimicrobial"}]},
    {"label": "ANTIBIOTIC", "pattern": [{"LOWER": "flucloxacillin"}]},
    {"label": "ANTIBIOTIC", "pattern": [{"LOWER": "co-amoxiclav"}]},
    {"label": "ANTIBIOTIC", "pattern": [{"LOWER": "vancomycin"}]},
    {"label": "ANTIBIOTIC", "pattern": [{"LOWER": "ciprofloxacin"}]},
    {"label": "ANTIBIOTIC", "pattern": [{"LOWER": "clindamycin"}]},
    {"label": "ANTIBIOTIC", "pattern": [{"LOWER": "iv"}, {"LOWER": "antibiotics"}]},
    # WOUND_TREATMENT
    {"label": "WOUND_TREATMENT", "pattern": [{"LOWER": "debridement"}]},
    {"label": "WOUND_TREATMENT", "pattern": [{"LOWER": "washout"}]},
    {"label": "WOUND_TREATMENT", "pattern": [{"LOWER": "dair"}]},
    {"label": "WOUND_TREATMENT", "pattern": [{"LOWER": "reoperation"}]},
    {"label": "WOUND_TREATMENT", "pattern": [{"LOWER": "re-operation"}]},
    {"label": "WOUND_TREATMENT", "pattern": [{"LOWER": "wound"}, {"LOWER": "exploration"}]},
    {"label": "WOUND_TREATMENT", "pattern": [{"LOWER": "irrigation"}]},
    {"label": "WOUND_TREATMENT", "pattern": [{"LOWER": "wound"}, {"LOWER": "revision"}]},
    # MICROBIOLOGY
    {"label": "MICROBIOLOGY", "pattern": [{"LOWER": "wound"}, {"LOWER": "swab"}]},
    {"label": "MICROBIOLOGY", "pattern": [{"LOWER": "culture"}, {"LOWER": "positive"}]},
    {"label": "MICROBIOLOGY", "pattern": [{"LOWER": "culture"}, {"LOWER": "negative"}]},
    {"label": "MICROBIOLOGY", "pattern": [{"LOWER": "mrsa"}]},
    {"label": "MICROBIOLOGY", "pattern": [{"LOWER": "staph"}]},
    {"label": "MICROBIOLOGY", "pattern": [{"LOWER": "staphylococcus"}]},
    {"label": "MICROBIOLOGY", "pattern": [{"LOWER": "organism"}, {"LOWER": "isolated"}]},
    {"label": "MICROBIOLOGY", "pattern": [{"LOWER": "bacteraemia"}]},
    # ANATOMICAL_DEPTH
    {"label": "ANATOMICAL_DEPTH", "pattern": [{"LOWER": "superficial"}]},
    {"label": "ANATOMICAL_DEPTH", "pattern": [{"LOWER": "deep"}]},
    {"label": "ANATOMICAL_DEPTH", "pattern": [{"LOWER": "fascia"}]},
    {"label": "ANATOMICAL_DEPTH", "pattern": [{"LOWER": "fascial"}]},
    {"label": "ANATOMICAL_DEPTH", "pattern": [{"LOWER": "periprosthetic"}]},
    {"label": "ANATOMICAL_DEPTH", "pattern": [{"LOWER": "prosthesis"}]},
    {"label": "ANATOMICAL_DEPTH", "pattern": [{"LOWER": "joint"}, {"LOWER": "space"}]},
    {"label": "ANATOMICAL_DEPTH", "pattern": [{"LOWER": "intra-articular"}]},
    # TEMPORAL
    {"label": "TEMPORAL", "pattern": [{"LOWER": "post-operative"}]},
    {"label": "TEMPORAL", "pattern": [{"LOWER": "postoperative"}]},
    {"label": "TEMPORAL", "pattern": [{"LOWER": "post-op"}]},
    {"label": "TEMPORAL", "pattern": [{"IS_DIGIT": True}, {"LOWER": "days"}, {"LOWER": "post"}]},
    {"label": "TEMPORAL", "pattern": [{"IS_DIGIT": True}, {"LOWER": "weeks"}, {"LOWER": "post"}]},
    {"label": "TEMPORAL", "pattern": [{"IS_DIGIT": True}, {"LOWER": "months"}, {"LOWER": "post"}]},
]
