# tests/test_ner_pipeline.py
import pytest
from src.ner.pipeline import NERPipeline


@pytest.fixture(scope="module")
def ner():
    return NERPipeline()


def test_affirmed_discharge_detected(ner):
    entities = ner.run("There is purulent discharge from the wound.")
    assertions = {e["label"]: e["assertion"] for e in entities}
    assert assertions.get("DISCHARGE") == "affirmed"


def test_negated_antibiotic(ner):
    entities = ner.run("No antibiotics required. Wound healing well.")
    assertions = {e["label"]: e["assertion"] for e in entities}
    assert assertions.get("ANTIBIOTIC") == "negated"


def test_clean_note_has_no_affirmed_entities(ner):
    entities = ner.run("Satisfactory post-operative recovery. No concerns.")
    affirmed = [e for e in entities if e["assertion"] == "affirmed"]
    assert len(affirmed) == 0
