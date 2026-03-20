# tests/test_assertion.py
from src.ner.assertion import format_entity_output


def test_affirmed_entity_formatted():
    entities = [
        {"text": "purulent discharge", "label": "DISCHARGE", "assertion": "affirmed"}
    ]
    pairs, snippets = format_entity_output(entities)
    assert "DISCHARGE:affirmed" in pairs
    assert "purulent discharge" in snippets


def test_negated_entity_formatted():
    entities = [{"text": "antibiotics", "label": "ANTIBIOTIC", "assertion": "negated"}]
    pairs, _ = format_entity_output(entities)
    assert "ANTIBIOTIC:negated" in pairs


def test_empty_returns_empty_strings():
    pairs, snippets = format_entity_output([])
    assert pairs == "" and snippets == ""


def test_multiple_pipe_separated():
    entities = [
        {"text": "pus", "label": "DISCHARGE", "assertion": "affirmed"},
        {"text": "antibiotics", "label": "ANTIBIOTIC", "assertion": "negated"},
    ]
    pairs, snippets = format_entity_output(entities)
    assert "|" in pairs and "|" in snippets
