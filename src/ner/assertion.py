"""Assertion status formatting for NER output columns."""


def format_entity_output(entities: list[dict]) -> tuple[str, str]:
    """Format entity list into pipe-separated output column strings.

    Args:
        entities: List of dicts with keys 'text', 'label', 'assertion'.

    Returns:
        Tuple of (extracted_entities, entity_snippets). Both empty strings
        if entities is empty.
    """
    if not entities:
        return "", ""
    pairs = "|".join(f"{e['label']}:{e['assertion']}" for e in entities)
    snippets = "|".join(f'"{e["text"]}"' for e in entities)
    return pairs, snippets
