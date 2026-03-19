"""Text cleaning for NHS clinical notes.

Handles encoding artefacts from NHS dictation software and document
management systems: null bytes, form feeds, carriage returns, and
excessive whitespace from copy-paste or OCR processes.
"""
import re


def clean_text(text) -> str:
    """Normalise clinical note text for NLP processing.

    Args:
        text: Raw note text, or None.

    Returns:
        Cleaned string. Returns empty string for None or empty input.
    """
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\x00", " ").replace("\x0c", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
