"""Text normalization for speech recognition error rates (rules in docs/experiments.md)."""

import re

# Anything that isn't a letter, digit, whitespace or apostrophe. `\w` covers Hangul.
_PUNCTUATION = re.compile(r"[^\w\s']|_")


def normalize(text: str, language: str) -> str:
    """Lowercase and drop punctuation; Korean also drops every space.

    Korean CER is measured on characters without spaces, so a spacing difference
    ("할 수" vs "할수") doesn't count as an error. English keeps single spaces for WER.
    """
    text = _PUNCTUATION.sub(" ", text.lower())
    if language == "ko":
        return "".join(text.split())
    return " ".join(text.split())
