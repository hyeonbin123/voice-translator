"""Text normalization for speech recognition error rates (rules in docs/experiments.md)."""

import re

# Anything that isn't a letter, digit, whitespace or apostrophe. `\w` covers Hangul.
_PUNCTUATION = re.compile(r"[^\w\s']|_")


def normalize(text: str, language: str) -> str:
    """Lowercase and drop punctuation; Korean also drops every space and apostrophe.

    Korean CER is measured on characters without spaces, so a spacing difference
    ("할 수" vs "할수") doesn't count as an error. Apostrophes are kept only for English
    contractions ("it's"); in Korean they are quotation marks ('희망') and count as punctuation.
    English keeps single spaces for WER.
    """
    text = _PUNCTUATION.sub(" ", text.lower())
    if language == "ko":
        return "".join(text.replace("'", " ").split())
    return " ".join(text.split())
