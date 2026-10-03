"""Two people, one screen (T35): which language a turn is in (docs/experiments.md 10).

The HTTP dialog API and the dialog mode of the live WebSocket (T77) both decide with this rule.
"""

from app.services.interfaces import Language

OTHER: dict[Language, Language] = {"ko": "en", "en": "ko"}


def choose_language(
    detected: Language, confidence: float, previous: Language | None, threshold: float
) -> tuple[Language, bool]:
    """The turn's language and whether it was guessed. When the detection is unsure and an earlier turn is
    known, the conversation is taken to alternate: the turn is in the language opposite to that turn's."""
    if previous is not None and confidence < threshold:
        return OTHER[previous], True
    return detected, False
