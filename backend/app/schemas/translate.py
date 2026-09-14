from typing import Annotated

from pydantic import BaseModel, StringConstraints, model_validator

from app.schemas.history import HistoryItem
from app.services.interfaces import Language


class LanguagePair(BaseModel):
    source_lang: Language
    target_lang: Language

    @model_validator(mode="after")
    def different_languages(self):
        if self.source_lang == self.target_lang:
            raise ValueError("Source and target languages must differ")
        return self


class TextRequest(LanguagePair):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class TranslationResponse(HistoryItem):
    tts_error: str | None


class DialogResponse(TranslationResponse):
    """A two-person conversation turn (T35): source_lang is the language that was taken as spoken."""

    language_confidence: float  # the detected language's share of the Korean and English probabilities
    language_guessed: bool  # too unsure: taken as the language opposite to the previous utterance
