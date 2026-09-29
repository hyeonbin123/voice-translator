import unicodedata
from typing import Annotated

from pydantic import BaseModel, StringConstraints, field_validator, model_validator

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

    @field_validator("text")
    @classmethod
    def has_text_to_translate(cls, value: str) -> str:
        # Runs after the strip and length limits. PostgreSQL text cannot hold NUL, and format or control
        # characters alone (zero-width space, BOM) leave the model nothing: opus_preprocess turns them into
        # spaces. Punctuation and symbols alone are still translated.
        if "\x00" in value:
            raise ValueError("Text must not contain NUL characters")
        if all(c.isspace() or unicodedata.category(c).startswith("C") for c in value):
            raise ValueError("Text has nothing to translate")
        return value


class TranslationResponse(HistoryItem):
    tts_error: str | None


class DialogResponse(TranslationResponse):
    """A two-person conversation turn (T35): source_lang is the language that was taken as spoken."""

    language_confidence: float  # the detected language's share of the Korean and English probabilities
    language_guessed: bool  # too unsure: taken as the language opposite to the previous utterance
