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
