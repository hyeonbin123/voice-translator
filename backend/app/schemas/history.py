from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    mode: Literal["text", "speech"]
    source_lang: Literal["en", "ko"]
    target_lang: Literal["en", "ko"]
    source_text: str
    translated_text: str
    stt_model: str | None
    mt_model: str
    tts_model: str | None
    stt_ms: int | None
    mt_ms: int
    tts_ms: int | None
    audio_id: UUID | None
    created_at: datetime


class HistoryPage(BaseModel):
    items: list[HistoryItem]
    total: int
