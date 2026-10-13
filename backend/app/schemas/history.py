# [파일 역할] 번역 기록 한 건(HistoryItem)과 기록 목록 페이지(HistoryPage)의 응답 형태.
# [설계 이유] 응답에 들어갈 필드를 명시해 DB 내부 값(user_id, 음성 파일 경로)이 나가지 않게 한다. 언어·mode를
# Literal로 두어 응답 스키마(Swagger)에 허용 값이 드러난다.
# [연결] routers/history.py가 응답에, services/pipeline.save가 번역 응답(TranslationResponse)의 바탕으로 쓰고,
# schemas/translate.py의 응답 모델이 이를 상속한다.

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


# 기록 한 건. from_attributes=True라 ORM Translation 객체의 속성을 그대로 읽는다. audio_id는
# Translation.audio_id 속성(음성 행의 id 또는 None)에서 온다.
# 모델 이름·시간(ms)이 None이면 그 단계를 실행하지 않았다는 뜻이다(글자 입력의 STT, 실패한 TTS).
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


# 목록 응답. total은 페이지 크기와 상관없는 본인 기록 전체 개수다.
class HistoryPage(BaseModel):
    items: list[HistoryItem]
    total: int
