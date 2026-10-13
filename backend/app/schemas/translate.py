# [파일 역할] 번역 API의 요청 검증(언어 쌍, 글자 입력)과 응답 형태(번역 결과, 두 사람 대화 결과).
# [설계 이유] 빈 글자·NUL·같은 언어 쌍처럼 번역할 수 없는 입력을 모델에 닿기 전에 422로 거절한다(docs/api.md
# '실패했을 때'). 응답은 기록 항목(HistoryItem)에 번역 응답에만 있는 필드를 더하는 상속으로 만든다.
# [연결] routers/translate.py와 routers/live.py가 요청 검증에, services/pipeline.save와 라우터가 응답에 쓴다.

import unicodedata
from typing import Annotated

# unicodedata: 문자의 유니코드 범주(C=기타)를 보려고. Language: 'en'|'ko' 리터럴 타입을 서비스 계층과 공유해
# 허용 언어를 한곳에서 정한다.
from pydantic import BaseModel, StringConstraints, field_validator, model_validator

from app.schemas.history import HistoryItem
from app.services.interfaces import Language


# 출발·도착 언어 쌍. 둘 다 'en' 또는 'ko'여야 하고(Language), 서로 같으면 거절한다. 음성 번역(form 필드)과
# 동시통역 첫 메시지 검증도 이 클래스를 직접 만들어 같은 규칙을 쓴다.
class LanguagePair(BaseModel):
    source_lang: Language
    target_lang: Language

    @model_validator(mode="after")
    def different_languages(self):
        if self.source_lang == self.target_lang:
            raise ValueError("Source and target languages must differ")
        return self


# 글자 번역 요청. StringConstraints가 앞뒤 공백을 자른 뒤 1~500자(유니코드 문자 수)인지 본다. 자른 값을
# 번역하고 저장한다.
class TextRequest(LanguagePair):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]

    @field_validator("text")
    @classmethod
    def has_text_to_translate(cls, value: str) -> str:
        # Runs after the strip and length limits. PostgreSQL text cannot hold NUL, and "other" (C*) characters
        # alone (control; format such as zero-width space or BOM; private use; unassigned) leave the model
        # nothing: opus_preprocess turns each of them except a newline into a space, so only white space is
        # left. Punctuation, symbols and marks such as a lone combining accent reach the model and are still
        # translated.
        if "\x00" in value:
            raise ValueError("Text must not contain NUL characters")
        # 모든 글자가 공백이거나 C 범주(제어·서식·사용자 정의·미배정)면 번역할 글자가 없다고 보고 거절한다.
        # 문장부호 하나("?")만 있는 입력은 통과해 그대로 번역된다(docs/api.md POST /api/translate/text).
        if all(c.isspace() or unicodedata.category(c).startswith("C") for c in value):
            raise ValueError("Text has nothing to translate")
        return value


# 번역 응답 = 저장된 기록 항목 + tts_error. tts_error는 합성이 실패했거나 합성 모델이 없을 때의 이유이고,
# 기록(DB)에는 저장하지 않는다(docs/api.md).
class TranslationResponse(HistoryItem):
    tts_error: str | None


# 두 사람 대화 응답. 화면은 language_guessed가 true일 때 방향 바꾸기 단추를 보여 준다(docs/api.md, README
# 알려진 한계).
class DialogResponse(TranslationResponse):
    """A two-person conversation turn (T35): source_lang is the language that was taken as spoken."""

    language_confidence: float  # the detected language's share of the Korean and English probabilities
    language_guessed: bool  # too unsure: taken as the language opposite to the previous utterance
