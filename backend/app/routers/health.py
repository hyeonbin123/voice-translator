"""Health check (docs/api.md): whether the API reaches its database, and which models it serves."""

# [파일 역할] 인증 없이 부르는 상태 확인 경로. DB 연결과 올라간 모델 이름, 오타 교정 준비 여부, 동시통역 가능
# 여부를 알려 준다.
# [설계 이유] Docker compose의 api 상태 검사가 이 경로를 쓰므로(docs/api.md), DB가 끊기면 503을 내 api가
# unhealthy가 되게 한다. 오류 이유는 응답에 넣지 않고 로그에만 종류와 위치로 남긴다.
# [연결] app/main.py가 /api 아래에 등록한다.

import asyncio
import logging

from fastapi import APIRouter, Response
from sqlalchemy import text

from app.dependencies import DbSession, Models
from app.services.errors import describe
from app.services.interfaces import LiveSpeechToText

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])
# DB 확인 제한 시간(초). DB가 응답하지 않을 때 상태 검사 자체가 오래 매달리지 않게 한다.
DB_TIMEOUT_S = 2.0


# 모델이 없으면(LOAD_MODELS=false, 합성 꺼짐 등) None, 있으면 model_name(예: "faster-whisper/large-v3-turbo").
def _name(model: object | None) -> str | None:
    return None if model is None else model.model_name


@router.get("/health")
async def health(response: Response, db: DbSession, models: Models) -> dict:
    # SELECT 1을 2초 안에 못 끝내면 어떤 예외든 DB를 쓸 수 없다고 보고 503을 낸다. 응답 본문은 그대로 만들어
    # 화면이 상태를 읽을 수 있게 한다.
    try:
        await asyncio.wait_for(db.execute(text("SELECT 1")), DB_TIMEOUT_S)
        database = "ok"
    except Exception as exc:  # noqa: BLE001 - whatever the failure, the database cannot be used
        # Types and place only, as everywhere in the log (T49): a driver's message can name the host.
        logger.warning("Health check could not reach the database: %s", describe(exc))
        database = "unavailable"
        response.status_code = 503
    corrector = models.corrector
    return {
        "status": "ok" if database == "ok" else "unavailable",
        "database": database,
        "models": {
            "speech_recognition": _name(models.stt),
            "translation": _name(models.translator),
            "speech_synthesis": _name(models.tts),
            "typo_correction": _name(corrector),
        },
        # The corrector prepares in the background and is not waited for (T37).
        "typo_correction_ready": corrector is not None and corrector.corrects("en"),
        # 동시통역 웹소켓은 자막용 인식(transcribe_live가 있는 LiveSpeechToText)과 번역 모델이 둘 다 있어야 쓸
        # 수 있다. runtime_checkable Protocol이라 isinstance로 메서드가 있는지 확인한다.
        "live": isinstance(models.stt, LiveSpeechToText) and models.translator is not None,
    }
