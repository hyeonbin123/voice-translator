"""상태 확인(docs/api.md): DB에 닿는지, 어떤 모델을 서비스하는지, 모델 호출이 연이어 실패하는지 알린다."""

# [파일 역할] 인증 없이 부르는 상태 확인 경로. DB 연결과 올라간 모델 이름, 오타 교정 준비 여부, 동시통역 가능
# 여부를 알려 주고, 모델을 올린 서버(LOAD_MODELS=true)는 모델 호출의 연속 실패 수와 models_ok도 알려 준다.
# [설계 이유] Docker compose의 api 상태 검사가 이 경로를 쓰므로(docs/api.md), DB가 끊기거나 모델 호출이
# 한도만큼 연이어 실패하면 503을 내 api가 unhealthy가 되게 한다. 오류 이유는 응답에 넣지 않고 로그에만
# 남긴다. 모델 필드를 LOAD_MODELS=true일 때만 넣는 것은, 모델 없이 도는 테스트·CI 서버의 응답과 동작을 이
# 필드가 생기기 전과 똑같이 두려는 것이다(그 서버의 번역 요청은 늘 503이라 셀 모델 호출도 없다).
# [연결] app/main.py가 /api 아래에 등록한다. 연속 실패 수는 services/model_health.py가 세고, 한도는
# config.Settings.model_failure_limit(MODEL_FAILURE_LIMIT)다.

import asyncio
import logging

from fastapi import APIRouter, Response
from sqlalchemy import text

from app.config import get_settings
from app.dependencies import DbSession, Models
from app.services import model_health
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
    except Exception as exc:  # noqa: BLE001 - 실패 이유가 무엇이든 DB를 쓸 수 없다
        # 로그의 다른 곳처럼 예외 종류와 위치만 남긴다. 드라이버의 메시지에는 DB 호스트 이름이 들어갈 수 있다.
        logger.warning("Health check could not reach the database: %s", describe(exc))
        database = "unavailable"
        response.status_code = 503
    corrector = models.corrector
    body = {
        "status": "ok" if database == "ok" else "unavailable",
        "database": database,
        "models": {
            "speech_recognition": _name(models.stt),
            "translation": _name(models.translator),
            "speech_synthesis": _name(models.tts),
            "typo_correction": _name(corrector),
        },
        # 오타 교정기는 백그라운드에서 준비되고 서버 시작이 그것을 기다리지 않으므로, 준비됐는지 따로 알린다.
        "typo_correction_ready": corrector is not None and corrector.corrects("en"),
        # 동시통역 웹소켓은 자막용 인식(transcribe_live가 있는 LiveSpeechToText)과 번역 모델이 둘 다 있어야 쓸
        # 수 있다. runtime_checkable Protocol이라 isinstance로 메서드가 있는지 확인한다.
        "live": isinstance(models.stt, LiveSpeechToText) and models.translator is not None,
    }
    settings = get_settings()
    if settings.load_models:
        # models_ok: 반드시 있어야 하는 두 모델(인식·번역 — 없으면 어떤 번역도 성공할 수 없다,
        # services/models.py)이 올라가 있고, 엔진 오류로 연이어 실패한 모델 호출이 한도보다 적은지.
        # 합성·교정은 선택이라 보지 않는다. 거짓이면 DB 503과 같이 503·"unavailable"로 답하고 이유는 로그에만
        # 남긴다(연속 실패 수는 응답에도 있다). 번역 요청은 그동안에도 모델로 가므로, 원인이 풀려 한 번
        # 성공하면 다음 확인부터 200으로 돌아온다.
        failures = model_health.MODEL_HEALTH.consecutive_failures
        loaded = models.stt is not None and models.translator is not None
        models_ok = loaded and failures < settings.model_failure_limit
        body["models_ok"] = models_ok
        body["consecutive_model_failures"] = failures
        if not models_ok:
            logger.warning(
                "Health check: models loaded=%s, model calls failed %d times in a row (limit %d)",
                loaded,
                failures,
                settings.model_failure_limit,
            )
            body["status"] = "unavailable"
            response.status_code = 503
    return body
