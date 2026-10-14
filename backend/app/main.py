# [파일 역할] FastAPI 앱의 진입점이다(`uvicorn app.main:app`). 로그 설정, 앱 수명 주기(lifespan: 모델
# 적재·예열·정리), 라우터 등록을 맡는다.
# [설계 이유] (1) 로그에 사용자 글·번역문이 새지 않게 예외 메시지를 지우는 로그 필터를 앱 로거와 uvicorn 오류
# 로거에 붙인다. (2) 모델 적재는 수십 초 걸리므로 이벤트 루프 밖 스레드에서 하고, 끝난 뒤에만 요청을 받는다.
# (3) 모델을 올린 직후 gc.freeze()로 번역 중 다른 요청이 막히던 문제를 없앤다(docs/experiments.md 5절).
# [흐름] 로그 필터·핸들러 설정 → lifespan(LOAD_MODELS면 load_models → warm_up → gc.freeze, 종료 때 오타 교정
# 준비 스레드 정리) → 라우터를 /api 아래에 등록.

import asyncio
import gc
import logging
from contextlib import asynccontextmanager

# 프로젝트 내부: 설정, 각 API 라우터, 예외를 메시지 없이 요약하는 describe, 모델 적재·예열 함수.
from fastapi import FastAPI

from app.config import get_settings
from app.routers import audio, auth, health, history, live, translate
from app.services.errors import describe
from app.services.models import load_models, warm_up

logger = logging.getLogger(__name__)


class WithoutExceptionMessages(logging.Filter):
    """예외를 종류의 사슬과 위치로만 기록하고 메시지나 traceback은 남기지 않는 로그 필터.

    로그 레코드에 예외 정보가 붙어 있으면 그 예외를 describe() 요약(예외 종류의 사슬과 위치)으로 바꾸고
    traceback·메시지를 지운다. 라이브러리 예외 메시지에는 사용자가 입력한 글이나 번역문이 그대로 들어갈 수
    있기 때문이다. 앱 자체 로그는 이미 describe()를 쓰고, 이 필터는 나머지를 잡는다: uvicorn이 "Exception
    in ASGI application"으로 남기는 예상하지 못한 오류(Starlette는 늘 다시 raise함)와, exc_info를 붙여
    남긴 앱 로그 레코드다. tests/test_log_privacy.py가 확인한다.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # getMessage()로 원래 로그 문장(인자 채운 것)만 남기고 뒤에 요약을 붙인다. args를 None으로 해 포매터가
        # 다시 % 치환을 하지 않게 하고, exc_info·exc_text를 지워 핸들러가 traceback을 찍지 않게 한다. 레코드는
        # 버리지 않는다(return True).
        if record.exc_info and record.exc_info[1] is not None:
            record.msg = f"{record.getMessage()}: {describe(record.exc_info[1])}"
            record.args = None
            record.exc_info = None
            record.exc_text = None
        return True


# 모듈을 불러올 때 로그 설정을 한 번 한다. 테스트처럼 같은 모듈을 여러 번 불러도 핸들러·필터가 중복으로 붙지
# 않게 '이미 있으면 건너뛰기'로 짰다.
_without_messages = WithoutExceptionMessages()
# uvicorn은 자기 로거만 설정하므로, 핸들러를 따로 붙이지 않으면 앱의 INFO 줄(모델 적재·예열 시간)이
# 버려졌다. 전파(propagate)는 켠 채로 두어 pytest의 caplog도 계속 이 줄을 받는다.
_app_logger = logging.getLogger("app")
if not _app_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
    _app_logger.addHandler(_handler)
for _handler in _app_logger.handlers:
    # 필터를 로거가 아니라 핸들러에 붙여, app.* 하위 로거에서 올라온 레코드에도 적용되게 한다.
    if not any(isinstance(f, WithoutExceptionMessages) for f in _handler.filters):
        _handler.addFilter(_without_messages)
_app_logger.setLevel(logging.INFO)
# 예상하지 못한 예외는 uvicorn이 'uvicorn.error' 로거로 traceback과 함께 남기므로 이 로거에도 같은 필터를
# 붙인다.
_uvicorn_errors = logging.getLogger("uvicorn.error")
if not any(isinstance(f, WithoutExceptionMessages) for f in _uvicorn_errors.filters):
    _uvicorn_errors.addFilter(_without_messages)


# 앱 수명 주기. yield 앞은 서버가 요청을 받기 전에 한 번, finally는 종료 때 한 번 실행된다. 그래서
# LOAD_MODELS=true면 모델 적재·예열이 끝난 뒤에야 /api/health가 응답하고, Docker compose의 상태 검사도 그때
# 통과한다(docs/api.md 상태 확인).
# LOAD_MODELS=false(테스트·CI)면 아무것도 올리지 않아 app.state.models가 없고, dependencies.get_models가 빈
# 묶음을 준다.
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.load_models:
        logger.info("Loading the models (about a minute on the first start)")
        # 적재는 CPU와 GPU를 수십 초 동안 쥐므로 이벤트 루프 밖 스레드에서 한다. app.state에 둔 모델
        # 묶음을 dependencies.get_models가 모든 요청에 넘겨준다.
        app.state.models = await asyncio.to_thread(load_models, settings)
        # 각 모델의 첫 호출은 추가로 더 올린다(MeloTTS의 BERT 등). 그 비용을 첫 요청 대신 여기서 치른다.
        # 아래 freeze보다 먼저 해야 예열이 만든 객체도 함께 얼려진다.
        if settings.warm_up:
            await asyncio.to_thread(warm_up, app.state.models)
        # faster-whisper는 디코딩한 업로드마다 전체 gc.collect()를 부른다. 모델의 수백만 객체 때문에 이것이
        # GIL을 약 0.4초 쥐어 다른 모든 요청이 멈췄다(docs/experiments.md 5절). freeze는 지금 살아 있는
        # 객체를 모두 컬렉터가 훑는 대상에서 빼므로 그 수집이 짧게 끝난다.
        if settings.gc_freeze:
            gc.freeze()
        logger.info("Models loaded")
    try:
        yield
    finally:
        # 종료 정리: 오타 교정기의 백그라운드 준비 스레드에 멈추라고 알리고(close는 기다리지 않음), 모델
        # 참조를 놓는다.
        models = getattr(app.state, "models", None)
        if models is not None:
            corrector = getattr(models, "corrector", None)
            if corrector is not None:
                corrector.close()
            del app.state.models


# FastAPI 앱 객체. 위의 lifespan으로 모델 적재·예열·정리를 건다.
app = FastAPI(title="voice-translator API", version="0.1.0", lifespan=lifespan)

# 모든 라우터를 /api 아래에 두어 빌드한 화면과 API가 한 출처(origin)를 함께 쓸 수 있게 한다. Docker에서는
# nginx가 화면과 `/api` 프록시를 함께 맡고(README 배포), 개발 중에는 Vite 개발 서버가 /api를 이
# 앱(8000번)으로 넘긴다.
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(translate.router, prefix="/api")
app.include_router(live.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
