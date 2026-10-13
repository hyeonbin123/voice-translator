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


# 로그 레코드에 예외 정보가 붙어 있으면 그 예외를 'describe()' 요약(예외 종류의 사슬과 위치)으로 바꾸고
# traceback·메시지를 지우는 필터다. 라이브러리 예외 메시지에 사용자 입력이 그대로 들어갈 수 있어서
# 둔다(tests/test_log_privacy.py가 확인).
class WithoutExceptionMessages(logging.Filter):
    """Log an exception as its chain of types and place, never its messages or traceback (T49, T54).

    A library's message can quote the user's text or the translation. The app's own logs already use
    describe(); this catches the rest: an unexpected error that uvicorn logs as "Exception in ASGI
    application" (Starlette always re-raises it), and any app record logged with exc_info.
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
# uvicorn configures only its own loggers, so without a handler the app's INFO lines (model loading and
# warm-up times) were dropped (T30). Propagation stays on, so pytest's caplog still receives them.
_app_logger = logging.getLogger("app")
if not _app_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
    _app_logger.addHandler(_handler)
for _handler in _app_logger.handlers:
    # On the handler, so it also covers records from app.* child loggers.
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
        # Loading holds the CPU and GPU for tens of seconds; keep it off the event loop.
        # app.state에 둔 모델 묶음을 dependencies.get_models가 모든 요청에 넘겨준다.
        app.state.models = await asyncio.to_thread(load_models, settings)
        # The first call of each model loads more (MeloTTS's BERT among others); pay for it here (T25).
        # Before the freeze below, so what it creates is frozen too.
        if settings.warm_up:
            await asyncio.to_thread(warm_up, app.state.models)
        # faster-whisper runs a full gc.collect() on every decoded upload; with the models' millions of
        # objects that held the GIL ~0.4 s and stalled every other request (docs/experiments.md, T23).
        # Freezing moves everything alive now out of the collector's reach, so that collection stays short.
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
            # Stop the typo correction model's background preparation (T39); close() does not wait.
            corrector = getattr(models, "corrector", None)
            if corrector is not None:
                corrector.close()
            del app.state.models


# 모든 라우터를 /api 아래에 둔다. Docker에서는 nginx가 화면과 `/api` 프록시를 함께 맡고(README 배포), 개발
# 중에는 Vite 개발 서버가 /api를 이 앱(8000번)으로 넘긴다.
app = FastAPI(title="voice-translator API", version="0.1.0", lifespan=lifespan)

# Every route lives under /api, so the built frontend and the API can share one origin
# (the Vite dev server proxies /api to this app).
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(translate.router, prefix="/api")
app.include_router(live.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
