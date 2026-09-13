import asyncio
import gc
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.routers import audio, auth, health, history, translate
from app.services.errors import describe
from app.services.models import load_models, warm_up

logger = logging.getLogger(__name__)


class WithoutExceptionMessages(logging.Filter):
    """Log an exception as its chain of types and place, never its messages or traceback (T49, T54).

    A library's message can quote the user's text or the translation. The app's own logs already use
    describe(); this catches the rest: an unexpected error that uvicorn logs as "Exception in ASGI
    application" (Starlette always re-raises it), and any app record logged with exc_info.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and record.exc_info[1] is not None:
            record.msg = f"{record.getMessage()}: {describe(record.exc_info[1])}"
            record.args = None
            record.exc_info = None
            record.exc_text = None
        return True


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
_uvicorn_errors = logging.getLogger("uvicorn.error")
if not any(isinstance(f, WithoutExceptionMessages) for f in _uvicorn_errors.filters):
    _uvicorn_errors.addFilter(_without_messages)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.load_models:
        logger.info("Loading the models (about a minute on the first start)")
        # Loading holds the CPU and GPU for tens of seconds; keep it off the event loop.
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
        models = getattr(app.state, "models", None)
        if models is not None:
            # Stop the typo correction model's background preparation (T39); close() does not wait.
            corrector = getattr(models, "corrector", None)
            if corrector is not None:
                corrector.close()
            del app.state.models


app = FastAPI(title="voice-translator API", version="0.1.0", lifespan=lifespan)

# Every route lives under /api, so the built frontend and the API can share one origin
# (the Vite dev server proxies /api to this app).
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(translate.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
