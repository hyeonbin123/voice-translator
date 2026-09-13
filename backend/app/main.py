import asyncio
import gc
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.routers import audio, auth, health, history, translate
from app.services.models import load_models

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.load_models:
        logger.info("Loading the models (about a minute on the first start)")
        # Loading holds the CPU and GPU for tens of seconds; keep it off the event loop.
        app.state.models = await asyncio.to_thread(load_models, settings)
        # faster-whisper runs a full gc.collect() on every decoded upload; with the models' millions of
        # objects that held the GIL ~0.4 s and stalled every other request (docs/experiments.md, T23).
        # Freezing moves everything alive now out of the collector's reach, so that collection stays short.
        if settings.gc_freeze:
            gc.freeze()
        logger.info("Models loaded")
    try:
        yield
    finally:
        if hasattr(app.state, "models"):
            del app.state.models


app = FastAPI(title="voice-translator API", version="0.1.0", lifespan=lifespan)

# Every route lives under /api, so the built frontend and the API can share one origin
# (the Vite dev server proxies /api to this app).
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(translate.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
