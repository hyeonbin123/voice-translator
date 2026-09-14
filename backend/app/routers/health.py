"""Health check (docs/api.md): whether the API reaches its database, and which models it serves."""

import asyncio
import logging

from fastapi import APIRouter, Response
from sqlalchemy import text

from app.dependencies import DbSession, Models
from app.services.errors import describe
from app.services.interfaces import LiveSpeechToText

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])
DB_TIMEOUT_S = 2.0


def _name(model: object | None) -> str | None:
    return None if model is None else model.model_name


@router.get("/health")
async def health(response: Response, db: DbSession, models: Models) -> dict:
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
        "live": isinstance(models.stt, LiveSpeechToText) and models.translator is not None,
    }
