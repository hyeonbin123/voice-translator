"""Live subtitles over a WebSocket (T34, T56; the contract is in docs/api.md).

While a person speaks, the browser streams the utterance's audio. Now and then the server recognizes all of
it again, translates the result and sends both, with the start that two results in a row agree on marked
stable. When the speaker pauses, the clip conversation mode would upload is complete: the server prepares
the final translation right then, and saves and sends it once the browser confirms the utterance ended.
"""

import asyncio
import json
import logging
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

import jwt
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.core.security import decode_token_with_expiry
from app.dependencies import Models, Sessions, StoredAudio
from app.models import User
from app.schemas.translate import LanguagePair
from app.services import pipeline
from app.services.audio_store import AudioStore
from app.services.errors import describe
from app.services.inference import run_live_model, run_model
from app.services.interfaces import (
    Language,
    LiveSpeechToText,
    ModelError,
    NoSpeechError,
    Translator,
    UndecodableAudioError,
)
from app.services.live import letters, stable_length, wav_from_pcm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/translate", tags=["translate"])

MAX_UTTERANCE_BYTES = 30 * 16_000 * 2  # the HTTP API's 30 s limit, in 16 kHz 16-bit audio
# The browser starts an utterance 64 ms into speech and sends the 192 ms before it as well (T33): this much
# audio is there when an utterance starts.
PREROLL_S = 0.256
UNAUTHORIZED, INVALID, UNAVAILABLE = 4401, 4422, 4503


@dataclass(frozen=True)
class LiveOptions:
    interval_s: float  # least time between the starts of two updates of one utterance
    start_timeout_s: float = 10.0
    update_thread: bool = False  # updates on a thread of their own (LIVE_UPDATE_THREAD, T61)


def get_live_options() -> LiveOptions:
    settings = get_settings()
    return LiveOptions(interval_s=settings.live_update_ms / 1000, update_thread=settings.live_update_thread)


@dataclass(eq=False)
class Utterance:
    id: int
    started_at: float
    pcm: bytearray = field(default_factory=bytearray)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    paused: bool = False
    too_long: bool = False
    closed: bool = False
    updater: asyncio.Task | None = None
    final: asyncio.Task | None = None  # prepared at a pause, dropped when speech resumes
    recognized: str | None = None  # the last update's text
    shown: tuple[str, int] | None = None  # the text and stable length last sent


def failure(exc: BaseException) -> str:
    """The HTTP API's detail for a translation that failed; logs it without any message (T49)."""
    if isinstance(exc, UndecodableAudioError):
        logger.warning("Undecodable audio: %s", describe(exc))
        return "Audio could not be decoded"
    if isinstance(exc, NoSpeechError):
        logger.warning("No speech recognized: %s", describe(exc))
        return "No speech was recognized"
    if isinstance(exc, pipeline.AudioTooLongError):
        return "Audio is longer than 30 seconds"
    logger.error("Live translation failed: %s", describe(exc))
    return "Translation service is unavailable"


class LiveConnection:
    def __init__(
        self,
        websocket: WebSocket,
        models: pipeline.PipelineModels,
        stt: LiveSpeechToText,
        translator: Translator,
        store: AudioStore,
        sessions: async_sessionmaker[AsyncSession],
        user_id: UUID,
        pair: LanguagePair,
        expires_at: datetime,
        options: LiveOptions,
    ) -> None:
        self.websocket = websocket
        self.models = models
        self.stt = stt
        self.translator = translator
        self.store = store
        self.sessions = sessions
        self.user_id = user_id
        self.source: Language = pair.source_lang
        self.target: Language = pair.target_lang
        self.expires_at = expires_at
        self.options = options
        self.run_update = run_live_model if options.update_thread else run_model
        self.current: Utterance | None = None
        self.tasks: set[asyncio.Task] = set()
        self.sending = asyncio.Lock()

    def spawn(self, coroutine: Coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        # A dropped final's failure is never awaited; retrieve it so asyncio does not report it.
        task.add_done_callback(lambda done: done.cancelled() or done.exception())
        return task

    async def send(self, message: dict) -> None:
        async with self.sending:
            try:
                await self.websocket.send_json(message)
            except (WebSocketDisconnect, RuntimeError, OSError):
                pass  # the browser has gone; the receive loop ends the connection

    async def run(self) -> None:
        try:
            while True:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if datetime.now(UTC) >= self.expires_at:
                    await self.websocket.close(UNAUTHORIZED)  # the browser reconnects with a new token
                    return
                if message.get("bytes") is not None:
                    self.audio(message["bytes"])
                elif not self.control(message.get("text")):
                    await self.websocket.close(INVALID)
                    return
        finally:
            self.drop(self.current)
            for task in list(self.tasks):
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)

    def control(self, text: str | None) -> bool:
        """Act on one control message; False when it breaks the contract."""
        try:
            message = json.loads(text or "")
            kind, number = message["type"], message["id"]
        except (ValueError, TypeError, KeyError):
            return False
        if type(number) is not int or kind not in ("utterance", "pause", "resume", "end", "cancel"):
            return False
        if kind == "utterance":
            self.drop(self.current)
            self.current = Utterance(number, started_at=asyncio.get_running_loop().time())
            self.current.updater = self.spawn(self.update(self.current))
            return True
        utterance = self.current
        if utterance is None or utterance.id != number:
            return True  # about an utterance that has already ended or been dropped
        if kind == "pause":
            self.pause(utterance)
        elif kind == "resume":
            self.resume(utterance)
        elif kind == "end":
            self.current = None
            self.end(utterance)
        else:
            self.current = None
            self.drop(utterance)
        return True

    def audio(self, data: bytes) -> None:
        utterance = self.current
        if utterance is None or utterance.too_long:
            return  # audio outside an utterance is dropped
        if len(utterance.pcm) + len(data) > MAX_UTTERANCE_BYTES:
            utterance.too_long = True
            utterance.pcm = bytearray()
            for task in (utterance.updater, utterance.final):
                if task is not None:
                    task.cancel()
            return
        utterance.pcm += data
        utterance.changed.set()

    def pause(self, utterance: Utterance) -> None:
        """The clip is complete (192 ms of quiet): prepare its final translation now."""
        if utterance.paused or utterance.too_long:
            return
        utterance.paused = True
        utterance.final = self.spawn(self.prepare(bytes(utterance.pcm)))

    def resume(self, utterance: Utterance) -> None:
        if not utterance.paused:
            return
        utterance.paused = False
        if utterance.final is not None:
            utterance.final.cancel()  # a model call already running still finishes, but nothing after it
            utterance.final = None
        utterance.changed.set()

    def end(self, utterance: Utterance) -> None:
        utterance.closed = True
        if utterance.updater is not None:
            utterance.updater.cancel()
        if utterance.too_long:
            self.spawn(
                self.send({"type": "error", "id": utterance.id, "detail": "Audio is longer than 30 seconds"})
            )
            return
        final = utterance.final or self.spawn(self.prepare(bytes(utterance.pcm)))  # no pause: all audio
        self.spawn(self.finish(utterance.id, final))

    def drop(self, utterance: Utterance | None) -> None:
        if utterance is None:
            return
        utterance.closed = True
        utterance.changed.set()
        for task in (utterance.updater, utterance.final):
            if task is not None:
                task.cancel()

    async def prepare(self, pcm: bytes) -> pipeline.Prepared:
        # The same WAV file and pipeline as conversation mode's upload, so the same result.
        return await pipeline.prepare(
            models=self.models, source=self.source, target=self.target, audio=wav_from_pcm(pcm)
        )

    async def finish(self, number: int, final: asyncio.Task) -> None:
        try:
            prepared = await final
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one utterance failed; the connection goes on
            await self.send({"type": "error", "id": number, "detail": failure(exc)})
            return
        try:
            async with self.sessions() as db:
                result = await pipeline.save(db=db, store=self.store, user_id=self.user_id, prepared=prepared)
        except Exception as exc:  # noqa: BLE001
            logger.error("Saving a live translation failed: %s", describe(exc))
            await self.send({"type": "error", "id": number, "detail": "The translation could not be saved"})
            return
        await self.send({"type": "final", "id": number, "result": result.model_dump(mode="json")})

    async def update(self, utterance: Utterance) -> None:
        """Recognize the utterance so far again and again, one update at a time (docs/experiments.md 8)."""
        loop = asyncio.get_running_loop()
        next_at = utterance.started_at + max(0.0, self.options.interval_s - PREROLL_S)
        seen = 0  # bytes of audio the last update recognized
        while not utterance.closed:
            if (delay := next_at - loop.time()) > 0:
                await asyncio.sleep(delay)
                continue
            utterance.changed.clear()
            if utterance.paused or utterance.too_long or len(utterance.pcm) <= seen:
                await utterance.changed.wait()
                continue
            pcm = bytes(utterance.pcm)
            seen, next_at = len(pcm), loop.time() + self.options.interval_s
            try:
                text = await self.run_update(self.stt.transcribe_live, pcm, self.source)
            except ModelError as exc:
                logger.warning("Live recognition failed: %s", describe(exc))
                continue
            stable = stable_length(utterance.recognized, text)
            utterance.recognized = text
            if (
                utterance.closed
                or (text, stable) == utterance.shown
                or (not text and utterance.shown is None)
            ):
                continue
            changed = utterance.shown is None or utterance.shown[0] != text
            utterance.shown = (text, stable)
            await self.send({"type": "source", "id": utterance.id, "text": text, "stable": stable})
            if not changed or not letters(text) or utterance.paused:
                continue  # at a pause the final translation is on its way
            try:
                translation = await self.run_update(self.translator.translate, text, self.source, self.target)
            except ModelError as exc:
                logger.warning("Live translation failed: %s", describe(exc))
                continue
            if utterance.closed:
                return
            # Nothing of a live translation is stable: as the source grows, its start changes too (16-20%
            # of what would be shown dark changed later, docs/experiments.md 8).
            await self.send({"type": "translation", "id": utterance.id, "text": translation, "stable": 0})


@router.websocket("/live")
async def live(
    websocket: WebSocket,
    models: Models,
    store: StoredAudio,
    sessions: Sessions,
    options: LiveOptions = Depends(get_live_options),
) -> None:
    await websocket.accept()
    # A browser cannot put headers on a WebSocket, and a token in the address ends up in logs: the token
    # comes in the first message.
    try:
        start = json.loads(await asyncio.wait_for(websocket.receive_text(), options.start_timeout_s))
        if start["type"] != "start" or not isinstance(start["token"], str):
            raise ValueError("not a start message")
        pair = LanguagePair(source_lang=start["source_lang"], target_lang=start["target_lang"])
    except WebSocketDisconnect:
        return
    except (TimeoutError, ValueError, TypeError, KeyError, ValidationError):
        await websocket.close(INVALID)
        return
    try:
        user_id, expires_at = decode_token_with_expiry(start["token"], "access")
    except jwt.InvalidTokenError:
        await websocket.close(UNAUTHORIZED)
        return
    async with sessions() as db:
        if await db.get(User, user_id) is None:
            await websocket.close(UNAUTHORIZED)
            return
    if not isinstance(models.stt, LiveSpeechToText) or models.translator is None:
        await websocket.close(UNAVAILABLE)
        return
    await websocket.send_json({"type": "ready"})
    await LiveConnection(
        websocket, models, models.stt, models.translator, store, sessions, user_id, pair, expires_at, options
    ).run()
