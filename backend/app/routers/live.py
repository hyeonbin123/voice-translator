"""Live subtitles over a WebSocket (T34, T56; the contract is in docs/api.md).

While a person speaks, the browser streams the utterance's audio. Now and then the server recognizes all of
it again, translates the result and sends both, with the start that two results in a row agree on marked
stable. When the speaker pauses, the clip conversation mode would upload is complete: the server prepares
the final translation right then, and saves and sends it once the browser confirms the utterance ended.

Conversation and dialog modes (T77) use the same connection without the subtitles: only the prepare at a
pause and the save at the end, so their result is the HTTP speech or dialog API's for the same clip. Dialog
turns are saved in the order they ended, since an unsure turn's language depends on the turn before.
"""

import asyncio
import json
import logging
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

import jwt
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.core.security import decode_token_with_expiry
from app.dependencies import Models, Sessions, StoredAudio
from app.models import User
from app.schemas.translate import DialogResponse, LanguagePair
from app.services import pipeline
from app.services.audio_store import AudioStore
from app.services.dialog import OTHER, choose_language
from app.services.errors import describe
from app.services.inference import run_live_model, run_model
from app.services.interfaces import (
    Language,
    LanguageDetector,
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
Mode = Literal["live", "conversation", "dialog"]
LANGUAGES: tuple[Language, ...] = ("ko", "en")


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


@dataclass(frozen=True)
class DialogFinal:
    """A dialog turn prepared with the language of the turn before as it was known then (T77)."""

    prepared: pipeline.Prepared
    wav: bytes
    detected: Language
    confidence: float
    previous: Language | None
    guessed: bool


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
        pair: LanguagePair | None,
        expires_at: datetime,
        options: LiveOptions,
        mode: Mode = "live",
        previous: Language | None = None,
        previous_id: int | None = None,
    ) -> None:
        self.websocket = websocket
        self.models = models
        self.stt = stt
        self.translator = translator
        self.store = store
        self.sessions = sessions
        self.user_id = user_id
        # A dialog connection has no fixed direction: each turn's language is detected.
        self.source: Language = pair.source_lang if pair else "ko"
        self.target: Language = pair.target_lang if pair else "en"
        self.expires_at = expires_at
        self.options = options
        self.mode = mode
        # Dialog: the language of the last turn saved (or turned around on the screen) and its id.
        self.previous, self.previous_id = previous, previous_id
        self.saving: asyncio.Task | None = None  # dialog: the last turn's save, which the next one waits for
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
        kinds = ("utterance", "pause", "resume", "end", "cancel")
        if type(number) is not int or kind not in (*kinds, *(("previous",) if self.mode == "dialog" else ())):
            return False
        if kind == "previous":
            # The screen turned a turn around (HTTP speech API, other direction). As in the HTTP dialog,
            # that counts only while it is the last turn saved.
            if message.get("lang") not in LANGUAGES:
                return False
            if number == self.previous_id:
                self.previous = message["lang"]
            return True
        if kind == "utterance":
            self.drop(self.current)
            self.current = Utterance(number, started_at=asyncio.get_running_loop().time())
            if self.mode == "live":
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
        utterance.final = self.spawn(self.prepare(bytes(utterance.pcm), self.previous))

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
        final = utterance.final or self.spawn(self.prepare(bytes(utterance.pcm), self.previous))  # all audio
        if self.mode == "dialog":
            self.saving = self.spawn(self.finish(utterance.id, final, after=self.saving))
        else:
            self.spawn(self.finish(utterance.id, final))

    def drop(self, utterance: Utterance | None) -> None:
        if utterance is None:
            return
        utterance.closed = True
        utterance.changed.set()
        for task in (utterance.updater, utterance.final):
            if task is not None:
                task.cancel()

    async def prepare(self, pcm: bytes, previous: Language | None) -> pipeline.Prepared | DialogFinal:
        # The same WAV file and pipeline as conversation mode's upload, so the same result.
        wav = wav_from_pcm(pcm)
        if self.mode == "dialog":
            return await self.prepare_dialog(wav, previous)
        return await pipeline.prepare(models=self.models, source=self.source, target=self.target, audio=wav)

    async def prepare_dialog(
        self, wav: bytes, previous: Language | None, detection: tuple[Language, float] | None = None
    ) -> DialogFinal:
        """As POST /api/translate/dialog answers it: check the audio, detect its language, translate."""
        if detection is None:
            await run_model(pipeline.validate_audio, wav)
            detection = await run_model(self.models.stt.detect_language, wav)
        detected, confidence = detection
        threshold = get_settings().dialog_language_threshold
        language, guessed = choose_language(detected, confidence, previous, threshold)
        prepared = await pipeline.prepare(
            models=self.models, source=language, target=OTHER[language], audio=wav
        )
        return DialogFinal(prepared, wav, detected, confidence, previous, guessed)

    async def finish(self, number: int, final: asyncio.Task, after: asyncio.Task | None = None) -> None:
        if after is not None:
            await asyncio.wait({after})  # dialog: the turn before is saved (or has failed) first
        try:
            prepared = await final
            turn = prepared if isinstance(prepared, DialogFinal) else None
            threshold = get_settings().dialog_language_threshold
            if turn is not None and turn.previous != self.previous and turn.confidence < threshold:
                # Prepared before the turn ahead was saved or turned around: an unsure turn's guess depends
                # on that turn's language, so it is redone with it (the detection stays).
                turn = await self.prepare_dialog(turn.wav, self.previous, (turn.detected, turn.confidence))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one utterance failed; the connection goes on
            await self.send({"type": "error", "id": number, "detail": failure(exc)})
            return
        try:
            ready = turn.prepared if turn is not None else prepared
            async with self.sessions() as db:
                result = await pipeline.save(db=db, store=self.store, user_id=self.user_id, prepared=ready)
        except Exception as exc:  # noqa: BLE001
            logger.error("Saving a live translation failed: %s", describe(exc))
            await self.send({"type": "error", "id": number, "detail": "The translation could not be saved"})
            return
        payload = result.model_dump(mode="json")
        if turn is not None:
            self.previous, self.previous_id = result.source_lang, number
            payload = DialogResponse(
                **result.model_dump(),
                language_confidence=round(turn.confidence, 3),
                language_guessed=turn.guessed,
            ).model_dump(mode="json")
        await self.send({"type": "final", "id": number, "result": payload})

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
        if not isinstance(start, dict) or start["type"] != "start" or not isinstance(start["token"], str):
            raise ValueError("not a start message")
        mode, pair, previous, previous_id = parse_mode(start)
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
    recognizes = {
        "live": isinstance(models.stt, LiveSpeechToText),
        "conversation": models.stt is not None,
        "dialog": isinstance(models.stt, LanguageDetector),
    }
    if not recognizes[mode] or models.translator is None:
        await websocket.close(UNAVAILABLE)
        return
    await websocket.send_json({"type": "ready"})
    await LiveConnection(
        websocket,
        models,
        models.stt,
        models.translator,
        store,
        sessions,
        user_id,
        pair,
        expires_at,
        options,
        mode=mode,
        previous=previous,
        previous_id=previous_id,
    ).run()


def parse_mode(start: dict) -> tuple[Mode, LanguagePair | None, Language | None, int | None]:
    """The start message's mode (live when left out), with its languages, or ValueError (T77).

    Live and conversation connections translate one fixed direction. A dialog connection detects each turn's
    language; when it replaces one that ended (a new token), the browser gives the last turn's language and
    id so that an unsure next turn and a later turnaround work as they did."""
    mode = start.get("mode", "live")
    if mode in ("live", "conversation"):
        pair = LanguagePair(source_lang=start["source_lang"], target_lang=start["target_lang"])
        return mode, pair, None, None
    if mode != "dialog":
        raise ValueError("unknown mode")
    previous, previous_id = start.get("previous_lang"), start.get("previous_id")
    if (previous is None) != (previous_id is None):
        raise ValueError("previous_lang and previous_id go together")
    if previous is not None and (previous not in LANGUAGES or type(previous_id) is not int):
        raise ValueError("bad previous turn")
    return mode, None, previous, previous_id
