"""Live subtitles over a WebSocket (T34, T56; the contract is in docs/api.md).

While a person speaks, the browser streams the utterance's audio. Now and then the server recognizes all of
it again, translates the result and sends both, with the start that two results in a row agree on marked
stable. When the speaker pauses, the clip conversation mode would upload is complete: the server prepares
the final translation right then, and saves and sends it once the browser confirms the utterance ended.

Conversation and dialog modes (T77) use the same connection without the subtitles: only the prepare at a
pause and the save at the end, so their result is the HTTP speech or dialog API's for the same clip. Dialog
turns are saved in the order they ended, since an unsure turn's language depends on the turn before.
"""

# [파일 역할] 웹소켓 /api/translate/live 하나로 세 모드를 처리한다: 동시통역(live, 말하는 동안 원문·번역문
# 자막 갱신), 대화 모드(conversation)와 두 사람 대화(dialog)의 '192ms 쉼에서 미리 처리'. 메시지 계약은
# docs/api.md 'WS /api/translate/live'에 있다.
# [설계 이유] 한 마디의 경계(시작·쉼·끝)는 브라우저의 Silero VAD가 정하고, 서버는 그 신호에 맞춰 일한다. 말이
# 192ms 멈추면(pause) 대화 모드가 올릴 소리가 이미 다 모였으므로 최종 인식·번역·합성을 바로 시작하고, 1초 쉼이
# 확인되면(end) 그때만 기록을 저장한다. 다시 말하면(resume) 준비한 결과를 버린다. 이렇게 해 말을 멈춘 뒤 번역
# 음성까지의 시간을 줄였다(README: 대화 모드 한국어 원문 1.86 → 1.24초 등, docs/experiments.md 11절).
# [주요 구성] LiveOptions(갱신 간격) → Utterance(한 마디의 상태) → LiveConnection(메시지 처리, 자막 갱신 루프,
# 미리 처리·저장) → live()(첫 메시지 인증과 모드 확인) → parse_mode(첫 메시지 해석).
# [연결] 모델 호출은 services/pipeline.py와 inference.py, 진하게 보일 길이는 services/live.py, 두 사람 대화의
# 언어 결정은 services/dialog.py. 화면 쪽은 frontend의 동시통역·대화 모드가 이 웹소켓을 쓴다.

import asyncio
import json
import logging
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

# jwt: 첫 메시지 토큰의 검증 실패 예외. async_sessionmaker: 연결이 길게 열려 있어도 DB 연결을 붙잡지 않도록
# 저장할 때만 세션을 열려고 세션 팩토리를 받는다.
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

# 한 마디 소리의 최대 바이트: 30초 × 16000표본/초 × 2바이트 = 960,000바이트. HTTP API의 30초 제한과 같다.
MAX_UTTERANCE_BYTES = 30 * 16_000 * 2  # the HTTP API's 30 s limit, in 16 kHz 16-bit audio
# The browser starts an utterance 64 ms into speech and sends the 192 ms before it as well (T33): this much
# audio is there when an utterance starts.
PREROLL_S = 0.256
# 웹소켓 닫힘 코드. 4000번대는 앱이 정의해 쓸 수 있는 범위라, HTTP의 401·422·503에 4000을 더해 뜻을
# 맞췄다(docs/api.md 닫힘 코드).
UNAUTHORIZED, INVALID, UNAVAILABLE = 4401, 4422, 4503
Mode = Literal["live", "conversation", "dialog"]
LANGUAGES: tuple[Language, ...] = ("ko", "en")


# 연결 설정. interval_s는 LIVE_UPDATE_MS(기본 1초), start_timeout_s는 첫 메시지(start)를 기다리는 시간(10초,
# docs/api.md). 테스트는 get_live_options를 바꿔 짧은 간격을 넣는다.
@dataclass(frozen=True)
class LiveOptions:
    interval_s: float  # least time between the starts of two updates of one utterance
    start_timeout_s: float = 10.0
    update_thread: bool = False  # updates on a thread of their own (LIVE_UPDATE_THREAD, T61)


def get_live_options() -> LiveOptions:
    settings = get_settings()
    return LiveOptions(interval_s=settings.live_update_ms / 1000, update_thread=settings.live_update_thread)


# 진행 중인 한 마디의 상태. eq=False라 필드 값이 아니라 객체 자체로 비교된다(같은 내용의 두 마디를 같은 것으로
# 보지 않음).
# pcm: 지금까지 받은 소리. changed: 새 소리나 상태 변화가 있으면 켜져 자막 갱신 루프를 깨운다. paused: 192ms
# 쉼 상태. too_long: 30초를 넘음. closed: 끝났거나 버려짐. updater: 자막 갱신 작업. final: 미리 처리 작업.
# recognized/shown: 바로 앞 인식 결과와 마지막으로 보낸 (글, 진한 길이).
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


# 두 사람 대화의 미리 처리 결과. 판별 결과(detected, confidence)와 그때 알던 직전 마디 언어(previous)를 함께
# 남겨, 저장할 때 직전 마디 언어가 바뀌었으면 판별을 다시 하지 않고 언어 결정만 다시 할 수 있게 한다.
@dataclass(frozen=True)
class DialogFinal:
    """A dialog turn prepared with the language of the turn before as it was known then (T77)."""

    prepared: pipeline.Prepared
    wav: bytes
    detected: Language
    confidence: float
    previous: Language | None
    guessed: bool


# 실패한 마디에 보낼 error 메시지의 detail을 고른다. 문구는 HTTP 음성 번역 API의 422·503 문구와
# 같다(routers/translate.failures_as_http와 같은 분류). 웹소켓은 HTTP 예외를 쓸 수 없어 문자열로 돌려준다.
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


# 인증을 마친 웹소켓 연결 하나. 한 번에 진행 중인 마디는 하나(current)이고, 미리 처리·저장·자막 갱신은
# 백그라운드 작업(tasks)으로 돈다.
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
        # run_update: 자막 갱신을 모델 스레드 대기열에 넣을지 전용 스레드로 돌릴지(LIVE_UPDATE_THREAD).
        # sending: 여러 작업이 동시에 send_json을 부르면 메시지가 섞일 수 있어 하나씩 보내도록 잠근다.
        self.run_update = run_live_model if options.update_thread else run_model
        self.current: Utterance | None = None
        self.tasks: set[asyncio.Task] = set()
        self.sending = asyncio.Lock()

    # 백그라운드 작업을 만들고 tasks 집합에 넣어 둔다. 연결이 끝날 때 남은 작업을 모두 취소하려는 것이고, 끝난
    # 작업은 스스로 집합에서 빠진다. asyncio는 약한 참조만 갖고 있어 참조를 들고 있지 않으면 작업이 도중에
    # 사라질 수 있다는 점도 이 집합이 막는다.
    def spawn(self, coroutine: Coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        # A dropped final's failure is never awaited; retrieve it so asyncio does not report it.
        task.add_done_callback(lambda done: done.cancelled() or done.exception())
        return task

    # JSON 메시지를 보낸다. 브라우저가 이미 떠났으면 보내기 오류를 무시한다. 연결 종료 처리는 받기 루프(run)가
    # 한 곳에서 맡는다.
    async def send(self, message: dict) -> None:
        async with self.sending:
            try:
                await self.websocket.send_json(message)
            except (WebSocketDisconnect, RuntimeError, OSError):
                pass  # the browser has gone; the receive loop ends the connection

    # 받기 루프. 메시지마다 먼저 토큰 만료를 확인해, 연결 중 access 토큰이 만료되면 만료 뒤 첫 메시지에서
    # 4401로 닫는다(화면은 새 토큰으로 다시 연결). 바이너리는 소리, 글은 제어 메시지이고, 계약에 없는 메시지면
    # 4422로 닫는다.
    # 끝날 때(정상·오류 모두) 진행 중인 마디를 버리고 남은 작업을 취소한 뒤 끝나기를 기다린다. 그래서 연결이
    # 끊기면 아직 저장하지 않은 결과는 버려진다(docs/api.md).
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

    # 제어 메시지 하나를 처리한다. type과 정수 id가 없거나 모르는 종류면 False(→ 4422). bool은 int의 하위
    # 타입이라 isinstance 대신 type(...) is int로 true/false를 id로 받지 않는다. previous 메시지는 두 사람
    # 대화에서만 허용한다.
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
        # 새 마디 시작: 진행 중이던 마디는 버린다. 동시통역 모드에서만 자막 갱신 작업을 띄운다(대화 모드·두
        # 사람 대화는 자막이 없음).
        if kind == "utterance":
            self.drop(self.current)
            self.current = Utterance(number, started_at=asyncio.get_running_loop().time())
            if self.mode == "live":
                self.current.updater = self.spawn(self.update(self.current))
            return True
        # pause·resume·end·cancel은 지금 마디의 id에만 반응한다. 이미 끝났거나 버린 마디에 대한 늦은 메시지는
        # 오류가 아니라 무시한다(네트워크로 순서가 엇갈릴 수 있음).
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

    # 소리(바이너리 메시지)를 지금 마디에 붙인다. 30초를 넘으면 소리를 비우고(메모리 절약) 갱신·미리 처리를
    # 멈춘 뒤, end가 오면 "Audio is longer than 30 seconds" 오류를 보낸다. 새 소리가 오면 changed를 켜 자막
    # 갱신 루프를 깨운다.
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

    # 말 끝 뒤 192ms 조용함. 지금까지의 소리가 대화 모드가 올릴 소리와 같으므로 최종 처리를 미리 시작한다.
    # bytes()로 복사해 넘겨, 이후 들어오는 소리가 준비 중인 입력을 바꾸지 않게 한다. 두 사람 대화면 그 시점의
    # 직전 마디 언어를 함께 넘긴다.
    def pause(self, utterance: Utterance) -> None:
        """The clip is complete (192 ms of quiet): prepare its final translation now."""
        if utterance.paused or utterance.too_long:
            return
        utterance.paused = True
        utterance.final = self.spawn(self.prepare(bytes(utterance.pcm), self.previous))

    # 1초가 되기 전에 다시 말함: 미리 처리를 취소하고 자막 갱신을 다시 깨운다. 이미 모델 스레드에서 돌고 있는
    # 호출은 멈출 수 없어 끝까지 돌지만, 그 뒤 단계는 실행되지 않는다(README: 버려지는 미리 처리가 GPU를 더
    # 씀).
    def resume(self, utterance: Utterance) -> None:
        if not utterance.paused:
            return
        utterance.paused = False
        if utterance.final is not None:
            utterance.final.cancel()  # a model call already running still finishes, but nothing after it
            utterance.final = None
        utterance.changed.set()

    # 1초 쉼 확인(또는 29초 강제 끊김): 자막 갱신을 멈추고, 미리 처리한 결과가 있으면 그것을, pause 없이
    # 끝났으면 지금 받은 소리 전체로 새로 준비해 저장·전송(finish)한다.
    # 두 사람 대화는 앞 마디의 저장(self.saving)이 끝난 뒤에 저장하도록 사슬로 잇는다. 확신이 낮은 마디의
    # 언어는 앞 마디 언어에 따라 정해지므로, 끝난 순서대로 저장해야 하기 때문이다(위 docstring).
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

    # 마디를 버린다(cancel 메시지, 새 마디 시작, 연결 종료): 닫힘 표시 후 갱신·미리 처리 작업을 취소한다.
    # changed를 켜 기다리던 갱신 루프가 깨어나 끝나게 한다.
    def drop(self, utterance: Utterance | None) -> None:
        if utterance is None:
            return
        utterance.closed = True
        utterance.changed.set()
        for task in (utterance.updater, utterance.final):
            if task is not None:
                task.cancel()

    # 최종 결과 준비(저장하지 않음). 받은 PCM을 대화 모드 업로드와 같은 WAV로 만들어 HTTP와 같은 파이프라인에
    # 넣으므로 결과가 같다(README: 두 모드 160문장 모두 판정 뒤 올린 결과와 같음).
    async def prepare(self, pcm: bytes, previous: Language | None) -> pipeline.Prepared | DialogFinal:
        # The same WAV file and pipeline as conversation mode's upload, so the same result.
        wav = wav_from_pcm(pcm)
        if self.mode == "dialog":
            return await self.prepare_dialog(wav, previous)
        return await pipeline.prepare(models=self.models, source=self.source, target=self.target, audio=wav)

    # 두 사람 대화 한 마디 준비: POST /api/translate/dialog와 같은 순서(음성 검사 → 언어 판별 → 언어 결정 →
    # 번역 흐름). detection을 주면 판별을 건너뛰고 그 결과로 언어 결정만 다시 한다.
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

    # 준비 작업의 결과를 기다려 저장하고 final 메시지를 보낸다. 실패하면 그 마디만 error를 보내고 연결은
    # 유지한다.
    # asyncio.wait({after})는 앞 작업이 실패·취소돼도 예외를 올리지 않고 끝나기만 기다린다(await after와 다른
    # 점).
    async def finish(self, number: int, final: asyncio.Task, after: asyncio.Task | None = None) -> None:
        if after is not None:
            await asyncio.wait({after})  # dialog: the turn before is saved (or has failed) first
        try:
            prepared = await final
            turn = prepared if isinstance(prepared, DialogFinal) else None
            # 두 사람 대화: 미리 처리할 때 알던 직전 언어와 지금(앞 마디 저장 뒤) 직전 언어가 다르고 확신이
            # 문턱보다 낮으면, 판별 결과는 그대로 두고 새 직전 언어로 다시 처리한다. 확신이 높으면 직전 언어와
            # 상관없이 결과가 같으므로 다시 하지 않는다.
            threshold = get_settings().dialog_language_threshold
            if turn is not None and turn.previous != self.previous and turn.confidence < threshold:
                # Prepared before the turn ahead was saved or turned around: an unsure turn's guess depends
                # on that turn's language, so it is redone with it (the detection stays).
                turn = await self.prepare_dialog(turn.wav, self.previous, (turn.detected, turn.confidence))
        except asyncio.CancelledError:
            # 취소는 오류 메시지 없이 그대로 올려 연결 종료 정리가 진행되게 한다. 그 밖의 예외는 이 마디의
            # 실패다.
            raise
        except Exception as exc:  # noqa: BLE001 - one utterance failed; the connection goes on
            await self.send({"type": "error", "id": number, "detail": failure(exc)})
            return
        # 저장은 이때만 짧게 DB 세션을 연다(연결 내내 붙잡지 않음). 저장 실패는 "The translation could not be
        # saved"로 알린다.
        try:
            ready = turn.prepared if turn is not None else prepared
            async with self.sessions() as db:
                result = await pipeline.save(db=db, store=self.store, user_id=self.user_id, prepared=ready)
        except Exception as exc:  # noqa: BLE001
            logger.error("Saving a live translation failed: %s", describe(exc))
            await self.send({"type": "error", "id": number, "detail": "The translation could not be saved"})
            return
        # 두 사람 대화면 저장한 마디의 언어와 id를 '직전 마디'로 기억하고, 응답에 확신·추측 여부를 더해 HTTP
        # 두 사람 대화 응답과 같은 형태로 만든다. mode="json"은 UUID·datetime을 JSON 문자열로 바꾼다.
        payload = result.model_dump(mode="json")
        if turn is not None:
            self.previous, self.previous_id = result.source_lang, number
            payload = DialogResponse(
                **result.model_dump(),
                language_confidence=round(turn.confidence, 3),
                language_guessed=turn.guessed,
            ).model_dump(mode="json")
        await self.send({"type": "final", "id": number, "result": payload})

    # 동시통역 자막 갱신 루프(마디당 하나). 한 번에 한 갱신만 돌리고, 끝나면 그때까지 받은 소리 전체로 다음
    # 갱신을 한다(밀린 갱신은 합쳐짐). 두 갱신의 시작은 interval_s 이상 떨어진다.
    # 첫 갱신 시각: 마디가 시작될 때 이미 약 256ms(PREROLL_S) 소리가 와 있으므로, 간격에서 그만큼 뺀 시각에 첫
    # 갱신을 해 소리가 약 interval_s만큼 쌓였을 때 인식한다.
    async def update(self, utterance: Utterance) -> None:
        """Recognize the utterance so far again and again, one update at a time (docs/experiments.md 8)."""
        loop = asyncio.get_running_loop()
        next_at = utterance.started_at + max(0.0, self.options.interval_s - PREROLL_S)
        seen = 0  # bytes of audio the last update recognized
        while not utterance.closed:
            if (delay := next_at - loop.time()) > 0:
                await asyncio.sleep(delay)
                continue
            # 쉼 상태(미리 처리 중), 30초 초과, 새 소리가 없음 중 하나면 다음 변화(changed)까지 잠든다. 그
            # 밖에는 지금까지의 소리를 복사해 다시 인식한다.
            utterance.changed.clear()
            if utterance.paused or utterance.too_long or len(utterance.pcm) <= seen:
                await utterance.changed.wait()
                continue
            pcm = bytes(utterance.pcm)
            seen, next_at = len(pcm), loop.time() + self.options.interval_s
            # 자막용 인식(transcribe_live): 말이 없으면 예외 대신 빈 문자열. 인식 실패는 이 갱신만 건너뛰고
            # 루프는 계속한다.
            try:
                text = await self.run_update(self.stt.transcribe_live, pcm, self.source)
            except ModelError as exc:
                logger.warning("Live recognition failed: %s", describe(exc))
                continue
            # 바로 앞 인식 결과와 같은 앞부분(단어 경계까지)을 진하게 보일 길이로 정한다. 보낸 것과 같은 결과,
            # 그리고 아직 아무것도 보내지 않았는데 빈 결과면 보내지 않는다. 마디가 그사이 끝났으면 보내지
            # 않는다.
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
            # 원문 글이 바뀌었을 때만 번역한다(진한 길이만 바뀐 경우는 번역하지 않음). 글자가
            # 없거나(문장부호뿐) 쉼 상태면 번역을 건너뛴다: 쉼에서는 최종 번역이 이미 준비되고 있다.
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


# 웹소켓 진입점. 순서: 연결 수락 → 10초 안에 첫 메시지(start) 받기·모드 해석(형식 오류 4422) → access 토큰
# 검증(실패 4401) → 사용자 존재 확인(없으면 4401) → 모드에 필요한 모델 확인(없으면 4503) → ready 전송 →
# LiveConnection.run().
# 모드별 필요 모델: 동시통역은 자막용 인식(LiveSpeechToText), 대화 모드는 인식 모델만 있으면 되고, 두 사람
# 대화는 언어 판별(LanguageDetector)이 필요하다. 셋 다 번역 모델이 있어야 한다.
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
    # 사용자 확인은 짧은 세션으로 하고 바로 닫는다. 연결 동안 DB 연결을 붙잡지 않기
    # 위해서다(dependencies.get_sessions).
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


# 첫 메시지에서 (모드, 언어 쌍, 직전 언어, 직전 id)를 꺼낸다. 동시통역·대화 모드는 source_lang·target_lang을
# LanguagePair로 검증하고(틀리면 ValidationError → 4422), 두 사람 대화는 방향이 없는 대신 다시 연결할 때
# previous_lang과 previous_id를 함께 받는다(한쪽만 있으면 오류).
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
