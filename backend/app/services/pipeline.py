"""Protocol-only translation orchestration; model loading belongs to the app lifespan.

A translation is first prepared (recognition, translation, synthesis) and then saved as a history record.
The HTTP API does both in one go. The live WebSocket prepares when the speaker pauses and saves only once
the pause turns out to be the end of the utterance (T56), so a prepared translation touches neither the
disk nor the database.
"""

# [파일 역할] 번역 한 건의 흐름을 조립한다: 음성 검사 → 음성 인식 → (영어 글자 입력이면 오타 교정) → 번역 →
# 음성 합성(prepare), 그리고 기록·음성 파일 저장(save).
# [설계 이유] 모델 자체는 Protocol(services/interfaces.py)로만 다뤄 실제 모델 없이 가짜로 테스트할 수 있다.
# 모델 호출은 모두 inference.run_* 으로 이벤트 루프 밖에서 돈다. 준비와 저장을 나눈 것은 동시통역·대화 모드
# 웹소켓이 192ms 쉼에서 미리 준비했다가 말이 정말 끝났을 때만 저장하기 위해서다(위 docstring). 기록 생성 시간
# 필드(stt_ms·mt_ms·tts_ms)는 모델 호출 시간만 재고 대기 시간은 넣지 않는다(docs/api.md 저장과 삭제).
# [연결] routers/translate.py가 translate(준비+저장 한 번에)와 validate_audio를, routers/live.py가 prepare와
# save를 따로 부른다.

import asyncio
import contextlib
import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter
from typing import Literal, TypeVar
from uuid import UUID, uuid4

# av(PyAV, FFmpeg 바인딩): 업로드된 음성을 실제로 풀어 보며 길이를 잰다. 파일의 Content-Type은 믿지
# 않는다(docs/api.md).
import av
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AudioFile, Translation
from app.schemas.history import HistoryItem
from app.schemas.translate import TranslationResponse
from app.services.audio_store import AudioStore
from app.services.errors import describe
from app.services.inference import run_correction, run_model, run_synthesis
from app.services.interfaces import (
    Language,
    ModelError,
    SpeechToText,
    SynthesizedAudio,
    TextToSpeech,
    Translator,
    TypoCorrector,
    UndecodableAudioError,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")
# 업로드 음성 파일 크기 한도 10MB(바이트). 넘으면 routers/translate.py가 413으로 답한다.
MAX_AUDIO_BYTES = 10 * 1024 * 1024


# 음성이 30초를 넘음. ModelError가 아니라 따로 둬, 라우터가 422 "Audio is longer than 30 seconds"로 구분해
# 답한다.
class AudioTooLongError(Exception):
    pass


# 서버가 올려 둔 모델 묶음. 인식·번역이 None이면(LOAD_MODELS=false) 번역 요청은 ModelError → 503이 된다.
# 합성·오타 교정은 없어도 된다. frozen이라 테스트는 dataclasses.replace로 일부만 바꾼 묶음을 만든다.
@dataclass(frozen=True)
class PipelineModels:
    stt: SpeechToText | None
    translator: Translator | None
    tts: TextToSpeech | None = None
    corrector: TypoCorrector | None = None
    # Target languages whose synthesis runs on the CPU: on the synthesis thread, not the model thread (T78).
    synthesis_off_model_thread: frozenset[Language] = frozenset()


# 준비는 끝났지만 아직 저장하지 않은 번역. 합성 음성(speech)이 메모리에만 있어, 동시통역에서 말이
# 이어지면(resume) 디스크·DB를 건드리지 않고 그냥 버릴 수 있다.
# tts_error의 기본값은 '합성 모델이 없음'이고, 합성에 성공하면 None, 실패하면 "Speech synthesis failed"로
# 바뀐다.
@dataclass
class Prepared:
    """A finished translation that is not a record yet: its speech is still in memory."""

    mode: Literal["text", "speech"]
    source: Language
    target: Language
    source_text: str
    translated_text: str
    mt_model: str
    mt_ms: int
    stt_model: str | None = None
    stt_ms: int | None = None
    speech: SynthesizedAudio | None = None
    tts_model: str | None = None
    tts_ms: int | None = None
    tts_error: str | None = "Speech synthesis is not available"


# 업로드가 풀리는 음성인지, 30초 이하인지 확인한다. 컨테이너 머리에 적힌 길이는 꾸며낼 수 있으므로 믿지 않고,
# 실제로 프레임을 풀며 표본 수 ÷ 표본 추출률을 더한다. Fraction(정확한 분수)으로 더해 부동소수점 오차 없이
# 30초 경계를 판단한다.
# 30초를 넘는 순간 멈춰(최대 30초 + 프레임 하나만 풂) 아주 긴 파일로 CPU를 오래 쓰지 않게 한다. 음성 스트림이
# 없거나, 풀린 표본이 0이거나, FFmpeg 오류가 나면 UndecodableAudioError(→ 422 "Audio could not be decoded").
# 모델 스레드에서 돈다(호출하는 쪽이 run_model로 부름).
def validate_audio(audio: bytes) -> None:
    """Decode at most 30 seconds plus one frame, without trusting container duration."""
    try:
        with av.open(io.BytesIO(audio)) as container:
            if not container.streams.audio:
                raise UndecodableAudioError("No audio stream")
            duration = Fraction(0)
            for frame in container.decode(audio=0):
                duration += Fraction(frame.samples, frame.sample_rate)
                if duration > 30:
                    raise AudioTooLongError
            if not duration:
                raise UndecodableAudioError("No decoded samples")
    except (av.error.FFmpegError, ValueError, ZeroDivisionError) as exc:
        raise UndecodableAudioError("Audio decoding failed") from exc


# 함수를 실행하고 (결과, 걸린 밀리초)를 돌려준다. 모델 스레드 안에서 실행되므로 대기열에서 기다린 시간은
# 빠지고 순수 모델 호출 시간만 잰다.
def _timed(fn: Callable[..., T], *args) -> tuple[T, int]:
    start = perf_counter()
    result = fn(*args)
    return result, round((perf_counter() - start) * 1000)


# 번역 하나를 준비한다(저장하지 않음). text와 audio 중 하나를 준다. 인식·번역 실패는 ModelError로 올라가고,
# 합성 실패는 삼키고 tts_error에 적는다(번역은 돌려준다, README 설계 판단).
async def prepare(
    *,
    models: PipelineModels,
    source: Language,
    target: Language,
    text: str | None = None,
    audio: bytes | None = None,
) -> Prepared:
    stt_ms = None
    stt_model = None
    # 음성 입력: 먼저 풀리는지·30초 이하인지 검사하고, 인식 언어는 사용자가 고른 source로 고정한다(언어 자동
    # 판별을 하지 않음, docs/api.md). 인식된 글자가 번역할 원문이 된다.
    if audio is not None:
        await run_model(validate_audio, audio)
        if models.stt is None:
            raise ModelError("Speech recognition model is unavailable")
        transcript, stt_ms = await run_model(_timed, models.stt.transcribe, audio, source)
        text = transcript.text
        stt_model = models.stt.model_name
    if models.translator is None:
        raise ModelError("Translation model is unavailable")
    # Typed text only: speech recognition output has no keyboard typos, and spoken replies should not wait
    # for another model. The record keeps the text as typed; mt_ms includes the correction.
    corrector = models.corrector if audio is None else None
    # The correction is an HTTP call to Ollama, not a model call in this process: it runs on a thread of its
    # own, one at a time, so a slow Ollama (up to CORRECTION_TIMEOUT_S) does not hold the model thread that
    # speech, dialog and live requests queue on (T71, T73).
    corrected, correction_ms = None, 0
    if corrector is not None and corrector.corrects(source):
        corrected, correction_ms = await run_correction(_timed, corrector.correct, text, source)
    # 교정이 성공했으면 교정문을, 실패(None)했거나 교정하지 않았으면 원문을 번역한다. 기록의 source_text는 늘
    # 사용자가 입력한 원문이다.
    translated, translate_ms = await run_model(
        _timed, models.translator.translate, corrected if corrected is not None else text, source, target
    )
    # mt_ms는 교정 시간 + 번역 시간이고, 교정문을 썼을 때만 mt_model에 교정 모델 이름을 붙인다(예: "... +
    # ollama/qwen2.5:1.5b-instruct").
    mt_ms = correction_ms + translate_ms
    mt_model = models.translator.model_name
    if corrected is not None and corrector is not None:
        mt_model = f"{mt_model} + {corrector.model_name}"
    prepared = Prepared(
        mode="speech" if audio is not None else "text",
        source=source,
        target=target,
        source_text=text,
        translated_text=translated,
        mt_model=mt_model,
        mt_ms=mt_ms,
        stt_model=stt_model,
        stt_ms=stt_ms,
    )
    # 합성: 번역문을 도착 언어로 읽는다. CPU 합성(Supertonic)으로 설정된 언어는 합성 전용 스레드로, 나머지는
    # 모델 스레드로 보낸다. OSError도 함께 잡는 것은 합성 중 파일 입출력 오류도 '합성 실패'로 다루기 위해서다.
    if models.tts is not None:
        try:
            run = run_synthesis if target in models.synthesis_off_model_thread else run_model
            prepared.speech, prepared.tts_ms = await run(_timed, models.tts.synthesize, translated, target)
            prepared.tts_model = models.tts.model_name
            prepared.tts_error = None
        except (ModelError, OSError) as exc:
            # Types and place only: a synthesis error's message can quote the translation (T49).
            logger.error("Speech synthesis or audio storage failed: %s", describe(exc))
            prepared.tts_error = "Speech synthesis failed"
    return prepared


# 준비된 번역을 기록으로 저장하고 API 응답을 만든다. 합성 음성이 있으면 먼저 파일로 쓰고, 그다음 기록과
# audio_files 행을 한 트랜잭션으로 저장한다. tts_model·tts_ms는 음성 파일 저장까지 성공했을 때만 기록에
# 넣는다(초기값 None).
async def save(
    *, db: AsyncSession, store: AudioStore, user_id: UUID, prepared: Prepared
) -> TranslationResponse:
    item = Translation(
        user_id=user_id,
        mode=prepared.mode,
        source_lang=prepared.source,
        target_lang=prepared.target,
        source_text=prepared.source_text,
        translated_text=prepared.translated_text,
        stt_model=prepared.stt_model,
        stt_ms=prepared.stt_ms,
        mt_model=prepared.mt_model,
        mt_ms=prepared.mt_ms,
        tts_model=None,
        tts_ms=None,
        audio_file=None,
    )
    tts_error = prepared.tts_error
    saved_path = None
    if prepared.speech is not None:
        audio_id = uuid4()
        # The write runs in a worker thread that a cancellation cannot stop: a live connection that closes
        # mid-save would leave a file no record points to (T59). So wait for it, then remove what it wrote.
        # asyncio.shield: 이 코루틴이 취소돼도(웹소켓이 끊김) 파일 쓰기 작업 자체는 끝까지 가게 한다. 취소되면
        # 쓰기가 끝나길 기다렸다가 쓴 파일을 지우고 취소를 다시 올린다. 파일 쓰기가 OSError로 실패하면 기록은
        # 음성 없이 저장하고 tts_error를 "Speech synthesis failed"로 바꾼다.
        writing = asyncio.ensure_future(store.save(audio_id, prepared.speech.wav))
        try:
            saved_path = await asyncio.shield(writing)
            item.audio_file = AudioFile(id=audio_id, path=saved_path, duration_ms=prepared.speech.duration_ms)
            item.tts_model = prepared.tts_model
            item.tts_ms = prepared.tts_ms
        except OSError as exc:
            logger.error("Speech synthesis or audio storage failed: %s", describe(exc))
            tts_error = "Speech synthesis failed"
        except asyncio.CancelledError:
            with contextlib.suppress(OSError, ValueError):
                await store.delete(await writing)
            raise
    # DB 저장. flush로 INSERT를 보내 DB가 채우는 값(created_at)과 제약 검사를 받은 뒤, commit 전에 응답 모델로
    # 검증한다. 검증이 실패하면 commit하지 않아 응답 없이 기록만 남는 일이 없다. 어떤 예외든(취소 포함,
    # BaseException) 롤백하고 이미 쓴 음성 파일을 지워 파일만 남지 않게 한다.
    try:
        db.add(item)
        await db.flush()
        # Validate before commit, so serialization errors cannot leave an invisible record.
        result = TranslationResponse(**HistoryItem.model_validate(item).model_dump(), tts_error=tts_error)
        await db.commit()
    except BaseException:
        await db.rollback()
        if saved_path is not None:
            await store.delete(saved_path)
        raise
    return result


# HTTP API용: 준비와 저장을 이어서 한 번에 한다(routers/translate.py).
async def translate(
    *,
    models: PipelineModels,
    store: AudioStore,
    db: AsyncSession,
    user_id: UUID,
    source: Language,
    target: Language,
    text: str | None = None,
    audio: bytes | None = None,
) -> TranslationResponse:
    prepared = await prepare(models=models, source=source, target=target, text=text, audio=audio)
    return await save(db=db, store=store, user_id=user_id, prepared=prepared)
