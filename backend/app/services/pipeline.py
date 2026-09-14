"""Protocol-only translation orchestration; model loading belongs to the app lifespan.

A translation is first prepared (recognition, translation, synthesis) and then saved as a history record.
The HTTP API does both in one go. The live WebSocket prepares when the speaker pauses and saves only once
the pause turns out to be the end of the utterance (T56), so a prepared translation touches neither the
disk nor the database.
"""

import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter
from typing import Literal, TypeVar
from uuid import UUID, uuid4

import av
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AudioFile, Translation
from app.schemas.history import HistoryItem
from app.schemas.translate import TranslationResponse
from app.services.audio_store import AudioStore
from app.services.errors import describe
from app.services.inference import run_model
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
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class AudioTooLongError(Exception):
    pass


@dataclass(frozen=True)
class PipelineModels:
    stt: SpeechToText | None
    translator: Translator | None
    tts: TextToSpeech | None = None
    corrector: TypoCorrector | None = None


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


def _timed(fn: Callable[..., T], *args) -> tuple[T, int]:
    start = perf_counter()
    result = fn(*args)
    return result, round((perf_counter() - start) * 1000)


def _translate_text(
    translator: Translator, corrector: TypoCorrector | None, text: str, source: Language, target: Language
) -> tuple[str, bool]:
    """Translate, first fixing typos when the corrector handles the source language (T32).

    Also returns whether a correction was used; without one the text is translated as typed.
    """
    corrected = (
        corrector.correct(text, source) if corrector is not None and corrector.corrects(source) else None
    )
    translated = translator.translate(corrected if corrected is not None else text, source, target)
    return translated, corrected is not None


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
    (translated, corrected), mt_ms = await run_model(
        _timed, _translate_text, models.translator, corrector, text, source, target
    )
    mt_model = models.translator.model_name
    if corrected and corrector is not None:
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
    if models.tts is not None:
        try:
            prepared.speech, prepared.tts_ms = await run_model(
                _timed, models.tts.synthesize, translated, target
            )
            prepared.tts_model = models.tts.model_name
            prepared.tts_error = None
        except (ModelError, OSError) as exc:
            # Types and place only: a synthesis error's message can quote the translation (T49).
            logger.error("Speech synthesis or audio storage failed: %s", describe(exc))
            prepared.tts_error = "Speech synthesis failed"
    return prepared


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
        try:
            audio_id = uuid4()
            saved_path = await store.save(audio_id, prepared.speech.wav)
            item.audio_file = AudioFile(id=audio_id, path=saved_path, duration_ms=prepared.speech.duration_ms)
            item.tts_model = prepared.tts_model
            item.tts_ms = prepared.tts_ms
        except OSError as exc:
            logger.error("Speech synthesis or audio storage failed: %s", describe(exc))
            tts_error = "Speech synthesis failed"
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
