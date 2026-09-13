"""Protocol-only translation orchestration; model loading belongs to the app lifespan."""

import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter
from typing import TypeVar
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
    item = Translation(
        user_id=user_id,
        mode="speech" if audio is not None else "text",
        source_lang=source,
        target_lang=target,
        source_text=text,
        translated_text=translated,
        stt_model=stt_model,
        stt_ms=stt_ms,
        mt_model=mt_model,
        mt_ms=mt_ms,
        tts_model=None,
        tts_ms=None,
        audio_file=None,
    )
    tts_error = "Speech synthesis is not available"
    saved_path = None
    if models.tts is not None:
        try:
            synthesized, tts_ms = await run_model(_timed, models.tts.synthesize, translated, target)
            audio_id = uuid4()
            saved_path = await store.save(audio_id, synthesized.wav)
            item.audio_file = AudioFile(id=audio_id, path=saved_path, duration_ms=synthesized.duration_ms)
            item.tts_model = models.tts.model_name
            item.tts_ms = tts_ms
            tts_error = None
        except (ModelError, OSError) as exc:
            # Types and place only: a synthesis error's message can quote the translation (T49).
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
