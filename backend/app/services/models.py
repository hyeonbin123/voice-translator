"""Load the models chosen in docs/experiments.md into the bundle the translation pipeline uses (T11)."""

import logging
import time
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services.interfaces import Language, TextToSpeech, TypoCorrector
from app.services.pipeline import PipelineModels

logger = logging.getLogger(__name__)

WARM_UP_TEXT: dict[Language, str] = {"ko": "안녕하세요.", "en": "Hello."}


def load_models(settings: Settings) -> PipelineModels:
    """Speech recognition and translation must load, or startup fails: no request could succeed without
    them. Speech synthesis is optional: if it cannot load, responses carry tts_error "Speech synthesis is
    not available" and the browser reads the translation aloud instead (docs/api.md).
    """
    from app.services.stt import WhisperSpeechToText
    from app.services.translation import DirectionalTranslator, MarianTranslator

    device = settings.model_device
    engine = {"device": device, "compute_type": "float16" if device == "cuda" else "int8"}
    stt = WhisperSpeechToText(
        settings.stt_model,
        vad_filter=settings.stt_vad_filter,
        own_decode=settings.stt_own_decode,
        live_beam_size=settings.live_beam_size,
        live_temperature_fallback=settings.live_temperature_fallback,
        **engine,
    )
    translator = DirectionalTranslator(
        {
            ("ko", "en"): MarianTranslator(settings.ct2_dir / "opus-mt-tc-big-ko-en", "ko", "en", **engine),
            # One sentence at a time for en->ko only (T17, docs/experiments.md 2-1).
            ("en", "ko"): MarianTranslator(
                settings.ct2_dir / "opus-mt-tc-big-en-ko", "en", "ko", by_sentence=True, **engine
            ),
        }
    )
    tts = load_speech_synthesis(device) if settings.tts_enabled else None
    corrector = load_typo_correction(settings) if settings.typo_correction else None
    return PipelineModels(stt=stt, translator=translator, tts=tts, corrector=corrector)


def warm_up(models: PipelineModels) -> None:
    """Run each model once, so the first request does not pay for what loads lazily (T25).

    In the container the first synthesis took 5-20 s and later ones about 0.2 s. Speech recognition gets
    the English synthesis as input, since VAD would skip the model on silence. A step that fails is
    logged and skipped: the server still starts, and a real request reports the failure as usual.
    """

    def step(name: str, run: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        try:
            result = run()
        except Exception:  # noqa: BLE001 - warming up must never stop the server
            logger.warning("Warm-up step %s failed", name, exc_info=True)
            return None
        logger.info("Warm-up %s took %.1f s", name, time.perf_counter() - start)
        return result

    translator, tts, stt = models.translator, models.tts, models.stt
    directions: tuple[tuple[Language, Language], ...] = (("ko", "en"), ("en", "ko"))
    if translator is not None:
        for source, target in directions:
            step(
                f"translation {source}->{target}",
                lambda s=source, t=target: translator.translate(WARM_UP_TEXT[s], s, t),
            )
    english = None
    if tts is not None:
        step("speech synthesis ko", lambda: tts.synthesize(WARM_UP_TEXT["ko"], "ko"))
        english = step("speech synthesis en", lambda: tts.synthesize(WARM_UP_TEXT["en"], "en"))
    if stt is not None and english is not None:
        step("speech recognition en", lambda: stt.transcribe(english.wav, "en"))


def load_typo_correction(settings: Settings) -> TypoCorrector:
    """Optional like synthesis, and never waited for (T37): the model is prepared on a background thread
    that retries until Ollama answers, and until then typed text is translated as is.
    """
    from app.services.correction import OllamaCorrector

    corrector = OllamaCorrector(
        settings.correction_model,
        base_url=settings.ollama_url,
        timeout_s=settings.correction_timeout_s,
        prepare_timeout_s=settings.correction_prepare_timeout_s,
    )
    corrector.start()
    return corrector


def load_speech_synthesis(device: str) -> TextToSpeech | None:
    try:
        from app.services.tts import KokoroTextToSpeech, LanguageTextToSpeech, MeloTextToSpeech

        return LanguageTextToSpeech(
            {"ko": MeloTextToSpeech("ko", device=device), "en": KokoroTextToSpeech(device=device)}
        )
    except Exception:  # noqa: BLE001 - a missing tts group or a load failure turns synthesis off, not the server
        logger.exception("Speech synthesis could not be loaded; translations will carry tts_error")
        return None
