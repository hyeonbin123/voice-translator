"""Load the models chosen in docs/experiments.md into the bundle the translation pipeline uses (T11)."""

import logging
import time
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services.interfaces import Language, ModelError, TextToSpeech, Translator, TypoCorrector
from app.services.pipeline import PipelineModels

logger = logging.getLogger(__name__)

HYMT_RETRY_S = 2.0  # seconds between tries while Ollama starts (T83)
HYMT_MIN_REQUEST_S = 0.01  # a try that starts just before the deadline still gets a valid request limit

WARM_UP_TEXT: dict[Language, str] = {"ko": "안녕하세요.", "en": "Hello."}


def load_models(settings: Settings) -> PipelineModels:
    """Speech recognition and translation must load, or startup fails: no request could succeed without
    them. Speech synthesis is optional: if it cannot load, responses carry tts_error "Speech synthesis is
    not available" and the browser reads the translation aloud instead (docs/api.md).
    """
    from app.services.stt import WhisperSpeechToText

    device = settings.model_device
    engine = {"device": device, "compute_type": "float16" if device == "cuda" else "int8"}
    stt = WhisperSpeechToText(
        settings.stt_model,
        vad_filter=settings.stt_vad_filter,
        own_decode=settings.stt_own_decode,
        live_beam_size=settings.live_beam_size,
        live_temperature_fallback=settings.live_temperature_fallback,
        num_workers=settings.stt_num_workers,
        **engine,
    )
    translator = load_translation(settings, engine)
    tts = load_speech_synthesis(settings) if settings.tts_enabled else None
    corrector = load_typo_correction(settings) if settings.typo_correction else None
    # Supertonic runs off the model thread, which the other models queue on: on the CPU (T78), and on the GPU
    # beside speech recognition (T82, where the measurement shows what that sharing costs).
    off_model_thread = (
        frozenset({"ko"}) if tts is not None and settings.ko_tts == "supertonic" else frozenset()
    )
    return PipelineModels(
        stt=stt,
        translator=translator,
        tts=tts,
        corrector=corrector,
        synthesis_off_model_thread=off_model_thread,
    )


def load_translation(settings: Settings, engine: dict) -> Translator:
    """opus-mt-tc-big both ways by default; Hy-MT2 on Ollama for a direction when its setting says so (T83).

    Startup waits up to HYMT_PREPARE_TIMEOUT_S for Ollama to answer with the model (compose starts the two
    containers together), and stops at once when Ollama has another build of it than HYMT_DIGEST.
    """
    from app.services.translation import DirectionalTranslator, HyMtTranslator, MarianTranslator

    def hymt(by_sentence: bool) -> Translator:
        model = HyMtTranslator(
            settings.hymt_model,
            base_url=settings.ollama_url,
            timeout_s=settings.hymt_timeout_s,
            by_sentence=by_sentence,
            expected_digest=settings.hymt_digest,
            prepare_timeout_s=settings.hymt_prepare_timeout_s,
        )
        _wait_until_ready(model, settings.hymt_prepare_timeout_s)
        return model

    if settings.ko_en_translation == "hy-mt2":
        ko_en = hymt(by_sentence=False)
    else:
        ko_en = MarianTranslator(settings.ct2_dir / "opus-mt-tc-big-ko-en", "ko", "en", **engine)
    if settings.en_ko_translation == "opus":
        # One sentence at a time for en->ko only (T17, docs/experiments.md 2-1).
        en_ko = MarianTranslator(
            settings.ct2_dir / "opus-mt-tc-big-en-ko", "en", "ko", by_sentence=True, **engine
        )
    else:
        en_ko = hymt(by_sentence=settings.en_ko_translation == "hy-mt2-split")
    return DirectionalTranslator({("ko", "en"): ko_en, ("en", "ko"): en_ko})


def _wait_until_ready(model, timeout_s: float) -> None:
    """Try prepare until it succeeds or `timeout_s` has passed.

    Each try gets only the time left, prepare gives each of its requests only what is left when that request
    starts (T87), and the pause before the next try is no longer than the time left either, so a request that
    hangs ends at about the limit (T86). About: httpx applies a limit to each phase of a request (connecting,
    then each read), not to the request as a whole.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            digest = model.prepare(timeout_s=max(deadline - time.monotonic(), HYMT_MIN_REQUEST_S))
        except ModelError as exc:
            left = deadline - time.monotonic()
            if left <= 0:
                raise
            logger.info("Waiting for the translation model on Ollama: %s", exc)
            time.sleep(min(HYMT_RETRY_S, left))
            continue
        logger.info("Translation model %s is ready (build %s)", model.model_name, digest)
        return


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


def load_speech_synthesis(settings: Settings) -> TextToSpeech | None:
    device = settings.model_device
    try:
        from app.services import tts

        if settings.ko_tts == "supertonic":
            korean = tts.SupertonicTextToSpeech(
                settings.supertonic_dir,
                steps=settings.supertonic_steps,
                threads=settings.supertonic_threads,
                provider=settings.supertonic_provider,
            )
        else:
            korean = tts.MeloTextToSpeech("ko", device=device)
        return tts.LanguageTextToSpeech({"ko": korean, "en": tts.KokoroTextToSpeech(device=device)})
    except Exception:  # noqa: BLE001 - a missing tts group or a load failure turns synthesis off, not the server
        logger.exception("Speech synthesis could not be loaded; translations will carry tts_error")
        return None
