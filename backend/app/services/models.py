"""Load the models chosen in docs/experiments.md into the bundle the translation pipeline uses (T11)."""

import logging

from app.config import Settings
from app.services.interfaces import TextToSpeech
from app.services.pipeline import PipelineModels

logger = logging.getLogger(__name__)


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
    return PipelineModels(stt=stt, translator=translator, tts=tts)


def load_speech_synthesis(device: str) -> TextToSpeech | None:
    try:
        from app.services.tts import KokoroTextToSpeech, LanguageTextToSpeech, MeloTextToSpeech

        return LanguageTextToSpeech(
            {"ko": MeloTextToSpeech("ko", device=device), "en": KokoroTextToSpeech(device=device)}
        )
    except Exception:  # noqa: BLE001 - a missing tts group or a load failure turns synthesis off, not the server
        logger.exception("Speech synthesis could not be loaded; translations will carry tts_error")
        return None
