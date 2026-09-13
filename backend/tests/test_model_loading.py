import pytest

from app import main
from app.config import Settings
from app.services import models, stt, translation, tts
from app.services.pipeline import PipelineModels
from tests.fakes import FakeSpeechToText, FakeTextToSpeech, FakeTranslator


class Recorder:
    """Stands in for a model class and records how it was built."""

    built: list[tuple[str, tuple, dict]] = []

    def __init__(self, *args, **kwargs):
        Recorder.built.append((type(self).__name__, args, kwargs))
        self.model_name = type(self).__name__


def recording(name):
    return type(name, (Recorder,), {})


@pytest.fixture
def fake_model_classes(monkeypatch):
    Recorder.built = []
    monkeypatch.setattr(stt, "WhisperSpeechToText", recording("Whisper"))
    monkeypatch.setattr(translation, "MarianTranslator", recording("Marian"))
    monkeypatch.setattr(tts, "MeloTextToSpeech", recording("Melo"))
    monkeypatch.setattr(tts, "KokoroTextToSpeech", recording("Kokoro"))
    return Recorder.built


def test_loads_the_chosen_models_on_the_configured_device(fake_model_classes, tmp_path):
    bundle = models.load_models(Settings(model_device="cpu", ct2_dir=tmp_path))

    built = {(name, args[0] if args else None) for name, args, _ in fake_model_classes}
    assert ("Whisper", "large-v3-turbo") in built
    assert ("Marian", tmp_path / "opus-mt-tc-big-ko-en") in built
    assert ("Marian", tmp_path / "opus-mt-tc-big-en-ko") in built
    assert ("Melo", "ko") in built
    assert all(kwargs.get("device") == "cpu" for _, _, kwargs in fake_model_classes)
    assert all(kwargs.get("compute_type", "int8") == "int8" for _, _, kwargs in fake_model_classes)
    assert "ko->en" in bundle.translator.model_name
    assert "ko: Melo" in bundle.tts.model_name and "en: Kokoro" in bundle.tts.model_name


def test_synthesis_that_cannot_load_is_turned_off_not_fatal(fake_model_classes, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(tts, "MeloTextToSpeech", broken)
    bundle = models.load_models(Settings(model_device="cpu"))
    assert bundle.tts is None
    assert bundle.stt is not None and bundle.translator is not None
    assert "Speech synthesis could not be loaded" in caplog.text


def test_synthesis_can_be_disabled(fake_model_classes):
    assert models.load_models(Settings(model_device="cpu", tts_enabled=False)).tts is None
    assert not any(name in ("Melo", "Kokoro") for name, _, _ in fake_model_classes)


def test_recognition_that_cannot_load_stops_startup(fake_model_classes, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("model files missing")

    monkeypatch.setattr(stt, "WhisperSpeechToText", broken)
    with pytest.raises(RuntimeError, match="model files missing"):
        models.load_models(Settings(model_device="cpu"))


@pytest.mark.parametrize("load", [True, False])
async def test_lifespan_loads_models_only_when_asked(monkeypatch, load):
    bundle = PipelineModels(stt=FakeSpeechToText(), translator=FakeTranslator(), tts=FakeTextToSpeech())
    monkeypatch.setattr(main, "get_settings", lambda: Settings(load_models=load))
    monkeypatch.setattr(main, "load_models", lambda settings: bundle)

    async with main.lifespan(main.app):
        assert getattr(main.app.state, "models", None) is (bundle if load else None)
    assert not hasattr(main.app.state, "models")  # released on shutdown
