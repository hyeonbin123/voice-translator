import logging
import threading

import httpx
import pytest
from pydantic import ValidationError

from app import main
from app.config import Settings
from app.services import correction, inference, models, stt, translation, tts
from app.services.interfaces import ModelError
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
    monkeypatch.setattr(
        translation, "HyMtTranslator", type("HyMt", (Recorder,), {"prepare": lambda self: "the-build"})
    )
    monkeypatch.setattr(tts, "MeloTextToSpeech", recording("Melo"))
    monkeypatch.setattr(tts, "KokoroTextToSpeech", recording("Kokoro"))
    # Tests must never reach a real Ollama on the machine.
    monkeypatch.setattr(
        correction,
        "OllamaCorrector",
        type("Ollama", (Recorder,), {"start": lambda self: setattr(self, "started", True)}),
    )
    return Recorder.built


def test_loads_the_chosen_models_on_the_configured_device(fake_model_classes, tmp_path):
    bundle = models.load_models(Settings(model_device="cpu", ct2_dir=tmp_path))

    built = {(name, args[0] if args else None) for name, args, _ in fake_model_classes}
    assert ("Whisper", "large-v3-turbo") in built
    assert ("Marian", tmp_path / "opus-mt-tc-big-ko-en") in built
    assert ("Marian", tmp_path / "opus-mt-tc-big-en-ko") in built
    assert ("Melo", "ko") in built
    # Ollama places the typo correction model itself.
    local = [kwargs for name, _, kwargs in fake_model_classes if name != "Ollama"]
    assert all(kwargs.get("device") == "cpu" for kwargs in local)
    assert all(kwargs.get("compute_type", "int8") == "int8" for kwargs in local)
    assert "ko->en" in bundle.translator.model_name
    assert "ko: Melo" in bundle.tts.model_name and "en: Kokoro" in bundle.tts.model_name


def test_speech_recognition_settings_reach_the_model(fake_model_classes):
    models.load_models(Settings(model_device="cpu", tts_enabled=False))
    models.load_models(
        Settings(model_device="cpu", tts_enabled=False, stt_vad_filter=False, stt_own_decode=True)
    )
    whisper = [kwargs for name, _, kwargs in fake_model_classes if name == "Whisper"]
    # VAD on by default (T14); decoding in our code off until T23 candidate C is judged.
    settings = [(kwargs["vad_filter"], kwargs["own_decode"]) for kwargs in whisper]
    assert settings == [(True, False), (False, True)]


def test_only_english_to_korean_translates_sentence_by_sentence(fake_model_classes, tmp_path):
    models.load_models(Settings(model_device="cpu", ct2_dir=tmp_path, tts_enabled=False))
    marian = {
        args[0].name: kwargs.get("by_sentence", False)
        for name, args, kwargs in fake_model_classes
        if name == "Marian"
    }
    # T17 (docs/experiments.md 2-1): splitting helped en->ko and did not help ko->en.
    assert marian == {"opus-mt-tc-big-en-ko": True, "opus-mt-tc-big-ko-en": False}


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


def test_translation_that_cannot_load_stops_startup(fake_model_classes, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("converted model missing")

    monkeypatch.setattr(translation, "MarianTranslator", broken)
    with pytest.raises(RuntimeError, match="converted model missing"):
        models.load_models(Settings(model_device="cpu"))


def test_typo_correction_uses_the_configured_ollama_model(fake_model_classes):
    bundle = models.load_models(
        Settings(
            model_device="cpu",
            tts_enabled=False,
            ollama_url="http://ollama:11434",
            correction_model="qwen2.5:1.5b-instruct",
            correction_timeout_s=3,
            correction_prepare_timeout_s=60,
        )
    )
    ollama = [(args, kwargs) for name, args, kwargs in fake_model_classes if name == "Ollama"]
    expected = {"base_url": "http://ollama:11434", "timeout_s": 3, "prepare_timeout_s": 60}
    assert ollama == [(("qwen2.5:1.5b-instruct",), expected)]
    # Preparation is started, not waited for (T37): loading returns while Ollama pulls or loads.
    assert bundle.corrector.started


def test_typo_correction_can_be_disabled(fake_model_classes):
    assert (
        models.load_models(Settings(model_device="cpu", tts_enabled=False, typo_correction=False)).corrector
        is None
    )
    assert not any(name == "Ollama" for name, _, _ in fake_model_classes)


@pytest.mark.parametrize("value", ["0", "2x"])
def test_an_invalid_model_thread_count_stops_startup(monkeypatch, value):
    # Checked with the other settings, not on the first translation: every request would answer 500.
    monkeypatch.setenv("MODEL_THREADS", value)
    with pytest.raises(ValidationError, match="model_threads"):
        Settings()


def test_the_model_threads_follow_the_setting(monkeypatch):
    monkeypatch.setenv("MODEL_THREADS", "2")
    settings = Settings()
    assert settings.model_threads == 2
    monkeypatch.setattr(inference, "get_settings", lambda: settings)
    pool = inference._model_threads.__wrapped__()  # a new pool, not the one the other tests share
    try:
        assert pool._max_workers == 2
    finally:
        pool.shutdown()


@pytest.fixture
def frozen(monkeypatch):
    """Records what app.state.models held at each gc.freeze() call."""
    calls = []
    monkeypatch.setattr(main.gc, "freeze", lambda: calls.append(getattr(main.app.state, "models", None)))
    return calls


@pytest.mark.parametrize("status", [500, 200])
async def test_shutdown_stops_the_typo_correction_preparation(monkeypatch, frozen, status):
    """T39: a preparation stalled at shutdown neither retries nor turns on afterwards, and a restart works."""
    gate, entered = threading.Event(), threading.Event()

    def ollama(stalled: bool):
        def answer(_):
            if stalled:
                entered.set()
                assert gate.wait(5)
            reply = {"message": {"content": "Hello."}, "done": True, "done_reason": "stop"}
            return httpx.Response(status if stalled else 200, json=reply)

        return answer

    correctors, workers = [], []

    def load(settings):
        corrector = correction.OllamaCorrector("qwen2.5:1.5b-instruct", retry_s=0.01)
        corrector._client.close()
        transport = httpx.MockTransport(ollama(stalled=not correctors))
        corrector._client = httpx.Client(base_url="http://ollama.test", transport=transport)
        correctors.append(corrector)
        workers.append(corrector.start())
        return PipelineModels(stt=FakeSpeechToText(), translator=FakeTranslator(), corrector=corrector)

    monkeypatch.setattr(main, "get_settings", lambda: Settings(load_models=True, warm_up=False))
    monkeypatch.setattr(main, "load_models", load)
    async with main.lifespan(main.app):
        # Shut down only once the worker is inside an HTTP call (T41: not before it reaches Ollama).
        assert entered.wait(5)
        assert not correctors[0].corrects("en")  # still preparing: requests translate as typed
    gate.set()  # the stalled call now ends, failing (500) or succeeding (200), after shutdown
    workers[0].join(5)
    assert not workers[0].is_alive()
    assert not correctors[0].corrects("en") and correctors[0]._client.is_closed

    async with main.lifespan(main.app):  # the next start prepares its own corrector
        workers[1].join(5)
        assert correctors[1].corrects("en")
    assert not correctors[1].corrects("en") and correctors[1]._client.is_closed
    assert not correctors[0].corrects("en")


@pytest.mark.parametrize("load", [True, False])
async def test_lifespan_loads_models_only_when_asked(monkeypatch, frozen, load):
    bundle = PipelineModels(stt=FakeSpeechToText(), translator=FakeTranslator(), tts=FakeTextToSpeech())
    monkeypatch.setattr(main, "get_settings", lambda: Settings(load_models=load))
    monkeypatch.setattr(main, "load_models", lambda settings: bundle)

    async with main.lifespan(main.app):
        assert getattr(main.app.state, "models", None) is (bundle if load else None)
        # Frozen once, after the models are in place (T23), and only when models are loaded.
        assert frozen == ([bundle] if load else [])
    assert not hasattr(main.app.state, "models")  # released on shutdown


async def test_freezing_can_be_turned_off(monkeypatch, frozen):
    bundle = PipelineModels(stt=FakeSpeechToText(), translator=FakeTranslator())
    monkeypatch.setattr(main, "get_settings", lambda: Settings(load_models=True, gc_freeze=False))
    monkeypatch.setattr(main, "load_models", lambda settings: bundle)

    async with main.lifespan(main.app):
        assert main.app.state.models is bundle
    assert frozen == []


async def test_failed_loading_stops_startup_without_freezing(monkeypatch, frozen):
    def broken(settings):
        raise RuntimeError("model files missing")

    monkeypatch.setattr(main, "get_settings", lambda: Settings(load_models=True))
    monkeypatch.setattr(main, "load_models", broken)
    with pytest.raises(RuntimeError, match="model files missing"):
        async with main.lifespan(main.app):
            pass
    assert frozen == []
    assert not hasattr(main.app.state, "models")


@pytest.mark.parametrize("enabled", [True, False])
async def test_models_are_warmed_up_before_freezing(monkeypatch, enabled):
    bundle = PipelineModels(stt=FakeSpeechToText(), translator=FakeTranslator(), tts=FakeTextToSpeech())
    order = []
    monkeypatch.setattr(main, "get_settings", lambda: Settings(load_models=True, warm_up=enabled))
    monkeypatch.setattr(main, "load_models", lambda settings: bundle)
    monkeypatch.setattr(main, "warm_up", lambda models: order.append(("warm_up", models)))
    monkeypatch.setattr(main.gc, "freeze", lambda: order.append(("freeze", None)))

    async with main.lifespan(main.app):
        pass
    # T25: what warming up creates must be frozen too, so it runs first; and it can be turned off.
    assert order == ([("warm_up", bundle)] if enabled else []) + [("freeze", None)]


class Spies:
    """Models that record each call; synthesis returns a real (silent) WAV from FakeTextToSpeech."""

    def __init__(self, synthesis_fails: bool = False) -> None:
        self.calls: list[tuple] = []
        self.synthesis_fails = synthesis_fails

    def translate(self, text, source, target):
        self.calls.append(("translate", source, target))
        return "translated"

    def synthesize(self, text, language):
        self.calls.append(("synthesize", language))
        if self.synthesis_fails:
            raise ModelError("synthesis failed on purpose")
        return FakeTextToSpeech().synthesize(text, language)

    def transcribe(self, audio, language):
        self.calls.append(("transcribe", language, audio[:4]))


def test_warm_up_runs_each_model_and_recognizes_the_english_synthesis():
    spies = Spies()
    models.warm_up(PipelineModels(stt=spies, translator=spies, tts=spies))
    assert spies.calls == [
        ("translate", "ko", "en"),
        ("translate", "en", "ko"),
        ("synthesize", "ko"),
        ("synthesize", "en"),
        ("transcribe", "en", b"RIFF"),
    ]


def test_a_failed_warm_up_step_is_logged_and_the_rest_still_run(caplog):
    spies = Spies(synthesis_fails=True)
    models.warm_up(PipelineModels(stt=spies, translator=spies, tts=spies))
    # Without an English synthesis there is nothing to recognize, so recognition is skipped.
    assert [call[0] for call in spies.calls] == ["translate", "translate", "synthesize", "synthesize"]
    assert "Warm-up step speech synthesis ko failed" in caplog.text


def test_warm_up_skips_models_that_are_not_loaded():
    models.warm_up(PipelineModels(stt=None, translator=None, tts=None))


def test_app_info_logs_reach_a_handler_under_uvicorn():
    # T30: uvicorn leaves the root logger without handlers; the app's own handler prints INFO lines
    # such as the warm-up times, and propagation still feeds pytest's caplog.
    app_logger = logging.getLogger("app")
    assert any(type(handler) is logging.StreamHandler for handler in app_logger.handlers)
    assert logging.getLogger("app.services.models").isEnabledFor(logging.INFO)
    assert app_logger.propagate


def test_korean_synthesis_can_be_supertonic_on_a_thread_of_its_own(fake_model_classes, monkeypatch, tmp_path):
    # T78 (docs/experiments.md 12): a trial setting, MeloTTS stays the default until the trial is decided.
    monkeypatch.setattr(tts, "SupertonicTextToSpeech", recording("Supertonic"))
    bundle = models.load_models(
        Settings(
            model_device="cpu",
            ko_tts="supertonic",
            supertonic_dir=tmp_path,
            supertonic_steps=2,
            supertonic_threads=3,
        )
    )
    built = {name: (args, kwargs) for name, args, kwargs in fake_model_classes}
    assert "Melo" not in built
    assert built["Supertonic"] == ((tmp_path,), {"steps": 2, "threads": 3, "provider": "cpu"})
    assert bundle.tts.model_name == "ko: Supertonic, en: Kokoro"
    # It runs on the CPU, off the model thread the GPU models queue on.
    assert bundle.synthesis_off_model_thread == frozenset({"ko"})


def test_supertonic_can_run_on_the_gpu_through_onnx_runtime(fake_model_classes, monkeypatch, tmp_path):
    # T82 (docs/experiments.md 14): the CUDA provider is a trial setting too; it stays off the model thread.
    monkeypatch.setattr(tts, "SupertonicTextToSpeech", recording("Supertonic"))
    bundle = models.load_models(
        Settings(model_device="cpu", ko_tts="supertonic", supertonic_dir=tmp_path, supertonic_provider="cuda")
    )
    built = {name: (args, kwargs) for name, args, kwargs in fake_model_classes}
    assert built["Supertonic"] == ((tmp_path,), {"steps": 8, "threads": 2, "provider": "cuda"})
    assert bundle.synthesis_off_model_thread == frozenset({"ko"})


def test_melo_stays_the_default_on_the_model_thread(fake_model_classes):
    bundle = models.load_models(Settings(model_device="cpu"))
    assert "ko: Melo" in bundle.tts.model_name
    assert bundle.synthesis_off_model_thread == frozenset()
    assert models.load_models(Settings(model_device="cpu", tts_enabled=False)).synthesis_off_model_thread == (
        frozenset()
    )


def test_supertonic_settings_are_checked():
    with pytest.raises(ValidationError):
        Settings(ko_tts="vits")
    with pytest.raises(ValidationError):
        Settings(supertonic_steps=0)
    with pytest.raises(ValidationError):
        Settings(supertonic_threads=0)
    with pytest.raises(ValidationError):
        Settings(supertonic_provider="tensorrt")
    defaults = Settings()
    assert (defaults.ko_tts, defaults.supertonic_steps, defaults.supertonic_threads) == ("melo", 8, 2)
    assert defaults.supertonic_provider == "cpu"
    assert defaults.supertonic_dir.parts[-3:] == ("data", "models", "supertonic-3")


# Hy-MT2 trial translation (T83, docs/experiments.md 15)


def test_hy_mt2_can_translate_english_to_korean_one_sentence_at_a_time(fake_model_classes, tmp_path):
    bundle = models.load_models(
        Settings(
            model_device="cpu",
            ct2_dir=tmp_path,
            tts_enabled=False,
            en_ko_translation="hy-mt2-split",
            ollama_url="http://ollama:11434",
            hymt_model="hy-mt2:1.8b-q8_0",
            hymt_digest="the-build",
        )
    )
    hymt = [(args, kwargs) for name, args, kwargs in fake_model_classes if name == "HyMt"]
    expected = {
        "base_url": "http://ollama:11434",
        "timeout_s": 30,
        "by_sentence": True,
        "expected_digest": "the-build",
    }
    assert hymt == [(("hy-mt2:1.8b-q8_0",), expected)]
    marian = [args[0].name for name, args, _ in fake_model_classes if name == "Marian"]
    assert marian == ["opus-mt-tc-big-ko-en"]
    assert "en->ko: HyMt" in bundle.translator.model_name
    assert "ko->en: Marian" in bundle.translator.model_name


@pytest.mark.parametrize("en_ko", ["opus", "hy-mt2"])
def test_hy_mt2_can_translate_korean_to_english_whole(fake_model_classes, tmp_path, en_ko):
    bundle = models.load_models(
        Settings(
            model_device="cpu",
            ct2_dir=tmp_path,
            tts_enabled=False,
            ko_en_translation="hy-mt2",
            en_ko_translation=en_ko,
        )
    )
    hymt = [kwargs["by_sentence"] for name, _, kwargs in fake_model_classes if name == "HyMt"]
    assert hymt == ([False] if en_ko == "opus" else [False, False])
    assert "ko->en: HyMt" in bundle.translator.model_name


def test_translation_defaults_to_opus_both_ways(fake_model_classes, tmp_path):
    models.load_models(Settings(model_device="cpu", ct2_dir=tmp_path, tts_enabled=False))
    assert not any(name == "HyMt" for name, _, _ in fake_model_classes)


@pytest.mark.parametrize("field", ["en_ko_translation", "ko_en_translation"])
def test_an_unknown_translation_setting_stops_startup(field):
    with pytest.raises(ValidationError):
        Settings(**{field: "nllb"})


def test_ko_en_has_no_sentence_split_hy_mt2_setting():
    # T17 kept whole input for ko->en; the trial registers only the whole input there.
    with pytest.raises(ValidationError):
        Settings(ko_en_translation="hy-mt2-split")


def flaky_hymt(failures: int, error: Exception):
    calls = []

    class HyMt:
        model_name = "HyMt"

        def __init__(self, *args, **kwargs):
            pass

        def prepare(self):
            calls.append(1)
            if len(calls) <= failures:
                raise error
            return "the-build"

    return HyMt, calls


def test_hy_mt2_startup_waits_for_ollama(fake_model_classes, monkeypatch):
    cls, calls = flaky_hymt(2, ModelError("Ollama is not available"))
    monkeypatch.setattr(translation, "HyMtTranslator", cls)
    monkeypatch.setattr(models, "HYMT_RETRY_S", 0)
    models.load_models(Settings(model_device="cpu", tts_enabled=False, en_ko_translation="hy-mt2"))
    assert len(calls) == 3


def test_hy_mt2_that_never_gets_ready_stops_startup(fake_model_classes, monkeypatch):
    cls, calls = flaky_hymt(10**6, ModelError("Ollama is not available"))
    monkeypatch.setattr(translation, "HyMtTranslator", cls)
    monkeypatch.setattr(models, "HYMT_RETRY_S", 0.01)
    settings = Settings(
        model_device="cpu", tts_enabled=False, en_ko_translation="hy-mt2", hymt_prepare_timeout_s=0.05
    )
    with pytest.raises(ModelError, match="not available"):
        models.load_models(settings)
    assert len(calls) >= 2


def test_hy_mt2_of_another_build_stops_startup_at_once(fake_model_classes, monkeypatch):
    cls, calls = flaky_hymt(10**6, RuntimeError("hy-mt2:1.8b-q8_0 is build another-build"))
    monkeypatch.setattr(translation, "HyMtTranslator", cls)
    monkeypatch.setattr(models, "HYMT_RETRY_S", 0)
    with pytest.raises(RuntimeError, match="another-build"):
        models.load_models(Settings(model_device="cpu", tts_enabled=False, en_ko_translation="hy-mt2"))
    assert len(calls) == 1
