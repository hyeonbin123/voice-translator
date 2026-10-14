# [파일 역할] 서버 시작 때 모델을 올리는 app/services/models.py(load_models, warm_up)와 앱 수명 주기
# app/main.py(lifespan)를 검증한다. 실제 모델 클래스는 생성 인자만 기록하는 대역(Recorder)으로 바꿔 GPU·가중치
# 없이 돈다.
# [무엇을 확인하나] 측정으로 고른 모델과 설정이 실제로 생성자에 들어가는지, 인식·번역은 실패하면 시작을 멈추고
# 합성은 끈 채 뜨는지, 오타 교정은 기다리지 않고 뒤에서 준비하며 종료 때 멈추는지, 설정 검증(MODEL_THREADS
# 등), gc.freeze와 예열의 순서, 비교용 설정(Supertonic, Hy-MT2)의 연결과 시작 대기 시간 한도.
# [왜 중요한가] 측정 결과(docs/experiments.md)는 이 설정들로 잰 것이라, 서버가 다른 모델이나 옵션으로 뜨면
# README의 수치가 서비스에 들어맞지 않는다.

import logging
import threading

# httpx: 실제 OllamaCorrector의 HTTP를 MockTransport로 바꿀 때 쓴다. ValidationError: 잘못된 설정값이 Settings
# 생성에서 거절되는지 확인한다.
import httpx
import pytest
from pydantic import ValidationError

# main: lifespan과 gc.freeze 호출을 시험한다. models·stt·translation·tts·correction: 모델 클래스를 대역으로
# 바꿀 모듈들. inference: 모델 스레드 풀 크기를 확인한다.
from app import main
from app.config import Settings
from app.services import correction, inference, models, stt, translation, tts
from app.services.interfaces import ModelError
from app.services.pipeline import PipelineModels
from tests.fakes import FakeSpeechToText, FakeTextToSpeech, FakeTranslator


# 모델 클래스 대역: 생성될 때 (클래스 이름, 위치 인자, 키워드 인자)를 공용 목록에 남기고, model_name을 클래스
# 이름으로 둔다. 모델을 실제로 올리지 않고 '무엇을 어떤 설정으로 만들었는가'만 본다.
class Recorder:
    """모델 클래스 대신 쓰이며, 어떻게 만들어졌는지 기록한다."""

    built: list[tuple[str, tuple, dict]] = []

    def __init__(self, *args, **kwargs):
        Recorder.built.append((type(self).__name__, args, kwargs))
        self.model_name = type(self).__name__


# 이름만 다른 Recorder 하위 클래스를 만든다(기록에서 모델 종류를 구분하려고).
def recording(name):
    return type(name, (Recorder,), {})


# 인식(Whisper)·번역(Marian, Hy-MT2)·합성(Melo, Kokoro)·오타 교정(Ollama) 클래스를 모두 대역으로 바꾸고 기록
# 목록을 돌려준다. Hy-MT2 대역은 prepare가 곧바로 빌드 이름을, 교정기 대역은 start가 표시만 남기므로 이 PC의
# 실제 Ollama에 닿지 않는다.
@pytest.fixture
def fake_model_classes(monkeypatch):
    Recorder.built = []
    monkeypatch.setattr(stt, "WhisperSpeechToText", recording("Whisper"))
    monkeypatch.setattr(translation, "MarianTranslator", recording("Marian"))
    monkeypatch.setattr(
        translation,
        "HyMtTranslator",
        type("HyMt", (Recorder,), {"prepare": lambda self, timeout_s=None: "the-build"}),
    )
    monkeypatch.setattr(tts, "MeloTextToSpeech", recording("Melo"))
    monkeypatch.setattr(tts, "KokoroTextToSpeech", recording("Kokoro"))
    monkeypatch.setattr(
        correction,
        "OllamaCorrector",
        type("Ollama", (Recorder,), {"start": lambda self: setattr(self, "started", True)}),
    )
    return Recorder.built


# 고른 모델이 실제로 만들어지는지 확인한다: 인식 large-v3-turbo, 번역 opus-mt-tc-big 두 방향(변환본 폴더
# ct2_dir 아래), 한국어 합성 MeloTTS, 영어 Kokoro. 로컬 모델은 모두 설정한 장치(cpu)와 그에 맞는 계산
# 형식(int8)을 받는다. 오타 교정 모델은 Ollama가 장치를 정하므로 제외한다.
def test_loads_the_chosen_models_on_the_configured_device(fake_model_classes, tmp_path):
    bundle = models.load_models(Settings(model_device="cpu", ct2_dir=tmp_path))

    built = {(name, args[0] if args else None) for name, args, _ in fake_model_classes}
    assert ("Whisper", "large-v3-turbo") in built
    assert ("Marian", tmp_path / "opus-mt-tc-big-ko-en") in built
    assert ("Marian", tmp_path / "opus-mt-tc-big-en-ko") in built
    assert ("Melo", "ko") in built
    local = [kwargs for name, _, kwargs in fake_model_classes if name != "Ollama"]
    assert all(kwargs.get("device") == "cpu" for kwargs in local)
    assert all(kwargs.get("compute_type", "int8") == "int8" for kwargs in local)
    assert "ko->en" in bundle.translator.model_name
    assert "ko: Melo" in bundle.tts.model_name and "en: Kokoro" in bundle.tts.model_name


# 음성 인식 설정(STT_VAD_FILTER, STT_OWN_DECODE)이 생성자에 그대로 들어가는지 기본값과 바꾼 값으로 확인한다.
# 기본은 VAD 켬(무음에서 문장을 지어내던 문제의 해결, README 설계 판단)과 자체 디코딩 끔이다.
def test_speech_recognition_settings_reach_the_model(fake_model_classes):
    models.load_models(Settings(model_device="cpu", tts_enabled=False))
    models.load_models(
        Settings(model_device="cpu", tts_enabled=False, stt_vad_filter=False, stt_own_decode=True)
    )
    whisper = [kwargs for name, _, kwargs in fake_model_classes if name == "Whisper"]
    # VAD 기본 켬은 T14 결정, 자체 디코딩(T23 후보 C) 기본 끔은 5절에서 B를 유지하기로 한 결정이다.
    settings = [(kwargs["vad_filter"], kwargs["own_decode"]) for kwargs in whisper]
    assert settings == [(True, False), (False, True)]


# 문장 단위 번역은 영→한에만 켠다. 나눠 번역하니 영→한은 누락이 줄었고 한→영은 차이가
# 없었다(docs/experiments.md 2-1절, README 설계 판단).
def test_only_english_to_korean_translates_sentence_by_sentence(fake_model_classes, tmp_path):
    models.load_models(Settings(model_device="cpu", ct2_dir=tmp_path, tts_enabled=False))
    marian = {
        args[0].name: kwargs.get("by_sentence", False)
        for name, args, kwargs in fake_model_classes
        if name == "Marian"
    }
    assert marian == {"opus-mt-tc-big-en-ko": True, "opus-mt-tc-big-ko-en": False}


# 합성 모델이 올라가지 못하면(예: GPU 메모리 부족) 서버는 합성 없이 뜨고 경고 로그를 남긴다. 인식·번역은
# 그대로 있다. 이때 응답은 tts_error를 싣고 화면이 브라우저 음성으로 읽는다(load_models docstring).
def test_synthesis_that_cannot_load_is_turned_off_not_fatal(fake_model_classes, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(tts, "MeloTextToSpeech", broken)
    bundle = models.load_models(Settings(model_device="cpu"))
    assert bundle.tts is None
    assert bundle.stt is not None and bundle.translator is not None
    assert "Speech synthesis could not be loaded" in caplog.text


# TTS_ENABLED=false면 합성 모델을 아예 만들지 않는다.
def test_synthesis_can_be_disabled(fake_model_classes):
    assert models.load_models(Settings(model_device="cpu", tts_enabled=False)).tts is None
    assert not any(name in ("Melo", "Kokoro") for name, _, _ in fake_model_classes)


# 인식 모델이 올라가지 못하면 예외를 그대로 올려 시작을 멈춘다. 인식 없이는 어떤 음성 요청도 성공할 수 없기
# 때문이다(load_models docstring).
def test_recognition_that_cannot_load_stops_startup(fake_model_classes, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("model files missing")

    monkeypatch.setattr(stt, "WhisperSpeechToText", broken)
    with pytest.raises(RuntimeError, match="model files missing"):
        models.load_models(Settings(model_device="cpu"))


# 번역 모델(변환본 없음 등)도 마찬가지로 시작을 멈춘다.
def test_translation_that_cannot_load_stops_startup(fake_model_classes, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("converted model missing")

    monkeypatch.setattr(translation, "MarianTranslator", broken)
    with pytest.raises(RuntimeError, match="converted model missing"):
        models.load_models(Settings(model_device="cpu"))


# 오타 교정기가 설정한 Ollama 주소·모델·요청 제한·준비 제한으로 만들어지고, 준비는 시작만 하고 기다리지
# 않는지(started) 확인한다. 서버는 Ollama가 모델을 받거나 올리는 동안에도 먼저 뜨고, 그동안 영어 글자는 교정
# 없이 번역된다(README).
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
    assert bundle.corrector.started


# TYPO_CORRECTION=false면 교정기를 만들지 않는다.
def test_typo_correction_can_be_disabled(fake_model_classes):
    assert (
        models.load_models(Settings(model_device="cpu", tts_enabled=False, typo_correction=False)).corrector
        is None
    )
    assert not any(name == "Ollama" for name, _, _ in fake_model_classes)


# MODEL_THREADS가 0이나 숫자가 아니면 다른 설정과 함께 시작 때 ValidationError로 멈춘다(커밋 57bdeee). 첫 번역
# 요청 때 가서야 실패하면 모든 요청이 500이 되기 때문이다(테스트 주석).
@pytest.mark.parametrize("value", ["0", "2x"])
def test_an_invalid_model_thread_count_stops_startup(monkeypatch, value):
    monkeypatch.setenv("MODEL_THREADS", value)
    with pytest.raises(ValidationError, match="model_threads"):
        Settings()


# MODEL_THREADS=2면 모델 스레드 풀의 작업자도 2개다. lru_cache로 공유되는 풀 대신 __wrapped__로 새 풀을 만들어
# 다른 테스트에 영향을 주지 않고, 끝나면 닫는다.
def test_the_model_threads_follow_the_setting(monkeypatch):
    monkeypatch.setenv("MODEL_THREADS", "2")
    settings = Settings()
    assert settings.model_threads == 2
    monkeypatch.setattr(inference, "get_settings", lambda: settings)
    pool = inference._model_threads.__wrapped__()  # 다른 테스트가 공유하는 풀이 아닌 새 풀
    try:
        assert pool._max_workers == 2
    finally:
        pool.shutdown()


# gc.freeze를 대역으로 바꿔, 불릴 때마다 그 순간 app.state.models에 무엇이 있었는지 기록한다. 모델을 올린 뒤에
# 얼렸는지 확인하려는 것이다.
@pytest.fixture
def frozen(monkeypatch):
    """gc.freeze()가 불릴 때마다 app.state.models에 무엇이 있었는지 기록한다."""
    calls = []
    monkeypatch.setattr(main.gc, "freeze", lambda: calls.append(getattr(main.app.state, "models", None)))
    return calls


# 회귀 테스트(커밋 f27ed92, 6d4add4): 오타 교정 준비가 Ollama 응답을 기다리며 멈춰 있는 동안 서버가 종료되면,
# 그 준비는 종료 뒤 실패(500)하든 성공(200)하든 다시 시도하지 않고 교정을 켜지도 않으며 HTTP 클라이언트를
# 닫아야 한다. 이어서 다시 시작하면 새 교정기가 정상적으로 준비되고, 종료 때 닫히며, 옛 교정기는 꺼진 채
# 남는지 확인한다.
# 방법: 첫 교정기의 준비 요청만 gate에서 붙잡아 '요청이 Ollama에 닿은 뒤' 종료하게 하고, 종료 뒤에 풀어 준다.
@pytest.mark.parametrize("status", [500, 200])
async def test_shutdown_stops_the_typo_correction_preparation(monkeypatch, frozen, status):
    """종료 때 멈춰 있던 준비는 그 뒤 재시도하거나 켜지지 않고, 다시 시작하면 정상 동작한다."""
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
        # 준비 작업이 HTTP 호출 안에 들어간 뒤에야 종료한다(Ollama에 닿기 전이 아니라).
        assert entered.wait(5)
        assert not correctors[0].corrects("en")  # 아직 준비 중: 요청은 입력 그대로 번역된다
    gate.set()  # 멈춰 있던 호출이 이제 종료 뒤에 실패(500)하거나 성공(200)하며 끝난다
    workers[0].join(5)
    assert not workers[0].is_alive()
    assert not correctors[0].corrects("en") and correctors[0]._client.is_closed

    async with main.lifespan(main.app):  # 다음 시작은 자기 교정기를 새로 준비한다
        workers[1].join(5)
        assert correctors[1].corrects("en")
    assert not correctors[1].corrects("en") and correctors[1]._client.is_closed
    assert not correctors[0].corrects("en")


# LOAD_MODELS가 켜졌을 때만 모델을 올려 app.state에 두고, 그 뒤에 gc.freeze를 한 번 부르며(모델을 얼려 전체
# 가비지 컬렉션이 훑지 않게, README 설계 판단), 종료하면 app.state에서 모델을 놓는지 확인한다. 꺼져 있으면
# 아무것도 올리지도 얼리지도 않는다(테스트·CI).
@pytest.mark.parametrize("load", [True, False])
async def test_lifespan_loads_models_only_when_asked(monkeypatch, frozen, load):
    bundle = PipelineModels(stt=FakeSpeechToText(), translator=FakeTranslator(), tts=FakeTextToSpeech())
    monkeypatch.setattr(main, "get_settings", lambda: Settings(load_models=load))
    monkeypatch.setattr(main, "load_models", lambda settings: bundle)

    async with main.lifespan(main.app):
        assert getattr(main.app.state, "models", None) is (bundle if load else None)
        assert frozen == ([bundle] if load else [])
    assert not hasattr(main.app.state, "models")  # 종료하면 놓는다


# GC_FREEZE=false면 모델은 올리되 얼리지 않는다(비교 측정용 설정).
async def test_freezing_can_be_turned_off(monkeypatch, frozen):
    bundle = PipelineModels(stt=FakeSpeechToText(), translator=FakeTranslator())
    monkeypatch.setattr(main, "get_settings", lambda: Settings(load_models=True, gc_freeze=False))
    monkeypatch.setattr(main, "load_models", lambda settings: bundle)

    async with main.lifespan(main.app):
        assert main.app.state.models is bundle
    assert frozen == []


# 모델 적재가 실패하면 예외가 lifespan 밖으로 나가 시작이 멈추고, 얼리지도 않고 app.state에 모델을 남기지도
# 않는다.
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


# 예열(warm_up)은 얼리기 전에 돈다. 예열이 만드는 객체(첫 호출 때 늦게 올라오는 것들)까지 함께 얼려야 하기
# 때문이다(테스트 주석, main.py). WARM_UP=false면 예열 없이 얼리기만 한다.
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
    assert order == ([("warm_up", bundle)] if enabled else []) + [("freeze", None)]


# 번역·합성·인식을 모두 흉내 내며 호출을 기록하는 대역. 합성은 실제 WAV(무음)를 돌려주고, 일부러 실패하게 할
# 수 있다.
class Spies:
    """호출마다 기록하는 모델들. 합성은 FakeTextToSpeech로 만든 실제 (무음) WAV를 돌려준다."""

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


# 예열 순서: 번역 두 방향 → 합성 두 언어 → 영어 합성 음성을 인식. 인식에 무음이 아니라 합성 음성(RIFF로
# 시작하는 WAV)을 넣는 것은 VAD가 무음이면 모델을 건너뛰어 예열이 되지 않기 때문이다(warm_up docstring).
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


# 예열 한 단계가 실패해도 경고 로그만 남기고 나머지 단계는 계속한다(예열은 서버 시작을 막으면 안 됨). 영어
# 합성이 실패하면 인식할 음성이 없으므로 인식 예열은 건너뛴다.
def test_a_failed_warm_up_step_is_logged_and_the_rest_still_run(caplog):
    spies = Spies(synthesis_fails=True)
    models.warm_up(PipelineModels(stt=spies, translator=spies, tts=spies))
    assert [call[0] for call in spies.calls] == ["translate", "translate", "synthesize", "synthesize"]
    assert "Warm-up step speech synthesis ko failed" in caplog.text


# 모델이 하나도 없어도 예외 없이 끝난다.
def test_warm_up_skips_models_that_are_not_loaded():
    models.warm_up(PipelineModels(stt=None, translator=None, tts=None))


# uvicorn은 루트 로거에 핸들러를 달지 않아, 앱 자체 핸들러가 없으면 모델 적재·예열 시간 같은 INFO 로그가
# 사라졌다(커밋 5e646ed). 'app' 로거에 StreamHandler가 있고 INFO가 켜져 있으며, 전파(propagate)도 켜져
# pytest의 caplog가 계속 받는지 확인한다.
def test_app_info_logs_reach_a_handler_under_uvicorn():
    app_logger = logging.getLogger("app")
    assert any(type(handler) is logging.StreamHandler for handler in app_logger.handlers)
    assert logging.getLogger("app.services.models").isEnabledFor(logging.INFO)
    assert app_logger.propagate


# 비교용 설정 KO_TTS=supertonic이면 한국어 합성만 Supertonic(설정한 폴더·단계 수·스레드 수, CPU 제공자)으로
# 바꾸고 MeloTTS는 만들지 않는다. CPU로 도므로 한국어 합성은 GPU 모델들이 줄 서는 모델 스레드가 아닌 합성 전용
# 스레드로 보낸다(synthesis_off_model_thread). 영어는 Kokoro 그대로다.
def test_korean_synthesis_can_be_supertonic_on_a_thread_of_its_own(fake_model_classes, monkeypatch, tmp_path):
    # T78(docs/experiments.md 12절)의 시험용 설정이다. 시험 결과 기본값은 MeloTTS 그대로다.
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
    assert bundle.synthesis_off_model_thread == frozenset({"ko"})


# SUPERTONIC_PROVIDER=cuda도 같은 비교용 설정이며(T82, docs/experiments.md 14절), 기본 단계 수 8·스레드 2와
# 함께 cuda 제공자가 넘어간다. 이때도 모델 스레드가 아닌 합성 전용 스레드를 쓴다.
def test_supertonic_can_run_on_the_gpu_through_onnx_runtime(fake_model_classes, monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "SupertonicTextToSpeech", recording("Supertonic"))
    bundle = models.load_models(
        Settings(model_device="cpu", ko_tts="supertonic", supertonic_dir=tmp_path, supertonic_provider="cuda")
    )
    built = {name: (args, kwargs) for name, args, kwargs in fake_model_classes}
    assert built["Supertonic"] == ((tmp_path,), {"steps": 8, "threads": 2, "provider": "cuda"})
    assert bundle.synthesis_off_model_thread == frozenset({"ko"})


# 기본 설정은 MeloTTS이고 모든 합성이 모델 스레드에서 돈다(빈 집합). 합성을 끈 경우도 빈 집합이다.
def test_melo_stays_the_default_on_the_model_thread(fake_model_classes):
    bundle = models.load_models(Settings(model_device="cpu"))
    assert "ko: Melo" in bundle.tts.model_name
    assert bundle.synthesis_off_model_thread == frozenset()
    assert models.load_models(Settings(model_device="cpu", tts_enabled=False)).synthesis_off_model_thread == (
        frozenset()
    )


# Supertonic 관련 설정 검증: 모르는 합성 이름, 단계·스레드 0, 모르는 제공자는 시작 때 거절한다. 기본값(melo,
# 8단계, 2스레드, cpu, data/models/supertonic-3 폴더)도 고정한다.
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


# 여기부터 Hy-MT2 번역 시험 설정(T83, docs/experiments.md 15절)


# EN_KO_TRANSLATION=hy-mt2-split이면 영→한만 Hy-MT2 문장 단위 팔로 바꾸고 한→영은 opus 그대로다. Hy-MT2에는
# Ollama 주소, 요청 제한 30초, 측정한 빌드 digest, 시작 대기 한도(적재 요청도 이 한도를 넘지 않음)가
# 넘어가는지 확인한다.
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
            hymt_prepare_timeout_s=45,
        )
    )
    hymt = [(args, kwargs) for name, args, kwargs in fake_model_classes if name == "HyMt"]
    expected = {
        "base_url": "http://ollama:11434",
        "timeout_s": 30,
        "by_sentence": True,
        "expected_digest": "the-build",
        "prepare_timeout_s": 45,  # 적재 요청도 시작 대기 한도보다 오래 기다리지 않는다
    }
    assert hymt == [(("hy-mt2:1.8b-q8_0",), expected)]
    marian = [args[0].name for name, args, _ in fake_model_classes if name == "Marian"]
    assert marian == ["opus-mt-tc-big-ko-en"]
    assert "en->ko: HyMt" in bundle.translator.model_name
    assert "ko->en: Marian" in bundle.translator.model_name


# KO_EN_TRANSLATION=hy-mt2면 한→영을 통째로 번역하는 Hy-MT2로 바꾼다. 영→한도 hy-mt2면 Hy-MT2가 방향마다
# 하나씩(둘 다 통째로) 만들어진다.
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


# 기본 설정은 두 방향 모두 opus이고 Hy-MT2는 만들지 않는다(시험에서 채택되지 않음, README 알려진 한계).
def test_translation_defaults_to_opus_both_ways(fake_model_classes, tmp_path):
    models.load_models(Settings(model_device="cpu", ct2_dir=tmp_path, tts_enabled=False))
    assert not any(name == "HyMt" for name, _, _ in fake_model_classes)


# 등록되지 않은 번역 설정값(nllb)은 시작 때 거절한다.
@pytest.mark.parametrize("field", ["en_ko_translation", "ko_en_translation"])
def test_an_unknown_translation_setting_stops_startup(field):
    with pytest.raises(ValidationError):
        Settings(**{field: "nllb"})


# 한→영은 T17에서 통째 입력을 유지했으므로 시험에도 통째로 번역만 등록했고, hy-mt2-split 값을 받지 않는다.
def test_ko_en_has_no_sentence_split_hy_mt2_setting():
    with pytest.raises(ValidationError):
        Settings(ko_en_translation="hy-mt2-split")


# 처음 failures번은 정해 준 예외로 실패하고 그 뒤엔 성공하는 Hy-MT2 대역과, prepare에 넘어간 시간 제한 기록을
# 만든다.
def flaky_hymt(failures: int, error: Exception):
    calls = []

    class HyMt:
        model_name = "HyMt"

        def __init__(self, *args, **kwargs):
            pass

        def prepare(self, timeout_s=None):
            calls.append(timeout_s)
            if len(calls) <= failures:
                raise error
            return "the-build"

    return HyMt, calls


# Ollama가 아직 준비되지 않아 ModelError가 나면 시작은 재시도하며 기다린다(두 번 실패 뒤 세 번째에 성공).
# 재시도 간격은 0으로 줄여 테스트를 빠르게 한다.
def test_hy_mt2_startup_waits_for_ollama(fake_model_classes, monkeypatch):
    cls, calls = flaky_hymt(2, ModelError("Ollama is not available"))
    monkeypatch.setattr(translation, "HyMtTranslator", cls)
    monkeypatch.setattr(models, "HYMT_RETRY_S", 0)
    models.load_models(Settings(model_device="cpu", tts_enabled=False, en_ko_translation="hy-mt2"))
    assert len(calls) == 3


# 시작 대기 한도(HYMT_PREPARE_TIMEOUT_S, 여기서는 0.05초)가 지나도 준비되지 않으면 마지막 ModelError로 시작을
# 멈춘다.
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


# 회귀 테스트(커밋 605eed4): 시작 대기 한도가 적재 요청이 멈춰 버리는 경우까지 포함한 전체 대기를 묶는지
# 확인한다(docs/api.md). 가짜 시계(time.monotonic·time.sleep)로, 시도마다 주어진 시간을 다 쓰는 경우(None)와
# 1초씩 쓰는 경우를 돌린다. 전체 시간이 10초를 넘지 않고, 첫 시도는 정확히 10초를 받으며, 이후 시도의 제한은
# 점점 줄어야 한다.
@pytest.mark.parametrize("attempt_s", [None, 1.0])
def test_hy_mt2_startup_waits_no_longer_than_its_limit(fake_model_classes, monkeypatch, attempt_s):
    clock = [0.0]
    asked = []

    class HyMt:
        model_name = "HyMt"

        def __init__(self, *args, **kwargs):
            self.default_s = kwargs.get("prepare_timeout_s", 300)

        def prepare(self, timeout_s=None):
            asked.append(timeout_s)
            limit = self.default_s if timeout_s is None else timeout_s
            clock[0] += limit if attempt_s is None else min(attempt_s, limit)
            raise ModelError("Ollama could not load hy-mt2:1.8b-q8_0: ReadTimeout")

    monkeypatch.setattr(translation, "HyMtTranslator", HyMt)
    monkeypatch.setattr(models.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(models.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    settings = Settings(
        model_device="cpu", tts_enabled=False, en_ko_translation="hy-mt2", hymt_prepare_timeout_s=10
    )
    with pytest.raises(ModelError, match="could not load"):
        models.load_models(settings)
    assert clock[0] <= 10 + 0.01
    assert asked[0] == pytest.approx(10)  # 시도마다 남은 시간만 받고, 그보다 더 받지 않는다
    assert all(later <= earlier for earlier, later in zip(asked, asked[1:], strict=False))


# 설치된 모델이 측정한 빌드와 다르면(RuntimeError) 기다려도 풀리지 않으므로 재시도 없이 한 번에 시작을 멈춘다.
def test_hy_mt2_of_another_build_stops_startup_at_once(fake_model_classes, monkeypatch):
    cls, calls = flaky_hymt(10**6, RuntimeError("hy-mt2:1.8b-q8_0 is build another-build"))
    monkeypatch.setattr(translation, "HyMtTranslator", cls)
    monkeypatch.setattr(models, "HYMT_RETRY_S", 0)
    with pytest.raises(RuntimeError, match="another-build"):
        models.load_models(Settings(model_device="cpu", tts_enabled=False, en_ko_translation="hy-mt2"))
    assert len(calls) == 1
