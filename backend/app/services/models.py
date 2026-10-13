"""Load the models chosen in docs/experiments.md into the bundle the translation pipeline uses (T11)."""

# [파일 역할] 서버가 시작할 때 측정으로 고른 모델들(인식·번역·합성·오타 교정)을 설정대로 올려 PipelineModels
# 묶음 하나로 만들고, 한 번씩 돌려 예열한다.
# [설계 이유] 모델마다 '반드시 올라와야 하는 것'과 '없어도 서버는 뜨는 것'을 나눈다: 인식·번역이 없으면 어떤
# 요청도 성공할 수 없어 시작을 멈추고, 합성은 실패하면 끈 채 뜨며(브라우저 음성으로 대체), 오타 교정은
# 기다리지 않고 뒤에서 준비한다. 무거운 라이브러리(faster-whisper, ctranslate2, MeloTTS 등)는 함수 안에서
# import해 LOAD_MODELS=false인 테스트·CI가 그 패키지 없이도 앱을 불러올 수 있게 한다.
# [연결] app/main.py의 lifespan이 load_models와 warm_up을 스레드에서 부른다. 만든 묶음은
# dependencies.get_models로 모든 요청에 전달된다.

import logging
import time
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services.interfaces import Language, ModelError, TextToSpeech, Translator, TypoCorrector
from app.services.pipeline import PipelineModels

logger = logging.getLogger(__name__)

# Hy-MT2(비교용 번역 설정)를 Ollama에서 기다릴 때의 재시도 간격(초)과, 마감 직전 시도에도 0이 아닌 요청 제한을
# 주기 위한 최소값(초).
HYMT_RETRY_S = 2.0  # seconds between tries while Ollama starts (T83)
HYMT_MIN_REQUEST_S = 0.01  # a try that starts just before the deadline still gets a valid request limit

# 예열에 쓰는 짧은 문장. 언어마다 번역·합성을 한 번씩 돌린다.
WARM_UP_TEXT: dict[Language, str] = {"ko": "안녕하세요.", "en": "Hello."}


def load_models(settings: Settings) -> PipelineModels:
    """Speech recognition and translation must load, or startup fails: no request could succeed without
    them. Speech synthesis is optional: if it cannot load, responses carry tts_error "Speech synthesis is
    not available" and the browser reads the translation aloud instead (docs/api.md).
    """
    # 모델 라이브러리를 쓰는 모듈은 여기서(함수 안에서) import한다. LOAD_MODELS=false면 이 함수가 불리지
    # 않으므로 테스트·CI는 GPU 패키지 없이 돈다.
    from app.services.stt import WhisperSpeechToText

    # GPU면 float16, CPU면 int8 연산 형식(docs/api.md MODEL_DEVICE). 인식 모델과 opus 번역 모델이 같은 설정을
    # 쓴다.
    # 인식기는 VAD 사용 여부, 직접 디코딩(비교용), 자막 갱신용 beam·temperature 설정, 복제
    # 수(STT_NUM_WORKERS)를 함께 받는다.
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
    # 번역은 필수(실패하면 예외가 올라가 시작이 멈춤), 합성과 오타 교정은 설정으로 끌 수 있고 실패해도
    # None이다.
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


# 방향별 번역기를 만들어 DirectionalTranslator 하나로 묶는다. 기본은 두 방향 모두 opus-mt-tc-big(CTranslate2
# 변환본, ct2_dir에서 읽음)이고, 영→한만 문장 단위로 번역한다: opus-mt가 여러 문장 입력에서 문장을 빠뜨려 나눠
# 번역하니 누락이 줄었고, 한→영은 차이가 없었다(README 설계 판단).
def load_translation(settings: Settings, engine: dict) -> Translator:
    """opus-mt-tc-big both ways by default; Hy-MT2 on Ollama for a direction when its setting says so (T83).

    Startup waits up to HYMT_PREPARE_TIMEOUT_S for Ollama to answer with the model (compose starts the two
    containers together), and stops at once when Ollama has another build of it than HYMT_DIGEST.
    """
    from app.services.translation import DirectionalTranslator, HyMtTranslator, MarianTranslator

    # Hy-MT2 번역기를 만들고, Ollama에 모델이 준비될 때까지(또는 HYMT_PREPARE_TIMEOUT_S가 지날 때까지)
    # 기다린다. 끝내 준비되지 않으면 ModelError가 올라가 서버가 뜨지 않는다(번역은 필수).
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


# 마감 시각(monotonic)을 하나 정해 두고, 시도마다 '남은 시간'만 제한으로 준다. 그래서 여러 번 재시도해도
# 전체가 대략 timeout_s 안에서 끝난다. 실패(ModelError)하면 남은 시간이 있을 때만 잠깐 쉬고 다시 시도하고,
# 마감이 지났으면 마지막 예외를 그대로 올린다. 성공하면 Ollama가 알려 준 모델 빌드(digest)를 로그에 남긴다.
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


# 순서: 두 방향 번역 → 한국어·영어 합성 → 영어 합성 결과를 인식. 인식에 합성 음성을 쓰는 것은, 무음을 넣으면
# VAD가 말이 없다고 보고 모델을 아예 돌리지 않아 예열이 되지 않기 때문이다(docstring). 각 단계의 걸린 시간을
# 로그에 남긴다.
def warm_up(models: PipelineModels) -> None:
    """Run each model once, so the first request does not pay for what loads lazily (T25).

    In the container the first synthesis took 5-20 s and later ones about 0.2 s. Speech recognition gets
    the English synthesis as input, since VAD would skip the model on silence. A step that fails is
    logged and skipped: the server still starts, and a real request reports the failure as usual.
    """

    # 한 단계를 실행하고 시간을 잰다. 실패하면 경고만 남기고 None을 돌려줘 다음 단계로 넘어간다(예열 실패로
    # 서버가 멈추면 안 됨).
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
                # 기본 인수(s=source, t=target)로 반복 변수의 현재 값을 붙잡는다. 그냥 source를 쓰면 람다가
                # 나중에 실행될 때의 값을 읽는 늦은 바인딩 문제가 생길 수 있어서다.
                lambda s=source, t=target: translator.translate(WARM_UP_TEXT[s], s, t),
            )
    english = None
    if tts is not None:
        step("speech synthesis ko", lambda: tts.synthesize(WARM_UP_TEXT["ko"], "ko"))
        english = step("speech synthesis en", lambda: tts.synthesize(WARM_UP_TEXT["en"], "en"))
    if stt is not None and english is not None:
        step("speech recognition en", lambda: stt.transcribe(english.wav, "en"))


# OllamaCorrector를 만들고 start()로 백그라운드 준비 스레드(내려받기·적재·첫 교정, 실패하면 다시 시도)를
# 시작한 뒤 바로 돌려준다. 준비가 끝나기 전에는 corrects()가 False라 글자 입력은 교정 없이 번역된다.
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


# 언어별 합성기를 LanguageTextToSpeech 하나로 묶는다: 한국어는 MeloTTS(기본) 또는 Supertonic 3(비교용), 영어는
# Kokoro-82M.
# 어떤 예외든(tts 의존성 그룹이 설치되지 않아 나는 ImportError, 가중치 해시 불일치, 적재 실패) 로그를 남기고
# None을 돌려준다. 그러면 서버는 합성 없이 뜨고 응답마다 tts_error "Speech synthesis is not available"이
# 들어간다.
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
