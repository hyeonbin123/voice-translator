"""docs/experiments.md에서 고른 모델들을 올려 번역 흐름이 쓰는 묶음(PipelineModels)으로 만든다."""

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

# Hy-MT2(비교용 번역 설정)를 Ollama에서 기다릴 때 쓰는 값. HYMT_RETRY_S는 Ollama가 시작되는 동안의 재시도
# 간격(초), HYMT_MIN_REQUEST_S는 마감 직전에 시작한 시도도 유효한(0이 아닌) 요청 제한을 받게 하는
# 최소값(초)이다.
HYMT_RETRY_S = 2.0
HYMT_MIN_REQUEST_S = 0.01

# 예열에 쓰는 짧은 문장. 언어마다 번역·합성을 한 번씩 돌린다.
WARM_UP_TEXT: dict[Language, str] = {"ko": "안녕하세요.", "en": "Hello."}


def load_models(settings: Settings) -> PipelineModels:
    """설정대로 모델을 올려 PipelineModels 묶음을 만든다.

    음성 인식과 번역은 반드시 올라와야 하고, 아니면 시작이 실패한다: 둘 없이는 어떤 요청도 성공할 수
    없다. 음성 합성은 선택이다: 올리지 못하면 응답에 tts_error "Speech synthesis is not available"이
    들어가고 브라우저가 대신 번역문을 읽어 준다(docs/api.md).
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
    # Supertonic은 다른 모델들이 줄 서는 모델 스레드 밖에서 돈다: CPU에서 돌 때도, GPU에서 음성 인식과
    # 나란히 돌 때도 그렇다(GPU를 함께 쓰는 비용은 docs/experiments.md 14절의 측정에 있다).
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
    """기본은 두 방향 모두 opus-mt-tc-big이고, 설정이 정하면 그 방향만 Ollama의 Hy-MT2를 쓴다.

    방향별 번역기를 DirectionalTranslator 하나로 묶는다. opus는 CTranslate2 변환본을 ct2_dir에서 읽고,
    영→한만 문장 단위로 번역한다: opus-mt가 여러 문장 입력에서 문장을 빠뜨려 나눠 번역하니 누락이 줄었고,
    한→영은 차이가 없었다(README 설계 판단). Hy-MT2를 쓰면 서버 시작이 Ollama가 그 모델로 답할 때까지
    HYMT_PREPARE_TIMEOUT_S만큼 기다리고(compose가 두 컨테이너를 함께 시작하므로), Ollama에 HYMT_DIGEST와
    다른 빌드가 있으면 바로 멈춘다. KO_REGISTER=formal이면 영→한 번역기(opus든 Hy-MT2든)를
    FormalRegisterTranslator로 감싸 문장 끝 말투를 합쇼체로 맞춘다(docs/experiments.md 21절, 기본 asis는
    감싸지 않음).
    """
    from app.services.translation import (
        DirectionalTranslator,
        FormalRegisterTranslator,
        HyMtTranslator,
        MarianTranslator,
    )

    # 말투 후처리의 형태소 분석기를 번역 모델보다 먼저 올린다: python-mecab-ko가 없으면 GPU에 모델을 올리기
    # 전에 분명한 오류(ModelError)로 시작을 멈춘다(번역은 필수라, 설정한 후처리 없이 몰래 뜨지 않게).
    formalize = None
    if settings.ko_register == "formal":
        from app.services import register

        formalize = register.load_formalizer()

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
        # 영→한만 한 문장씩 번역한다(docs/experiments.md 2-1절).
        en_ko = MarianTranslator(
            settings.ct2_dir / "opus-mt-tc-big-en-ko", "en", "ko", by_sentence=True, **engine
        )
    else:
        en_ko = hymt(by_sentence=settings.en_ko_translation == "hy-mt2-split")
    if formalize is not None:
        en_ko = FormalRegisterTranslator(en_ko, formalize)
    return DirectionalTranslator({("ko", "en"): ko_en, ("en", "ko"): en_ko})


def _wait_until_ready(model, timeout_s: float) -> None:
    """prepare가 성공하거나 `timeout_s`가 지날 때까지 다시 시도한다.

    마감 시각(monotonic)을 하나 정해 두고 시도마다 남은 시간만 제한으로 준다. prepare도 각 요청에 그
    요청이 시작될 때 남은 시간만 주고, 다음 시도 전의 쉼도 남은 시간보다 길지 않다. 그래서 멈춘 요청도
    대략 제한 시각에 끝난다. '대략'인 이유: httpx는 요청 전체가 아니라 요청의 각 단계(연결, 그 뒤 읽기
    하나하나)에 제한을 건다. 실패(ModelError)하면 남은 시간이 있을 때만 잠깐 쉬고 다시 시도하고, 마감이
    지났으면 마지막 예외를 그대로 올린다. 성공하면 Ollama가 알려 준 모델 빌드(digest)를 로그에 남긴다.
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
    """모델마다 한 번씩 돌려, 첫 요청이 지연 적재되는 부분의 비용을 치르지 않게 한다.

    컨테이너에서 첫 합성은 5~20초, 이후는 약 0.2초 걸렸다. 순서: 두 방향 번역 → 한국어·영어 합성 → 영어
    합성 결과를 인식. 음성 인식에는 영어 합성 결과를 입력으로 준다. 무음을 넣으면 VAD가 모델을 아예
    건너뛰어 예열이 되지 않기 때문이다. 실패한 단계는 로그에 남기고 건너뛴다: 서버는 그래도 시작하고,
    실제 요청이 평소처럼 실패를 알린다. 각 단계의 걸린 시간을 로그에 남긴다.
    """

    # 한 단계를 실행하고 시간을 잰다. 실패하면 경고만 남기고 None을 돌려줘 다음 단계로 넘어간다(예열 실패로
    # 서버가 멈추면 안 됨).
    def step(name: str, run: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        try:
            result = run()
        except Exception:  # noqa: BLE001 - 예열 실패로 서버가 멈추면 안 된다
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


def load_typo_correction(settings: Settings) -> TypoCorrector:
    """오타 교정기를 만든다. 합성처럼 선택이고, 서버 시작이 절대 기다리지 않는다.

    OllamaCorrector를 만들고 start()로 백그라운드 준비 스레드(내려받기·적재·첫 교정, Ollama가 답할 때까지
    다시 시도)를 시작한 뒤 바로 돌려준다. 준비가 끝나기 전에는 corrects()가 False라 글자 입력은 교정 없이
    그대로 번역된다.
    """
    from app.services.correction import OllamaCorrector

    # expected_digest(CORRECTION_DIGEST): 비어 있지 않으면 준비 단계가 Ollama의 빌드를 확인하고, 다르면
    # 교정만 끈다(서버는 그대로 뜬다).
    corrector = OllamaCorrector(
        settings.correction_model,
        base_url=settings.ollama_url,
        timeout_s=settings.correction_timeout_s,
        prepare_timeout_s=settings.correction_prepare_timeout_s,
        expected_digest=settings.correction_digest,
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
            # regex_cache(MELO_REGEX_CACHE, 기본 끔): 한국어 전처리 g2pkk의 정규식 컴파일 결과를 다시
            # 쓴다(tts.py).
            korean = tts.MeloTextToSpeech("ko", device=device, regex_cache=settings.melo_regex_cache)
        return tts.LanguageTextToSpeech({"ko": korean, "en": tts.KokoroTextToSpeech(device=device)})
    except Exception:  # noqa: BLE001 - tts 그룹이 없거나 적재가 실패하면 서버가 아니라 합성만 끈다
        logger.exception("Speech synthesis could not be loaded; translations will carry tts_error")
        return None
