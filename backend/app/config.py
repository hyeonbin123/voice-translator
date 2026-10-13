# [파일 역할] 백엔드 전체의 실행 설정(DB 주소, JWT, 모델 선택·옵션, 동시통역·두 사람 대화·오타 교정 설정)을
# 한곳에 모은 pydantic-settings 클래스다. 필드 이름의 대문자가 환경 변수 이름이다(database_url ↔
# DATABASE_URL). 각 값의 뜻은 docs/api.md '실행 설정과 DB' 표에 있다.
# [설계 이유] `.env`를 자동으로 읽지 않고 기본값과 환경 변수만 쓴다. 기본값이 compose 개발 DB와 같아 그대로
# 실행되고, 모델을 올리지 않는 기본값 덕분에 테스트·CI는 GPU 없이 돈다. 타입·범위(Literal, Field의 gt/ge/le)를
# 여기서 검증하므로 잘못된 값이면 요청을 받기 전에 시작 단계에서 멈춘다.
# [연결] get_settings()를 app/main.py(lifespan), core/security.py, db/session.py, dependencies.py,
# routers(auth·live·translate), services/inference.py, alembic/env.py가 부르고, services/models.py는
# Settings를 받아 모델을 올린다.

# secrets: JWT_SECRET_KEY가 없을 때 쓸 임의 키를 만든다. lru_cache: 설정 객체를 프로세스당 한 번만 만든다.
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

# pydantic: 값 범위 검증(Field), 비밀값을 로그·repr에 드러내지 않는 타입(SecretStr), 사용자 정의
# 검증(field_validator). pydantic_settings.BaseSettings: 필드를 같은 이름의 환경 변수에서 읽어 오는 기반
# 클래스.
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


# 서버 설정 전체. 인스턴스를 만들 때 환경 변수를 읽고 검증한다. 테스트는 Settings(load_models=True, ...)처럼
# 직접 만들어 원하는 조합을 넣는다(tests/test_model_loading.py).
class Settings(BaseSettings):
    # Environment variables override defaults; no .env file is loaded implicitly.
    # 127.0.0.1, not localhost: the compose DB listens on IPv4 only (T50), and on Windows "localhost" tries
    # ::1 first, which cost about 2 s per new connection.
    # 기본 DB 주소: docker compose의 개발 DB. 호스트 포트 55442는 Windows가 예약하는 5432~5631 범위를 피한
    # 값이다(README 로컬 개발). 운영·다른 환경은 DATABASE_URL 환경 변수로 바꾼다.
    database_url: str = "postgresql+asyncpg://vtuser:vtpass@127.0.0.1:55442/voicetranslator"
    # JWT 서명 키. 환경 변수가 없으면 프로세스마다 임의 키(48바이트)를 만든다. 이 경우 서버를 다시 시작하면
    # 기존 토큰이 모두 무효가 되고, 여러 worker를 쓰면 worker마다 키가 달라진다. 그래서 운영에서는 같은 키를
    # 모든 프로세스에 환경 변수로 준다(docs/api.md).
    # 토큰 수명: access 30분, refresh 7일(기본값). 0 이하이면 시작 때 검증 오류.
    jwt_secret_key: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(48)))
    access_token_expire_minutes: int = Field(default=30, gt=0)
    refresh_token_expire_days: int = Field(default=7, gt=0)
    # 합성한 번역 음성 파일을 두는 폴더. parents[2]는 app/config.py → app → backend → 저장소 루트이므로 기본은
    # <저장소>/work/audio(커밋하지 않는 폴더)다. Docker 이미지는 /data/audio를 쓴다(docs/api.md).
    audio_dir: Path = Path(__file__).resolve().parents[2] / "work" / "audio"

    # Models (T11, chosen in docs/experiments.md). Off by default so tests and CI need no GPU or model
    # files; the server sets LOAD_MODELS=true. Without them every translation answers 503.
    # 모델을 올릴지. 기본 False라 pytest·CI는 GPU·모델 파일 없이 API만 띄우고, 번역 요청은 503이 된다. Docker
    # 이미지와 서버 실행은 LOAD_MODELS=true를 준다.
    load_models: bool = False
    # Threads that run the model calls (app/services/inference.py). 1 was kept by measurement
    # (docs/experiments.md 4): the models share one GPU.
    # 모델 호출 스레드 수(services/inference.py의 ThreadPoolExecutor 크기). 1 미만이면 시작 때 멈춘다. 세
    # 모델이 GPU 하나를 나눠 쓰므로 스레드를 늘려도 서로 GPU를 다투기만 해서 측정으로 1을 골랐다.
    model_threads: int = Field(default=1, ge=1)
    # 모델을 돌릴 장치. services/models.py가 cuda면 float16, cpu면 int8 연산 형식을 고른다. stt_model은
    # faster-whisper 모델 이름, ct2_dir은 eval.mt_convert가 CTranslate2로 변환해 둔 번역 모델 폴더다.
    model_device: Literal["cuda", "cpu"] = "cuda"
    stt_model: str = "large-v3-turbo"
    ct2_dir: Path = Path(__file__).resolve().parents[2] / "data" / "models" / "ct2"
    # Translation per direction: opus-mt-tc-big (chosen in T3; en->ko one sentence at a time, T17) or
    # Hy-MT2-1.8B served by Ollama, a trial adopted only by its measurement (T83, docs/experiments.md 15). The
    # Ollama model is created from the official GGUF with eval.hymt_setup; startup waits for it (translation
    # must load) and stops when it is not the expected build (HYMT_DIGEST, the digest in /api/tags).
    # 방향별 번역 모델 선택. 기본 opus(측정으로 고름). hy-mt2는 Ollama에서 도는 번역 LLM으로, 시험에서
    # 채택되지 않아 비교용 설정으로만 남았다(docs/experiments.md 15절). hy-mt2-split은 Hy-MT2를 문장 단위로
    # 부르는 변형이다.
    # hymt_digest: 기대하는 모델 빌드의 digest. 주면 다른 빌드가 올라와 있을 때 시작을 멈춰, 측정한 것과 다른
    # 모델로 서비스하는 일을 막는다. hymt_timeout_s는 번역 요청 하나의 제한, hymt_prepare_timeout_s는 시작할
    # 때 Ollama와 모델을 기다리는 최대 시간(초)이다.
    en_ko_translation: Literal["opus", "hy-mt2", "hy-mt2-split"] = "opus"
    ko_en_translation: Literal["opus", "hy-mt2"] = "opus"
    hymt_model: str = "hy-mt2:1.8b-q8_0"
    hymt_digest: str | None = None
    hymt_timeout_s: float = Field(default=30, gt=0)
    hymt_prepare_timeout_s: float = Field(default=300, gt=0)
    # 끄면 음성 합성 모델을 올리지 않고, 번역 응답에 tts_error가 들어가 화면이 브라우저 내장 음성으로 읽는다.
    tts_enabled: bool = True
    # Korean speech synthesis: MeloTTS on the GPU (chosen in T4) or Supertonic 3 on the CPU, a trial that is
    # adopted only by its measurement and a blind listening test (T78, docs/experiments.md 12). Supertonic
    # runs on a synthesis thread of its own, beside the model thread; eval.supertonic_download gets its files.
    # 한국어 합성 모델 선택. 기본 MeloTTS. supertonic은 두 번의 시험(CPU, GPU)에서 채택되지 않은 비교용
    # 설정이다(docs/experiments.md 12·14절).
    # supertonic_steps: 흐름 정합 단계 수(1~100, 적을수록 빠르고 음질이 낮음). supertonic_threads: ONNX
    # Runtime intra-op 스레드 수.
    ko_tts: Literal["melo", "supertonic"] = "melo"
    supertonic_dir: Path = Path(__file__).resolve().parents[2] / "data" / "models" / "supertonic-3"
    supertonic_steps: int = Field(default=8, ge=1, le=100)
    supertonic_threads: int = Field(default=2, ge=1)
    # ONNX Runtime provider for Supertonic: the CPU (T78), or the GPU's CUDA provider, a trial that needs the
    # onnxruntime-gpu package in place of onnxruntime (T82, docs/experiments.md 14).
    supertonic_provider: Literal["cpu", "cuda"] = "cpu"
    # Freeze the loaded models out of the garbage collector's reach (T23). Off only to measure without it.
    # 모델을 올린 직후 gc.freeze()를 부를지. faster-whisper가 업로드를 풀 때마다 부르는 전체 가비지 컬렉션이
    # 모델의 수많은 객체를 훑으며 GIL을 쥐어 다른 요청이 약 0.3초씩 막혔고, freeze로 0.33초 → 0.03초가
    # 됐다(README 설계 판단, docs/experiments.md 5절).
    gc_freeze: bool = True
    # Pass only the speech parts to Whisper (T14, docs/experiments.md 1-1): no empty results on speech, and
    # silence comes back as "no speech" instead of made-up text.
    # VAD를 켜기 전에는 무음에서 "Thank you." 같은 문장을 지어냈다(README 설계 판단). 켜면 말소리가 없을 때
    # NoSpeechError가 되어 422 "No speech was recognized"로 답한다.
    stt_vad_filter: bool = True
    # Decode uploads in app/services/stt.py instead of inside faster-whisper (T23 candidate C).
    # 비교용으로만 남긴 설정(docs/experiments.md 5절 후보 C). 기본은 faster-whisper가 직접 디코딩한다.
    stt_own_decode: bool = False
    # Live subtitles over the WebSocket (T34, docs/experiments.md 8): the least time between the starts of two
    # updates of one utterance, and how an update recognizes. Chosen by measurement: every 1000 ms, with the
    # final recognition's settings (the least flicker). The final result always uses the settings above, as
    # conversation mode does.
    # 동시통역 자막 갱신 설정. live_update_ms는 한 마디 안의 두 갱신 시작 사이 최소 간격(ms)이다. 250·500ms
    # 갱신보다 1000ms가 덜 흔들려 골랐다(README). 자막 갱신의 beam 크기와 temperature fallback은 최종 인식과
    # 같은 값(5, 켬)이 기본이다.
    live_update_ms: int = Field(default=1000, ge=0)
    live_beam_size: int = Field(default=5, ge=1)
    live_temperature_fallback: bool = True
    # Updates on a thread of their own, and recognition replicas so they can run beside a final (T61).
    # 자막 갱신을 모델 스레드 대기열 대신 전용 스레드(inference.run_live_model)에서 돌릴지. 인식 호출이 실제로
    # 동시에 돌려면 stt_num_workers(faster-whisper 복제 수)도 2 이상이어야 한다. 시험했지만 자막이 빨라지지
    # 않아 기본은 끔이다(README 알려진 한계).
    live_update_thread: bool = False
    stt_num_workers: int = Field(default=1, ge=1)
    # Two-person conversation (T35, docs/experiments.md 10): below this confidence in the detected language,
    # an utterance is taken as the language opposite to the previous one. 0.6 was chosen on validation.
    # 판별 확신은 한국어·영어 확률 중 큰 쪽의 몫(큰 쪽 ÷ 둘의 합)이라 늘 0.5 이상이다. 그래서 문턱은 0.5보다
    # 커야 의미가 있고(gt=0.5), 1.0을 넘을 수 없다.
    dialog_language_threshold: float = Field(default=0.6, gt=0.5, le=1.0)
    # Run every model once after loading, so the first request does not pay for lazy loading (T25).
    # 모델을 올린 뒤 번역·합성·인식을 한 번씩 돌려 둔다(services/models.warm_up). 컨테이너에서 첫 합성이
    # 5~20초, 이후 약 0.2초였다(models.py docstring).
    warm_up: bool = True
    # Fix typos in typed English before translating it, with a small LLM served by Ollama (T32,
    # docs/experiments.md 6). When Ollama cannot be reached, text is translated as typed.
    # 글자로 입력한 영어만 번역 전에 Ollama의 qwen2.5 1.5B로 오타·띄어쓰기를 고친다. 오타를 섞은 영→한 chrF가
    # 27.5 → 34.8로 올랐다(README). correction_timeout_s(10초)를 넘거나 Ollama가 없으면 입력 그대로 번역한다.
    # correction_prepare_timeout_s는 첫 시작의 내려받기·적재·첫 교정 요청마다의 제한이고, 실패하면 다시
    # 시도한다(services/correction.py).
    typo_correction: bool = True
    ollama_url: str = "http://localhost:11434"
    correction_model: str = "qwen2.5:1.5b-instruct"
    correction_timeout_s: float = Field(default=10, gt=0)
    # Limit for each slow preparation call (download, load, first correction); a failed try repeats (T37).
    correction_prepare_timeout_s: float = Field(default=600, gt=0)

    # JWT 키가 UTF-8 기준 32바이트(256비트) 미만이면 시작을 멈춘다. HS256은 SHA-256 기반이라 그보다 짧은 키는
    # 서명 강도를 떨어뜨린다. 자동 생성 키(48바이트)는 늘 통과한다.
    @field_validator("jwt_secret_key")
    @classmethod
    def validate_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET_KEY must contain at least 32 UTF-8 bytes")
        return value


# 설정 객체를 처음 부를 때 한 번 만들어 계속 재사용한다(환경 변수를 요청마다 다시 읽지 않음). 자동 생성 JWT
# 키도 이 덕분에 프로세스 안에서 하나로 유지된다. 테스트는 각 모듈의 get_settings를 monkeypatch로 바꿔 다른
# 설정을 넣는다.
@lru_cache
def get_settings() -> Settings:
    return Settings()
