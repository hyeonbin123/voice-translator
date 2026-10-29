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
    # 환경 변수가 기본값을 덮어쓴다. .env 파일은 암묵적으로 읽지 않는다.
    # 기본 DB 주소: docker compose의 개발 DB. 호스트 포트 55442는 Windows가 예약하는 5432~5631 범위를 피한
    # 값이다(README 로컬 개발). 운영·다른 환경은 DATABASE_URL 환경 변수로 바꾼다.
    # localhost가 아니라 127.0.0.1을 쓰는 이유: compose DB는 IPv4로만 연결을 받는데, Windows에서
    # "localhost"는 ::1(IPv6)을 먼저 시도해 새 연결마다 약 2초를 잃었다.
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

    # 모델 설정(각 모델은 docs/experiments.md의 측정으로 골랐다).
    # load_models: 모델을 올릴지. 기본 False라 pytest·CI는 GPU·모델 파일 없이 API만 띄우고, 이때 번역
    # 요청은 모두 503이 된다. Docker 이미지와 서버 실행은 LOAD_MODELS=true를 준다.
    load_models: bool = False
    # 모델 호출 스레드 수(services/inference.py의 ThreadPoolExecutor 크기). 1 미만이면 시작 때 멈춘다. 세
    # 모델이 GPU 하나를 나눠 쓰므로 스레드를 늘려도 서로 GPU를 다투기만 해서, 측정으로 1을 유지했다
    # (docs/experiments.md 4절).
    model_threads: int = Field(default=1, ge=1)
    # 모델 호출(인식·번역·합성·언어 판별·자막 갱신)이 모델 엔진 오류로 연이어 이만큼 실패하면
    # /api/health가 503(models_ok false)을 낸다(services/model_health.py, routers/health.py). 모델 호출이
    # 한 번 성공하면 수가 0으로 돌아간다. 입력 탓 실패와 오타 교정 실패는 세지 않는다. 5인 이유: 다른
    # 프로그램이 GPU 메모리를 잠깐 차지해 생긴 메모리 부족 한두 번으로는 상태를 바꾸지 않고, 사이에 성공
    # 없이 다섯 번 이어지면 사람이 볼 만한 고장으로 본다. LOAD_MODELS=false면 쓰이지 않는다. 1 미만이면
    # 시작 때 멈춘다.
    model_failure_limit: int = Field(default=5, ge=1)
    # 모델을 돌릴 장치. services/models.py가 cuda면 float16, cpu면 int8 연산 형식을 고른다. stt_model은
    # faster-whisper 모델 이름, ct2_dir은 eval.mt_convert가 CTranslate2로 변환해 둔 번역 모델 폴더다.
    model_device: Literal["cuda", "cpu"] = "cuda"
    stt_model: str = "large-v3-turbo"
    ct2_dir: Path = Path(__file__).resolve().parents[2] / "data" / "models" / "ct2"
    # 방향별 번역 모델 선택. 기본은 opus-mt-tc-big(측정으로 고름, 영→한은 한 문장씩 번역)이다. hy-mt2는
    # Ollama에서 도는 번역 LLM Hy-MT2-1.8B로, 측정 결과가 좋을 때만 채택하기로 한 시험 설정이었고 채택되지
    # 않아 비교용으로만 남았다(docs/experiments.md 15절). hy-mt2-split은 Hy-MT2를 문장 단위로 부르는 변형이다.
    # Ollama 모델은 eval.hymt_setup이 공식 GGUF로 만든다. 번역 모델은 반드시 올라와야 하므로 서버 시작이 이
    # 모델을 기다리고, 기대한 빌드가 아니면 시작을 멈춘다(HYMT_DIGEST와 /api/tags의 digest를 비교).
    # hymt_digest: 기대하는 모델 빌드의 digest. 주면 다른 빌드가 올라와 있을 때 시작을 멈춰, 측정한 것과 다른
    # 모델로 서비스하는 일을 막는다. hymt_timeout_s는 번역 요청 하나의 제한, hymt_prepare_timeout_s는 시작할
    # 때 Ollama와 모델을 기다리는 최대 시간(초)이다.
    en_ko_translation: Literal["opus", "hy-mt2", "hy-mt2-split"] = "opus"
    ko_en_translation: Literal["opus", "hy-mt2"] = "opus"
    hymt_model: str = "hy-mt2:1.8b-q8_0"
    hymt_digest: str | None = None
    hymt_timeout_s: float = Field(default=30, gt=0)
    hymt_prepare_timeout_s: float = Field(default=300, gt=0)
    # 영→한 번역 결과의 문장 끝 말투. asis는 번역 모델이 낸 그대로, formal은 문장 끝의 해라체
    # 평서(…다·…ㄴ다·…는다)를 합쇼체(…습니다·…ㅂ니다)로 바꾼다(app/services/register.py, 형태소 분석기
    # python-mecab-ko). 영→한은 문장마다 따로 번역해(docs/experiments.md 2-1절) 한 응답 안에서 말투가 섞일 수
    # 있어 그것을 맞추려는 시험 설정이다. 측정 규칙(21절)을 먼저 정해 잰 뒤 기본값을 정하므로 지금은 asis이고,
    # asis면 이 설정이 없던 때와 같다. formal인데 python-mecab-ko(서버 이미지의 tts 그룹에 있음)가 없으면 서버
    # 시작이 멈춘다.
    ko_register: Literal["asis", "formal"] = "asis"
    # 끄면 음성 합성 모델을 올리지 않고, 번역 응답에 tts_error가 들어가 화면이 브라우저 내장 음성으로 읽는다.
    tts_enabled: bool = True
    # 한국어 합성 모델 선택. 기본은 GPU에서 도는 MeloTTS(측정으로 고름)다. supertonic(Supertonic 3, CPU)은
    # 측정과 블라인드 청취 시험을 통과해야만 채택하기로 한 시험 설정으로, 두 번의 시험(CPU, GPU)에서 채택되지
    # 않아 비교용으로만 남았다(docs/experiments.md 12·14절). Supertonic은 모델 스레드 옆의 전용 합성
    # 스레드에서 돌고, 모델 파일은 eval.supertonic_download가 내려받는다.
    # supertonic_steps: 흐름 정합 단계 수(1~100, 적을수록 빠르고 음질이 낮음). supertonic_threads: ONNX
    # Runtime intra-op 스레드 수.
    ko_tts: Literal["melo", "supertonic"] = "melo"
    supertonic_dir: Path = Path(__file__).resolve().parents[2] / "data" / "models" / "supertonic-3"
    supertonic_steps: int = Field(default=8, ge=1, le=100)
    supertonic_threads: int = Field(default=2, ge=1)
    # Supertonic을 돌릴 ONNX Runtime provider: CPU, 또는 GPU의 CUDA provider. cuda는 onnxruntime 대신
    # onnxruntime-gpu 패키지가 있어야 하는 시험 설정이다(docs/experiments.md 14절).
    supertonic_provider: Literal["cpu", "cuda"] = "cpu"
    # MeloTTS 한국어 전처리(g2pkk)가 쓰는 정규식을 패턴마다 한 번만 컴파일해 두고 다시
    # 쓸지(app/services/tts.py _cache_g2pkk_regex). g2pkk가 어절마다 서로 다른 정규식 약 570개를 쓰는데
    # Python의 re 캐시는 512개라, 끄면 호출마다 모두 다시 컴파일한다. 켜도 음소 출력은
    # 같고(tests/test_g2pkk_regex_cache.py) 컴파일 시간만 준다. 서비스 지연에 주는 효과를 측정 규칙을 먼저
    # 정해 잰 뒤 기본값을 정하므로 지금은 끔이다(끄면 이 설정이 없던 때와 같다).
    melo_regex_cache: bool = False
    # 모델을 올린 직후 gc.freeze()를 불러 올린 모델 객체를 가비지 컬렉터가 훑는 대상에서 뺄지. 끄는 것은
    # 이것 없이 측정할 때뿐이다. faster-whisper가 업로드를 풀 때마다 부르는 전체 가비지 컬렉션이
    # 모델의 수많은 객체를 훑으며 GIL을 쥐어 다른 요청이 약 0.3초씩 막혔고, freeze로 0.33초 → 0.03초가
    # 됐다(README 설계 판단, docs/experiments.md 5절).
    gc_freeze: bool = True
    # Whisper에 말소리 구간만 넘긴다(VAD, docs/experiments.md 1-1절). 그러면 말소리에서 빈 결과가 나오지 않고,
    # 무음은 지어낸 글 대신 "말소리 없음"으로 돌아온다. VAD를 켜기 전에는 무음에서 "Thank you." 같은 문장을
    # 지어냈다(README 설계 판단). 켜면 말소리가 없을 때 NoSpeechError가 되어 422 "No speech was
    # recognized"로 답한다.
    stt_vad_filter: bool = True
    # 업로드 음성을 faster-whisper 안이 아니라 app/services/stt.py에서 직접 디코딩할지. 비교용으로만 남긴
    # 설정이다(docs/experiments.md 5절 후보 C). 기본은 faster-whisper가 직접 디코딩한다.
    stt_own_decode: bool = False
    # 웹소켓 동시통역 자막 갱신 설정(docs/experiments.md 8절). live_update_ms는 한 마디 안의 두 갱신 시작
    # 사이 최소 간격(ms)이고, live_beam_size·live_temperature_fallback은 자막 갱신이 인식하는 방식이다.
    # 측정으로 1000ms 간격과 최종 인식과 같은 설정(beam 5, temperature fallback 켬)을 골랐다. 250·500ms
    # 갱신보다 덜 흔들렸다(README). 최종 결과는 대화 모드처럼 늘 위의 인식 설정을 쓴다.
    live_update_ms: int = Field(default=1000, ge=0)
    live_beam_size: int = Field(default=5, ge=1)
    live_temperature_fallback: bool = True
    # 자막 갱신을 모델 스레드 대기열 대신 전용 스레드(inference.run_live_model)에서 돌려 최종 인식과
    # 나란히 돌 수 있게 할지, 그리고 그때 쓸 인식 복제 수(stt_num_workers). 인식 호출이 실제로
    # 동시에 돌려면 stt_num_workers(faster-whisper 복제 수)도 2 이상이어야 한다. 시험했지만 자막이 빨라지지
    # 않아 기본은 끔이다(README 알려진 한계).
    live_update_thread: bool = False
    stt_num_workers: int = Field(default=1, ge=1)
    # 진단용(docs/experiments.md 17절): 켜면 웹소켓의 final 메시지에 그 마디의 단계별 시각(timings: 미리
    # 처리를 시작한 때부터 모델 스레드 대기, 인식·번역·합성의 시작과 길이, end 받음, 저장, final 보냄)을
    # 더한다. 말을 멈춘 뒤 번역 음성까지의 시간이 어느 단계에서 쓰이는지 나눠 보려는 것이다. 기본은 끔이고,
    # 끄면 메시지와 동작이 이 설정이 없던 때와 같다. 켜도 결과(result)와 처리 순서는 같고, 시각만 더 읽는다.
    live_timings: bool = False
    # 두 사람 대화(docs/experiments.md 10절): 판별한 언어의 확신이 이 값보다 낮으면 그 마디를 직전 마디와
    # 반대 언어로 본다. 0.6은 validation으로 골랐다. 판별 확신은 한국어·영어 확률 중 큰 쪽의 몫(큰 쪽 ÷
    # 둘의 합)이라 늘 0.5 이상이다. 그래서 문턱은 0.5보다 커야 의미가 있고(gt=0.5), 1.0을 넘을 수 없다.
    dialog_language_threshold: float = Field(default=0.6, gt=0.5, le=1.0)
    # 모델을 올린 뒤 번역·합성·인식을 한 번씩 돌려 둔다(services/models.warm_up). 첫 요청이 지연 적재
    # 비용을 치르지 않게 하려는 것이다. 컨테이너에서 첫 합성이 5~20초, 이후 약 0.2초였다(models.py docstring).
    warm_up: bool = True
    # 글자로 입력한 영어만 번역 전에 Ollama가 서비스하는 작은 LLM(qwen2.5 1.5B)으로 오타·띄어쓰기를
    # 고친다(docs/experiments.md 6절). 오타를 섞은 영→한 chrF가
    # 27.5 → 34.8로 올랐다(README). correction_timeout_s(10초)를 넘거나 Ollama가 없으면 입력 그대로 번역한다.
    # correction_prepare_timeout_s는 첫 시작의 내려받기·적재·첫 교정 요청마다의 제한이고, 실패하면 다시
    # 시도한다(services/correction.py).
    typo_correction: bool = True
    ollama_url: str = "http://localhost:11434"
    correction_model: str = "qwen2.5:1.5b-instruct"
    # correction_digest: 기대하는 교정 모델 빌드의 digest(Ollama /api/tags가 알려 주는 64자리 16진수,
    # 앞에 "sha256:" 없이). 같은 태그라도 Ollama 저장소에서 다시 받으면 다른 빌드가 올 수 있어, 측정한
    # 것과 다른 모델로 조용히 교정하는 일을 막으려는 설정이다. 빈 값(기본)이면 지금처럼 확인하지 않는다.
    # 주면 준비 단계가 모델을 GPU에 올리기 전에 비교하고, 다르면 교정을 켜지 않고 경고를 남긴다
    # (services/correction.py). 번역 시험 설정의 hymt_digest는 다르면 서버 시작을 멈추지만, 교정은 보조
    # 단계라 교정만 끄고 서버는 그대로 뜬다. 측정한 빌드는 docs/experiments.md 16절 '실행 조건'에 적은
    # 값이다(.env.example 예시).
    correction_digest: str = ""
    correction_timeout_s: float = Field(default=10, gt=0)
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
