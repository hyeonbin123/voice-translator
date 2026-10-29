# [파일 역할] 상태 확인 API(GET /api/health, routers/health.py)의 응답 계약을 검증한다(docs/api.md의 GET
# /api/health).
# [무엇을 확인하나] DB 연결 상태, 올라간 모델 이름, 오타 교정 준비 여부(typo_correction_ready), 동시통역 가능
# 여부(live), DB에 닿지 못할 때 503을 내면서 오류 메시지를 응답·로그에 남기지 않는지, 그리고 모델을 올린
# 서버(LOAD_MODELS=true)에서만 모델 호출의 연속 실패(models_ok, consecutive_model_failures)를 알리고 한도에
# 닿으면 503을 내는지. LOAD_MODELS=false(테스트·CI)의 응답은 이 필드가 생기기 전과 같아야 한다.
# [왜 중요한가] Docker compose의 api 상태 검사가 이 경로를 쓰고, 응답 필드는 문서에 적힌 공개 계약이다.

import logging

import pytest

from app.config import Settings

# get_db·get_models를 dependency_overrides로 바꿔 'DB가 끊긴 서버'와 '모델이 올라간 서버'를 흉내 낸다. 가짜
# 모델은 model_name이 fake-* 로 정해져 있어 응답의 이름을 그대로 비교할 수 있다.
from app.dependencies import get_db, get_models
from app.main import app
from app.routers import health as health_router
from app.services import model_health
from app.services.inference import run_model
from app.services.interfaces import ModelError
from app.services.model_health import ModelHealth
from app.services.pipeline import PipelineModels
from tests.fakes import (
    FakeCorrector,
    FakeLiveSpeechToText,
    FakeSpeechToText,
    FakeTextToSpeech,
    FakeTranslator,
)

# 모델을 올리지 않은 서버(LOAD_MODELS 없음, 테스트·CI)의 models 필드. 네 모델 모두 null이다.
NO_MODELS = {
    "speech_recognition": None,
    "translation": None,
    "speech_synthesis": None,
    "typo_correction": None,
}


# 기본 테스트 서버(테스트 DB는 있고 모델은 없음)에서 200과 함께 database ok, 모델 모두 null, 오타
# 교정·동시통역 모두 불가로 답하는지 확인한다. 응답 전체를 비교해 필드가 늘거나 빠지는 변경도 잡는다.
async def test_health_reports_the_database_and_that_no_models_are_loaded(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "models": NO_MODELS,
        "typo_correction_ready": False,
        "live": False,
    }


# 네 모델이 모두 있으면 각 model_name을 그대로 보여 주고, 영어를 교정할 수 있는 교정기가 있으니
# typo_correction_ready, 자막용 인식(transcribe_live)이 있는 인식기와 번역기가 있으니 live가 true인지
# 확인한다.
async def test_health_names_the_loaded_models(client):
    models = PipelineModels(FakeLiveSpeechToText(), FakeTranslator(), FakeTextToSpeech(), FakeCorrector())
    app.dependency_overrides[get_models] = lambda: models
    body = (await client.get("/api/health")).json()
    assert body["models"] == {
        "speech_recognition": "fake-stt",
        "translation": "fake-mt",
        "speech_synthesis": "fake-tts",
        "typo_correction": "fake-corrector",
    }
    assert body["typo_correction_ready"] is True and body["live"] is True


# 일반 인식기(FakeSpeechToText)에는 transcribe_live가 없어 LiveSpeechToText가 아니므로, 번역기가 있어도 live는
# false여야 한다. 화면이 동시통역을 켜도 되는지 판단하는 값이라 인식 기능까지 따진다.
async def test_live_needs_live_recognition(client):
    app.dependency_overrides[get_models] = lambda: PipelineModels(FakeSpeechToText(), FakeTranslator())
    assert (await client.get("/api/health")).json()["live"] is False


# DB 질의가 예외를 던지면 503과 status·database "unavailable"로 답하는지, 그리고 예외 메시지("secret
# connection detail")는 응답에도 로그에도 없고 예외 종류(RuntimeError)만 로그에 남는지 확인한다. 연결 문자열
# 같은 내부 정보가 새지 않게 하는 규칙(services/errors.py의 describe)을 지키는지 보는 것이다.
# Unreachable은 execute만 가진 가짜 세션이고, broken_db는 get_db처럼 세션을 yield하는 비동기 생성기다.
async def test_health_is_503_without_the_database_and_logs_no_message(client, caplog):
    class Unreachable:
        async def execute(self, statement):
            raise RuntimeError("secret connection detail")

    async def broken_db():
        yield Unreachable()

    app.dependency_overrides[get_db] = broken_db
    with caplog.at_level(logging.WARNING):
        response = await client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable" and response.json()["database"] == "unavailable"
    assert "secret" not in response.text and "secret" not in caplog.text
    assert "RuntimeError" in caplog.text


# 모델 실행기에서 엔진 오류로 실패한 호출 하나(services/inference.run_model 경계를 실제로 지난다). 어댑터의
# 예외 경계처럼 엔진 예외를 원인으로 단 ModelError를 내고, 메시지의 "secret"은 로그·응답에 새면 안 되는
# 표식이다.
async def fail_one_model_call() -> None:
    def engine_call():
        try:
            raise RuntimeError("CUDA out of memory (secret input)")
        except RuntimeError as exc:
            raise ModelError("the model failed: RuntimeError") from exc

    with pytest.raises(ModelError):
        await run_model(engine_call)


# 이 파일의 모든 테스트에서 프로세스 전체가 쓰는 모델 상태 객체를 새 것으로 바꿔, 다른 테스트 파일이 쌓은
# 연속 실패 수와 섞이지 않게 0에서 시작한다(종료 함수는 쓰이지 않지만 os._exit 대신 기록용으로 둔다).
@pytest.fixture(autouse=True)
def fresh_model_health(monkeypatch) -> ModelHealth:
    fresh = ModelHealth(exit_process=[].append)
    monkeypatch.setattr(model_health, "MODEL_HEALTH", fresh)
    return fresh


# 모델을 올린 서버(LOAD_MODELS=true)를 흉내 낸다: 상태 확인이 읽는 설정만 바꾸고 실제 모델 대신 가짜 네 개를
# 끼운다. 돌려주는 함수로 설정(예: model_failure_limit)을 바꿀 수 있다.
@pytest.fixture
def model_server(monkeypatch):
    def serve(**settings) -> None:
        current = Settings(load_models=True, **settings)
        monkeypatch.setattr(health_router, "get_settings", lambda: current)

    models = PipelineModels(FakeLiveSpeechToText(), FakeTranslator(), FakeTextToSpeech(), FakeCorrector())
    app.dependency_overrides[get_models] = lambda: models
    serve()
    return serve


# 모델을 올린 서버는 기존 필드 뒤에 models_ok와 consecutive_model_failures를 더한다. 실패가 없으면 200·true·0.
async def test_a_model_server_also_reports_model_health(client, model_server):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "models": {
            "speech_recognition": "fake-stt",
            "translation": "fake-mt",
            "speech_synthesis": "fake-tts",
            "typo_correction": "fake-corrector",
        },
        "typo_correction_ready": True,
        "live": True,
        "models_ok": True,
        "consecutive_model_failures": 0,
    }


# 연속 실패가 MODEL_FAILURE_LIMIT(여기서는 2)보다 작으면 200, 닿으면 503·status "unavailable"·models_ok false.
# DB는 멀쩡하므로 database는 "ok"다. 이유(연속 실패 수와 한도)는 로그에만 남고 실패의 메시지는 어디에도 없다.
async def test_health_is_503_once_model_failures_in_a_row_reach_the_limit(client, model_server, caplog):
    model_server(model_failure_limit=2)
    await fail_one_model_call()
    response = await client.get("/api/health")
    assert response.status_code == 200 and response.json()["consecutive_model_failures"] == 1
    await fail_one_model_call()
    with caplog.at_level(logging.WARNING):
        response = await client.get("/api/health")
    body = response.json()
    assert response.status_code == 503
    assert body["status"] == "unavailable" and body["database"] == "ok"
    assert body["models_ok"] is False and body["consecutive_model_failures"] == 2
    assert "2 times in a row" in caplog.text
    assert "secret" not in response.text and "secret" not in caplog.text


# models_ok는 인식·번역 모델이 올라가 있어야 참이다(둘이 없으면 어떤 번역도 성공할 수 없다). 합성·교정은
# 선택이라 없어도 된다.
async def test_models_ok_needs_the_recognition_and_translation_models(client, model_server):
    app.dependency_overrides[get_models] = lambda: PipelineModels(FakeSpeechToText(), FakeTranslator())
    assert (await client.get("/api/health")).json()["models_ok"] is True
    app.dependency_overrides[get_models] = lambda: PipelineModels(None, None)
    response = await client.get("/api/health")
    assert response.status_code == 503 and response.json()["models_ok"] is False


# DB와 모델 상태는 따로 보인다: DB만 끊기면 503·database "unavailable"이고 models_ok는 그대로 true다.
async def test_a_model_server_without_the_database_is_503_with_models_ok(client, model_server):
    class Unreachable:
        async def execute(self, statement):
            raise RuntimeError("secret connection detail")

    async def broken_db():
        yield Unreachable()

    app.dependency_overrides[get_db] = broken_db
    response = await client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["database"] == "unavailable" and response.json()["models_ok"] is True


# LOAD_MODELS=false(기본, 테스트·CI)면 모델 실패가 한도를 넘게 쌓여도 응답은 이 기능이 생기기 전과 같다:
# 200이고 새 필드가 없다.
async def test_without_load_models_model_failures_do_not_change_the_response(client, fresh_model_health):
    for _ in range(6):
        await fail_one_model_call()
    assert fresh_model_health.consecutive_failures == 6
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "models": NO_MODELS,
        "typo_correction_ready": False,
        "live": False,
    }
