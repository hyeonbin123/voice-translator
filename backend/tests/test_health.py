# [파일 역할] 상태 확인 API(GET /api/health, routers/health.py)의 응답 계약을 검증한다(docs/api.md의 GET
# /api/health).
# [무엇을 확인하나] DB 연결 상태, 올라간 모델 이름, 오타 교정 준비 여부(typo_correction_ready), 동시통역 가능
# 여부(live), 그리고 DB에 닿지 못할 때 503을 내면서 오류 메시지를 응답·로그에 남기지 않는지.
# [왜 중요한가] Docker compose의 api 상태 검사가 이 경로를 쓰고, 응답 필드는 문서에 적힌 공개 계약이다.

import logging

# get_db·get_models를 dependency_overrides로 바꿔 'DB가 끊긴 서버'와 '모델이 올라간 서버'를 흉내 낸다. 가짜
# 모델은 model_name이 fake-* 로 정해져 있어 응답의 이름을 그대로 비교할 수 있다.
from app.dependencies import get_db, get_models
from app.main import app
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
