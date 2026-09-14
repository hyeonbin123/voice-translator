import logging

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

NO_MODELS = {
    "speech_recognition": None,
    "translation": None,
    "speech_synthesis": None,
    "typo_correction": None,
}


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


async def test_live_needs_live_recognition(client):
    app.dependency_overrides[get_models] = lambda: PipelineModels(FakeSpeechToText(), FakeTranslator())
    assert (await client.get("/api/health")).json()["live"] is False


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
