"""POST /api/translate/dialog: two people, one screen (T35), with fake models."""

import logging

import pytest

from app.dependencies import get_audio_store, get_models
from app.main import app
from app.services.audio_store import AudioStore
from app.services.interfaces import ModelError
from app.services.pipeline import PipelineModels
from tests.fakes import FakeSpeechToText, FakeTextToSpeech, FakeTranslator, silent_wav


@pytest.fixture
def stt(client, tmp_path):
    recognizer = FakeSpeechToText(text="recognized words")
    app.dependency_overrides[get_models] = lambda: PipelineModels(
        recognizer, FakeTranslator(), FakeTextToSpeech()
    )
    app.dependency_overrides[get_audio_store] = lambda: AudioStore(tmp_path / "audio")
    return recognizer


async def turn(client, headers, content=None, **form):
    files = {"audio": ("turn.wav", silent_wav(500) if content is None else content, "audio/wav")}
    return await client.post("/api/translate/dialog", headers=headers, files=files, data=form)


@pytest.mark.parametrize(("language", "target"), [("ko", "en"), ("en", "ko")])
async def test_the_detected_language_is_translated_into_the_other(
    client, auth_headers, stt, language, target
):
    stt.language = language
    response = await turn(client, auth_headers)
    assert response.status_code == 201
    body = response.json()
    assert (body["source_lang"], body["target_lang"]) == (language, target)
    assert body["translated_text"] == f"[{language}->{target}] recognized words"
    assert (body["language_confidence"], body["language_guessed"]) == (1.0, False)
    assert body["mode"] == "speech" and body["audio_id"] is not None
    assert response.headers["location"] == f"/api/history/{body['id']}"


async def test_an_unsure_turn_takes_the_language_opposite_to_the_previous_one(client, auth_headers, stt):
    stt.language, stt.confidence = "ko", 0.55
    body = (await turn(client, auth_headers, previous_lang="ko")).json()
    assert (body["source_lang"], body["target_lang"], body["language_guessed"]) == ("en", "ko", True)
    assert body["language_confidence"] == 0.55


async def test_an_unsure_first_turn_keeps_the_detected_language(client, auth_headers, stt):
    stt.language, stt.confidence = "en", 0.55
    body = (await turn(client, auth_headers)).json()
    assert (body["source_lang"], body["language_guessed"]) == ("en", False)


async def test_a_sure_turn_ignores_the_previous_language(client, auth_headers, stt):
    stt.language, stt.confidence = "ko", 0.99
    body = (await turn(client, auth_headers, previous_lang="ko")).json()
    assert (body["source_lang"], body["language_guessed"]) == ("ko", False)


async def test_failures_answer_as_the_speech_api_does(client, auth_headers, stt, caplog):
    undecodable = await turn(client, auth_headers, content=b"not audio at all")
    assert (undecodable.status_code, undecodable.json()["detail"]) == (422, "Audio could not be decoded")
    stt.error = ModelError("secret detail")
    with caplog.at_level(logging.WARNING):
        failed = await turn(client, auth_headers)
    assert (failed.status_code, failed.json()["detail"]) == (503, "Translation service is unavailable")
    assert "secret" not in caplog.text


async def test_an_unknown_previous_language_is_rejected(client, auth_headers, stt):
    assert (await turn(client, auth_headers, previous_lang="ja")).status_code == 422


async def test_a_turn_needs_login(client, stt):
    assert (await turn(client, {})).status_code == 401


async def test_without_recognition_the_service_is_unavailable(client, auth_headers):
    app.dependency_overrides[get_models] = lambda: PipelineModels(None, FakeTranslator())
    assert (await turn(client, auth_headers)).status_code == 503
