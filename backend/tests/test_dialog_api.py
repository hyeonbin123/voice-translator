"""POST /api/translate/dialog: two people, one screen (T35), with fake models."""

# [파일 역할] 두 사람 대화 API(POST /api/translate/dialog, routers/translate.py의 translate_dialog)를 가짜
# 모델로 검증한다.
# [무엇을 확인하나] 판별된 언어를 상대 언어로 번역하는지, 판별 확신이 문턱(dialog_language_threshold, 기본
# 0.6) 아래일 때 직전 마디의 반대 언어로 처리하는지(services/dialog.py의 choose_language), 실패·인증·잘못된
# 입력에 음성 번역 API와 같은 답을 주는지.
# [방법] conftest.py의 client(테스트 DB를 쓰는 ASGI 클라이언트)와 auth_headers를 쓰고, 모델과 음성 저장소는
# dependency_overrides로 tests/fakes.py의 가짜 구현과 임시 폴더로 바꾼다. GPU나 모델 다운로드 없이 돈다.

import logging

import pytest

# get_models·get_audio_store: 라우터가 Depends로 받는 의존성. 테스트에서 가짜 모델·임시 저장소로 바꿔 끼우려고
# 가져온다.
# fakes: 언어 판별 결과(language·confidence)와 실패(error)를 테스트가 정할 수 있는 가짜 음성 인식·번역·합성.
from app.dependencies import get_audio_store, get_models
from app.main import app
from app.services.audio_store import AudioStore
from app.services.interfaces import ModelError
from app.services.pipeline import PipelineModels
from tests.fakes import FakeSpeechToText, FakeTextToSpeech, FakeTranslator, silent_wav


# 가짜 모델 묶음을 앱에 끼우고, 테스트가 판별 결과를 바꿀 수 있도록 가짜 음성 인식기를 돌려준다.
# client fixture에 의존하므로 client가 끝날 때 dependency_overrides가 원래대로 돌아간다(conftest.py). 번역
# 음성은 tmp_path 아래에 저장돼 테스트마다 지워진다.
@pytest.fixture
def stt(client, tmp_path):
    recognizer = FakeSpeechToText(text="recognized words")
    app.dependency_overrides[get_models] = lambda: PipelineModels(
        recognizer, FakeTranslator(), FakeTextToSpeech()
    )
    app.dependency_overrides[get_audio_store] = lambda: AudioStore(tmp_path / "audio")
    return recognizer


# 한 마디를 올리는 도우미. content를 주지 않으면 0.5초 무음 WAV를 보낸다. form에는 previous_lang 같은 폼
# 필드를 넘긴다.
async def turn(client, headers, content=None, **form):
    files = {"audio": ("turn.wav", silent_wav(500) if content is None else content, "audio/wav")}
    return await client.post("/api/translate/dialog", headers=headers, files=files, data=form)


# 확신이 높은(가짜 기본값 1.0) 판별에서 한국어는 영어로, 영어는 한국어로 번역되는지 두 방향 모두 확인한다.
# 응답에 확신·추측 여부(language_guessed=False), 음성 모드와 번역 음성 id, 201 응답의 Location
# 헤더(/api/history/{id})까지 음성 번역과 같은 계약이 지켜지는지도 본다.
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


# 판별 확신 0.55(문턱 0.6 미만)이고 직전 마디가 한국어면, 판별 결과(ko)를 버리고 '번갈아 말한다'고 보아 영어로
# 처리하는지 확인한다(docs/experiments.md 10절의 규칙). 추측했다는 표시와 원래 확신 값도 응답에 남아야 화면이
# 방향 바꾸기 단추를 띄울 수 있다.
async def test_an_unsure_turn_takes_the_language_opposite_to_the_previous_one(client, auth_headers, stt):
    stt.language, stt.confidence = "ko", 0.55
    body = (await turn(client, auth_headers, previous_lang="ko")).json()
    assert (body["source_lang"], body["target_lang"], body["language_guessed"]) == ("en", "ko", True)
    assert body["language_confidence"] == 0.55


# 첫 마디(previous_lang 없음)는 확신이 낮아도 기댈 직전 언어가 없으므로 판별 결과를 그대로 쓰고 추측으로
# 표시하지 않는지 확인한다.
async def test_an_unsure_first_turn_keeps_the_detected_language(client, auth_headers, stt):
    stt.language, stt.confidence = "en", 0.55
    body = (await turn(client, auth_headers)).json()
    assert (body["source_lang"], body["language_guessed"]) == ("en", False)


# 확신이 문턱 이상이면 직전 마디와 같은 언어라도 판별 결과를 따른다. 같은 사람이 연달아 말하는 경우를 막지
# 않는지 확인한다.
async def test_a_sure_turn_ignores_the_previous_language(client, auth_headers, stt):
    stt.language, stt.confidence = "ko", 0.99
    body = (await turn(client, auth_headers, previous_lang="ko")).json()
    assert (body["source_lang"], body["language_guessed"]) == ("ko", False)


# 실패 응답이 음성 번역 API와 같은지 확인한다: 소리로 읽을 수 없는 업로드는 422 "Audio could not be decoded",
# 모델 오류는 503 "Translation service is unavailable"(routers/translate.py의 failures_as_http).
# 모델 오류의 메시지("secret detail")가 로그에 남지 않는지도 caplog로 본다. 로그에는 예외 종류와 위치만
# 남긴다는 원칙(services/errors.py의 describe)이 깨지면 입력 내용이 로그로 새기 때문이다.
async def test_failures_answer_as_the_speech_api_does(client, auth_headers, stt, caplog):
    undecodable = await turn(client, auth_headers, content=b"not audio at all")
    assert (undecodable.status_code, undecodable.json()["detail"]) == (422, "Audio could not be decoded")
    stt.error = ModelError("secret detail")
    with caplog.at_level(logging.WARNING):
        failed = await turn(client, auth_headers)
    assert (failed.status_code, failed.json()["detail"]) == (503, "Translation service is unavailable")
    assert "secret" not in caplog.text


# previous_lang은 ko·en만 허용한다. 다른 값(ja)은 폼 검증에서 422로 막히는지 확인한다.
async def test_an_unknown_previous_language_is_rejected(client, auth_headers, stt):
    assert (await turn(client, auth_headers, previous_lang="ja")).status_code == 422


# 인증 헤더 없이는 401이다. 번역 기록과 음성을 사용자별로 저장하므로 로그인이 필수다.
async def test_a_turn_needs_login(client, stt):
    assert (await turn(client, {})).status_code == 401


# 음성 인식 모델이 없으면(LOAD_MODELS 없이 띄운 서버 등) 언어 판별 기능이 없으므로 503으로 답하는지 확인한다.
# 라우터는 models.stt가 LanguageDetector가 아닐 때 판별 전에 503을 낸다.
async def test_without_recognition_the_service_is_unavailable(client, auth_headers):
    app.dependency_overrides[get_models] = lambda: PipelineModels(None, FakeTranslator())
    assert (await turn(client, auth_headers)).status_code == 503
