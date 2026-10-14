"""동시통역 자막 웹소켓을 가짜 모델과 Starlette 테스트 클라이언트로 검증한다."""

# [파일 역할] 웹소켓 WS /api/translate/live(routers/live.py)를 가짜 모델과 실제 테스트 DB로, Starlette 테스트
# 클라이언트를 통해 검증한다. 세 모드를 모두 다룬다: 동시통역(live, 말하는 동안 자막), 대화
# 모드(conversation), 두 사람 대화(dialog). 프로토콜은 docs/api.md의 'WS /api/translate/live' 절과 같다.
# [무엇을 확인하나] 자막 갱신과 진한 부분(stable), pause에서 미리 시작한 최종 처리와 resume·cancel·연결 끊김
# 때 버리기, end에서 기록 하나 저장, 실패를 마디 단위 error로 알리고 연결 유지, 잘못된 시작·메시지의 닫힘
# 코드(4401·4422·4503), 토큰 만료, 갱신 간격과 전용 스레드, 대화 모드 결과가 HTTP 음성 번역과 같은지, 두 사람
# 대화의 언어 판별과 직전 마디 언어.
# [왜 중요한가] 미리 처리는 말이 이어지면 버려져야 하고, 버린 결과가 기록·파일로 남거나 순서가 꼬이면 사용자가
# 보는 결과와 저장된 기록이 달라진다. 실제 서비스의 지연 수치(README)도 이 경로에서 나왔다.

import asyncio
import logging
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

# jwt: 1초 뒤 만료되는 토큰을 직접 만든다. NullPool: 테스트 클라이언트가 자기 이벤트 루프에서 앱을 돌리므로,
# 루프 사이에 연결을 공유하지 않는 엔진을 만든다. TestClient·WebSocketDisconnect: 웹소켓 연결과 닫힘 코드
# 확인.
import jwt
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# 앱의 의존성(get_sessions·get_models·get_audio_store·get_live_options·get_db)을 바꿔 끼워 테스트 DB, 가짜
# 모델, 임시 음성 폴더, 빠른 갱신 설정을 쓴다. wav_from_pcm은 대화 모드 결과를 HTTP 업로드와 견줄 때 같은
# WAV를 만들려고 가져온다.
from app.config import get_settings
from app.core.security import create_token
from app.db.session import ENGINE_OPTIONS
from app.dependencies import get_audio_store, get_db, get_models, get_sessions
from app.main import app
from app.models import Translation
from app.routers.live import LiveOptions, get_live_options
from app.services.audio_store import AudioStore
from app.services.interfaces import ModelError
from app.services.live import wav_from_pcm
from app.services.pipeline import PipelineModels
from tests.fakes import FakeLiveSpeechToText, FakeSpeechToText, FakeTextToSpeech, FakeTranslator

# 16 kHz 16비트 소리 1초(32000바이트). 가짜 실시간 인식기는 0.1초(3200바이트)마다 단어 하나를 듣는다고 치므로
# 1초는 열 단어다. words(바이트 수)는 그 소리에서 가짜가 들을 글자를 계산한다.
SECOND = b"\x01\x00" * 16_000
words = FakeLiveSpeechToText.words


# 웹소켓 시험 환경: 풀 없는 엔진의 세션 공장, 가짜 모델 묶음, 임시 음성 폴더, 갱신 간격 0초·시작 대기 2초
# 설정을 앱에 끼우고, 테스트 클라이언트·모델·로그인 토큰을 돌려준다. 끝나면 의존성을 원래대로 돌리고 엔진을
# 닫는다.
# 웹소켓은 말하는 동안 오래 열려 있으므로 DB 연결을 계속 잡지 않도록, 필요할 때만 세션을 여는 get_sessions를
# 쓴다(dependencies.py).
@pytest.fixture
def live(migrated_database, user, tmp_path):
    # 테스트 클라이언트는 자기 이벤트 루프에서 앱을 돌린다. 풀 없는 엔진은 그 루프와 테스트 쪽 루프 사이에
    # 연결을 나눠 쓰지 않는다.
    engine = create_async_engine(migrated_database, poolclass=NullPool, **ENGINE_OPTIONS)
    models = PipelineModels(FakeLiveSpeechToText(), FakeTranslator(), FakeTextToSpeech())
    previous = app.dependency_overrides.copy()
    app.dependency_overrides.update(
        {
            get_sessions: lambda: async_sessionmaker(engine, expire_on_commit=False),
            get_models: lambda: models,
            get_audio_store: lambda: AudioStore(tmp_path / "audio"),
            get_live_options: lambda: LiveOptions(interval_s=0.0, start_timeout_s=2.0),
        }
    )
    try:
        yield SimpleNamespace(client=TestClient(app), models=models, token=create_token(user.id, "access"))
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)
        engine.sync_engine.dispose()


# 첫 메시지(start: 토큰·언어)를 보내고 ready를 받는 도우미. 브라우저 WebSocket은 인증 헤더를 붙일 수 없고
# 주소의 토큰은 로그에 남으므로 토큰은 첫 메시지로 보낸다(docs/api.md).
def start(ws, token, **changes):
    ws.send_json({"type": "start", "token": token, "source_lang": "ko", "target_lang": "en", **changes})
    assert ws.receive_json() == {"type": "ready"}


# 지정한 종류의 메시지가 올 때까지 받은 메시지를 모두 돌려준다.
def until(ws, kind):
    """이 종류의 첫 메시지까지(그 메시지 포함) 받은 모든 메시지."""
    seen = [ws.receive_json()]
    while seen[-1]["type"] != kind:
        seen.append(ws.receive_json())
    return seen


# 서버가 연결을 닫을 때까지 받다가 닫힘 코드를 돌려준다.
def closed_with(ws) -> int:
    with pytest.raises(WebSocketDisconnect) as closed:
        while True:
            ws.receive_json()
    return closed.value.code


# DB에 저장된 번역 기록 수.
async def records(db_session) -> int:
    return await db_session.scalar(select(func.count()).select_from(Translation))


# 기본 흐름: 소리 1초를 보내면 원문 자막(source)과 번역 자막(translation)이 오고, 첫 결과는 비교할 앞 결과가
# 없어 진한 부분이 0이다. 1초를 더 보내면 두 결과가 같은 앞 열 단어가 진해지고, 번역은 늘 0이다(번역 앞부분이
# 자주 바뀌어 최종 전까지 흐리게 둠). pause → end면 전체 소리의 최종 결과(final)가 음성 모드 기록과 번역 음성
# id와 함께 오고, 기록은 정확히 하나다.
async def test_updates_follow_the_speech_and_the_end_saves_one_record(live, db_session):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        first = until(ws, "translation")
        assert first == [
            {"type": "source", "id": 1, "text": words(len(SECOND)), "stable": 0},
            {"type": "translation", "id": 1, "text": f"[ko->en] {words(len(SECOND))}", "stable": 0},
        ]
        ws.send_bytes(SECOND)
        second = until(ws, "translation")
        # 두 결과가 공유하는 앞부분이 단어 경계까지 확정(stable)된다.
        assert second[-2] == {
            "type": "source",
            "id": 1,
            "text": words(2 * len(SECOND)),
            "stable": len(words(len(SECOND))),
        }
        assert second[-1] == {
            "type": "translation",
            "id": 1,
            "text": f"[ko->en] {words(2 * len(SECOND))}",
            "stable": 0,
        }
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        final = until(ws, "final")[-1]
    assert final["id"] == 1
    assert final["result"]["source_text"] == words(2 * len(SECOND))
    assert final["result"]["translated_text"] == f"[ko->en] {words(2 * len(SECOND))}"
    assert final["result"]["mode"] == "speech" and final["result"]["audio_id"] is not None
    assert await records(db_session) == 1


# pause로 최종 처리를 미리 시작했어도 resume(1초가 되기 전에 다시 말함)이 오면 그 결과를 버리고, 다음
# pause에서 전체 2초 소리로 다시 처리한다. 저장되는 것은 최종 결과 하나뿐이다.
async def test_speech_after_a_pause_drops_the_prepared_final(live, db_session):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "resume", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        final = until(ws, "final")[-1]
    assert final["result"]["source_text"] == words(2 * len(SECOND))
    assert await records(db_session) == 1


# pause 없이 end가 오면 받은 소리 전체로 처리하고, 브라우저가 정한 마디 id(7)를 그대로 돌려준다.
async def test_an_end_without_a_pause_uses_all_the_audio(live, db_session):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 7})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "end", "id": 7})
        final = until(ws, "final")[-1]
    assert (final["id"], final["result"]["source_text"]) == (7, words(len(SECOND)))


# 흔적을 남기지 않아야 하는 경우들: 마디 밖의 소리는 버리고, cancel된 마디 1의 뒤늦은 end와 지금 마디가 아닌
# id(9)의 pause는 무시한다. 마디 1에서는 자막만 오고 결과는 없으며, 마디 2만 그 소리(0.5초)로 저장된다.
async def test_cancelled_stale_and_stray_messages_leave_no_trace(live, db_session):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_bytes(SECOND)  # 마디 밖의 소리
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "cancel", "id": 1})
        ws.send_json({"type": "end", "id": 1})  # 이미 버린 마디
        ws.send_json({"type": "utterance", "id": 2})
        ws.send_json({"type": "pause", "id": 9})  # 지금 마디가 아님
        ws.send_bytes(SECOND[:16_000])
        ws.send_json({"type": "end", "id": 2})
        messages = until(ws, "final")
    assert all(message["id"] != 1 or message["type"] in ("source", "translation") for message in messages)
    assert messages[-1]["id"] == 2
    assert messages[-1]["result"]["source_text"] == words(16_000)
    assert await records(db_session) == 1


# 말소리가 없는 마디는 그 마디만 error "No speech was recognized"(음성 번역 API와 같은 문구)로 알리고 기록을
# 남기지 않으며, 연결은 그대로라 다음 마디는 정상 저장된다.
async def test_an_utterance_without_speech_is_an_error_and_no_record(live, db_session):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND[:1000])  # 가짜 단어 하나보다 짧다
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        messages = until(ws, "error")
        # 연결은 그대로 이어져 다음 마디를 받는다.
        ws.send_json({"type": "utterance", "id": 2})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "end", "id": 2})
        final = until(ws, "final")[-1]
    assert messages == [{"type": "error", "id": 1, "detail": "No speech was recognized"}]
    assert final["id"] == 2
    assert await records(db_session) == 1


# 한 마디 소리가 30초를 넘으면 end에서 "Audio is longer than 30 seconds" 오류이고 기록이 없다(HTTP 업로드와
# 같은 30초 제한).
async def test_more_than_thirty_seconds_is_an_error(live, db_session):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        for _ in range(31):
            ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        error = until(ws, "error")[-1]
    assert error == {"type": "error", "id": 1, "detail": "Audio is longer than 30 seconds"}
    assert await records(db_session) == 0


# 합성이 실패해도 번역은 저장하고 final에 tts_error를 싣는다(HTTP 경로와 같은 규칙).
async def test_failed_synthesis_still_saves_the_translation(live, db_session):
    app.dependency_overrides[get_models] = lambda: replace(live.models, tts=FakeTextToSpeech(fail=True))
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "end", "id": 1})
        result = until(ws, "final")[-1]["result"]
    assert (result["audio_id"], result["tts_error"]) == (None, "Speech synthesis failed")
    assert await records(db_session) == 1


# 자막 갱신용 인식이 실패하면 아무것도 보내지 않고, 최종 인식이 실패하면 그 마디가 error "Translation service
# is unavailable"이 된다. 로그에는 예외 종류(ModelError)만 있고 메시지("secret ...")는 없으며, 기록도 없다.
async def test_a_model_failure_is_reported_without_its_message(live, db_session, caplog):
    stt = FakeLiveSpeechToText(
        error=ModelError("secret spoken words"), live_error=ModelError("secret live words")
    )
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    with caplog.at_level(logging.WARNING), live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        messages = until(ws, "error")
    # 실패한 자막 갱신은 아무것도 보이지 않고, 실패한 최종 처리는 그 마디의 error가 된다.
    assert messages == [{"type": "error", "id": 1, "detail": "Translation service is unavailable"}]
    assert "ModelError" in caplog.text and "secret" not in caplog.text
    assert await records(db_session) == 0


# 잘못된 첫 메시지의 닫힘 코드: 틀린 토큰이나 refresh 토큰은 4401(인증), 같은 언어끼리·지원하지 않는
# 언어·start가 아닌 메시지·JSON이 아닌 글·바이너리는 4422(계약 위반).
@pytest.mark.parametrize(
    ("first", "code"),
    [
        ({"type": "start", "token": "not a token", "source_lang": "ko", "target_lang": "en"}, 4401),
        ({"type": "start", "token": "REFRESH", "source_lang": "ko", "target_lang": "en"}, 4401),
        ({"type": "start", "token": "ACCESS", "source_lang": "ko", "target_lang": "ko"}, 4422),
        ({"type": "start", "token": "ACCESS", "source_lang": "ja", "target_lang": "en"}, 4422),
        ({"type": "utterance", "id": 1}, 4422),
        ("not json", 4422),
        (b"\x00\x00", 4422),
    ],
)
async def test_a_bad_start_closes_the_connection(live, user, first, code):
    tokens = {"ACCESS": live.token, "REFRESH": create_token(user.id, "refresh")}
    with live.client.websocket_connect("/api/translate/live") as ws:
        if isinstance(first, bytes):
            ws.send_bytes(first)
        elif isinstance(first, str):
            ws.send_text(first)
        else:
            if "token" in first:
                first = {**first, "token": tokens.get(first["token"], first["token"])}
            ws.send_json(first)
        assert closed_with(ws) == code


# 시작 대기 시간(여기서는 0.1초) 안에 start가 오지 않으면 4422로 닫는다. 서비스 기본값은 10초다(docs/api.md).
async def test_no_start_message_in_time_closes_the_connection(live):
    app.dependency_overrides[get_live_options] = lambda: LiveOptions(interval_s=0.0, start_timeout_s=0.1)
    with live.client.websocket_connect("/api/translate/live") as ws:
        assert closed_with(ws) == 4422


# 서명은 맞지만 DB에 없는 사용자의 토큰은 4401이다.
async def test_an_unknown_user_is_unauthorized(live):
    with live.client.websocket_connect("/api/translate/live") as ws:
        ws.send_json(
            {
                "type": "start",
                "token": create_token(uuid4(), "access"),
                "source_lang": "ko",
                "target_lang": "en",
            }
        )
        assert closed_with(ws) == 4401


# 동시통역은 자막용 인식(transcribe_live)이 필요하므로, 인식 모델이 없거나 일반 인식기뿐이면 4503으로 닫는다.
@pytest.mark.parametrize("stt", [None, FakeSpeechToText()])
async def test_without_live_recognition_the_service_is_unavailable(live, stt):
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    with live.client.websocket_connect("/api/translate/live") as ws:
        ws.send_json({"type": "start", "token": live.token, "source_lang": "ko", "target_lang": "en"})
        assert closed_with(ws) == 4503


# 시작 뒤에도 계약에 없는 제어 메시지(JSON 아님, id가 문자열, 모르는 type)는 4422로 연결을 닫는다.
@pytest.mark.parametrize(
    "message", ["not json", '{"type": "utterance", "id": "1"}', '{"type": "stop", "id": 1}']
)
async def test_a_bad_control_message_closes_the_connection(live, message):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_text(message)
        assert closed_with(ws) == 4422


# 연결 중에 액세스 토큰이 만료되면 만료 뒤 첫 메시지에서 4401로 닫는다(docs/api.md). 화면은 토큰을 새로 받아
# 다시 연결한다. 1초 뒤 만료되는 토큰을 앱과 같은 비밀키로 직접 만든다.
async def test_the_connection_closes_once_the_token_expires(live, user):
    now = datetime.now(UTC)
    claims = {
        "sub": str(user.id),
        "type": "access",
        "iat": now,
        "exp": now + timedelta(seconds=1),
        "jti": "j",
    }
    token = jwt.encode(claims, get_settings().jwt_secret_key.get_secret_value(), algorithm="HS256")
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, token)
        time.sleep(1.2)
        ws.send_json({"type": "utterance", "id": 1})
        assert closed_with(ws) == 4401


# 첫 최종 인식을 모델 스레드 안에서 풀어 줄 때까지 붙잡고, 자막 갱신이 시작된 시각을 기록하는 가짜 인식기.
# 빠른 가짜로는 서버가 모델 호출 도중인 순간을 잡을 수 없어서 만들었다.
class HeldFinal(FakeLiveSpeechToText):
    """첫 최종 인식을 풀어 줄 때까지 모델 스레드 안에 붙잡고, 각 자막 갱신이 시작된 시각을 적는다(위의
    빠른 가짜로는 서버가 호출 도중인 순간을 잡을 수 없다)."""

    def __init__(self, hold: bool = True) -> None:
        super().__init__()
        self.entered, self.release, self.left = threading.Event(), threading.Event(), threading.Event()
        self.hold = hold
        self.update_starts: list[float] = []

    def transcribe(self, audio, language):
        if self.hold:
            self.hold = False
            self.entered.set()
            try:
                assert self.release.wait(5)
            finally:
                self.left.set()
        return super().transcribe(audio, language)

    def transcribe_live(self, pcm, language):
        self.update_starts.append(time.perf_counter())
        return super().transcribe_live(pcm, language)


# 미리 처리한 최종 인식이 아직 모델 안에 있을 때 resume과 새 소리가 와도, 그 결과는 버려지고 두 번째 pause의
# 2초 소리 결과만 저장되는지 확인한다. 붙잡힌 호출이 resume 전후 언제 끝나든 결과가 같아야 한다.
async def test_speech_while_the_final_is_prepared_drops_that_final(live, db_session):
    stt = HeldFinal()
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 1})
        assert stt.entered.wait(5)  # 미리 처리한 최종 인식이 인식 모델 안에 있다
        ws.send_json({"type": "resume", "id": 1})
        ws.send_bytes(SECOND)
        stt.release.set()
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        final = until(ws, "final")[-1]
    assert final["result"]["source_text"] == words(2 * len(SECOND))
    assert await records(db_session) == 1


# 회귀 테스트(커밋 7fa678e): end를 보낸 뒤 최종 처리가 모델 안에 있는 동안 연결이 끊기면, 서버는 그 처리를
# 취소하고 기록도 음성 파일도 남기지 않아야 한다. 연결을 닫고 잠시 기다린 뒤 붙잡은 호출을 풀어 주고, DB와
# 음성 폴더가 비었는지 본다.
async def test_a_connection_closed_while_the_final_is_prepared_saves_nothing(live, db_session, tmp_path):
    stt = HeldFinal()
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        assert stt.entered.wait(5)
    time.sleep(0.2)  # 서버가 연결 끊김을 보고 최종 처리를 취소할 시간
    stt.release.set()
    assert stt.left.wait(5)
    await asyncio.sleep(0.3)
    assert await records(db_session) == 0
    assert not list((tmp_path / "audio").glob("*.wav"))


# 자막 갱신은 한 마디에서 한 번에 하나만 돌고, 두 갱신의 시작은 갱신 간격(여기서는 0.3초, 서비스는
# LIVE_UPDATE_MS) 이상 떨어져야 한다. 실시간보다 빠르게 소리를 보내도 갱신 시작 간격이 0.25초(여유 포함)
# 아래로 줄지 않는지 확인한다.
async def test_updates_start_at_least_the_interval_apart(live):
    stt = HeldFinal(hold=False)
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    app.dependency_overrides[get_live_options] = lambda: LiveOptions(interval_s=0.3)
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        began = time.perf_counter()
        while time.perf_counter() - began < 1.6:
            ws.send_bytes(SECOND[:3200])  # 0.1초 소리를 실시간보다 빠르게 보낸다
            time.sleep(0.05)
        ws.send_json({"type": "end", "id": 1})
        until(ws, "final")
    starts = stt.update_starts
    assert len(starts) >= 4
    assert all(later - earlier >= 0.25 for earlier, later in zip(starts, starts[1:], strict=False))


# 저장(DB commit)이 실패하면 그 마디가 error "The translation could not be saved"가 되고, 기록도 이미 쓴 음성
# 파일도 남지 않는지 확인한다. 로그에는 예외 종류만 있고 메시지와 토큰은 없어야 한다. 첫 세션은 시작 때 사용자
# 확인에 쓰이므로 두 번째 세션부터 commit을 실패시킨다.
async def test_a_failed_save_is_an_error_and_leaves_no_file(live, db_session, tmp_path, caplog):
    factory = app.dependency_overrides[get_sessions]()
    opened = []

    def sessions():
        session = factory()
        opened.append(session)
        if len(opened) > 1:  # 첫 세션은 시작할 때 사용자를 확인하는 데 쓴다

            async def fail_commit():
                raise RuntimeError("secret database words")

            session.commit = fail_commit
        return session

    app.dependency_overrides[get_sessions] = lambda: sessions
    with caplog.at_level(logging.WARNING), live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "end", "id": 1})
        error = until(ws, "error")[-1]
    assert error == {"type": "error", "id": 1, "detail": "The translation could not be saved"}
    assert await records(db_session) == 0
    assert not list((tmp_path / "audio").glob("*.wav"))
    assert "RuntimeError" in caplog.text
    assert "secret" not in caplog.text and live.token not in caplog.text


# LIVE_UPDATE_THREAD를 켜면 자막 갱신은 'live' 스레드에서, 최종 인식은 'model' 스레드에서 도는지 스레드
# 이름으로 확인한다(커밋 61845a1의 후보 설정). 갱신이 최종 처리 뒤에 줄 서지 않게 하려는
# 설정이다(inference.run_live_model docstring).
async def test_updates_can_run_on_a_thread_of_their_own(live):
    calls: list[tuple[str, str]] = []

    class Named(FakeLiveSpeechToText):
        def transcribe_live(self, pcm, language):
            calls.append(("update", threading.current_thread().name))
            return super().transcribe_live(pcm, language)

        def transcribe(self, audio, language):
            calls.append(("final", threading.current_thread().name))
            return super().transcribe(audio, language)

    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=Named())
    app.dependency_overrides[get_live_options] = lambda: LiveOptions(interval_s=0.0, update_thread=True)
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        until(ws, "translation")
        ws.send_json({"type": "end", "id": 1})
        until(ws, "final")
    updates = [name for kind, name in calls if kind == "update"]
    finals = [name for kind, name in calls if kind == "final"]
    assert updates and all(name.startswith("live") for name in updates)
    assert finals and all(name.startswith("model") for name in finals)


# 여기부터 대화 모드와 두 사람 대화(T77, docs/experiments.md 11절): 쉼에서 미리 처리하고 끝에서 저장하는 같은
# 흐름을 자막 없이 쓴다. 이 모드들이 예전에 말 끝 뒤 HTTP로 올리던 클립에 쓰인다.


# HTTP 경로도 웹소켓과 같은 테스트 DB 세션 공장을 쓰게 하는 get_db 대체. 대화 모드 결과를 같은 DB에서 HTTP
# 음성 번역 결과와 견주려고 둔다.
def http_db():
    """HTTP 경로의 DB 세션을 테스트 DB로 돌려, 웹소켓 결과와 견줄 수 있게 한다."""
    sessions = app.dependency_overrides[get_sessions]()

    async def db():
        async with sessions() as session:
            yield session

    return db


# 두 사람 대화용 가짜 인식기: 마디마다 정해 둔 (언어, 확신)을 차례로 판별 결과로 내고, 첫 최종 인식을 붙잡아
# 둘 수 있다.
class Turns(FakeLiveSpeechToText):
    """두 사람 대화의 마디마다 목록에서 차례로 언어를 판별하고, 첫 최종 인식을 풀어 줄 때까지 모델 스레드
    안에 붙잡아 둘 수 있다."""

    def __init__(self, detections: list[tuple[str, float]], hold: bool = False) -> None:
        super().__init__()
        self.detections = list(detections)
        self.hold = hold
        self.entered, self.release = threading.Event(), threading.Event()

    def detect_language(self, audio):
        return self.detections.pop(0)

    def transcribe(self, audio, language):
        if self.hold:
            self.hold = False
            self.entered.set()
            assert self.release.wait(5)
        return super().transcribe(audio, language)


# 한 마디(소리 → pause → end)를 보내고 final을 받는 도우미.
def turn(ws, number, audio=SECOND):
    ws.send_json({"type": "utterance", "id": number})
    ws.send_bytes(audio)
    ws.send_json({"type": "pause", "id": number})
    ws.send_json({"type": "end", "id": number})
    return until(ws, "final")[-1]


# 두 사람 대화 모드로 시작하는 도우미. 언어는 서버가 판별하므로 보내지 않고, 다시 연결할 때만 직전 마디 정보를
# 넣는다.
def start_dialog(ws, token, **fields):
    ws.send_json({"type": "start", "token": token, "mode": "dialog", **fields})
    assert ws.receive_json() == {"type": "ready"}


# 웹소켓 결과와 HTTP 결과가 같아야 하는 필드. 기록 id·처리 시간·음성 바이트는 호출마다 달라 빼고, 합성 모델은
# 따로 비교한다.
SAME = ("mode", "source_lang", "target_lang", "source_text", "translated_text", "stt_model", "mt_model")


# 대화 모드는 자막을 보내지 않고 final 하나만 보내며, 그 결과가 같은 소리를 POST /api/translate/speech에
# 올렸을 때와 같은지 확인한다. 웹소켓으로 미리 처리해도 업로드와 결과가 같다는 것이 이 방식을 채택한
# 조건이었다(docs/experiments.md 11절). 두 경로 모두 기록을 남겨 총 2개다.
async def test_conversation_mode_sends_no_updates_and_answers_as_the_speech_api(live, db_session):
    app.dependency_overrides[get_db] = http_db()
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token, mode="conversation")
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        messages = until(ws, "final")
    assert [message["type"] for message in messages] == ["final"]
    spoken = messages[-1]["result"]
    uploaded = live.client.post(
        "/api/translate/speech",
        headers={"Authorization": f"Bearer {live.token}"},
        files={"audio": ("conversation.wav", wav_from_pcm(2 * SECOND), "audio/wav")},
        data={"source_lang": "ko", "target_lang": "en"},
    )
    assert uploaded.status_code == 201
    assert {key: spoken[key] for key in SAME} == {key: uploaded.json()[key] for key in SAME}
    assert spoken["audio_id"] is not None and spoken["tts_model"] == uploaded.json()["tts_model"]
    assert await records(db_session) == 2


# 대화 모드에서도 resume이 오면 미리 처리한 결과를 버리고 다음 pause에서 다시 처리한다. 대화 모드는 자막용
# 인식을 한 번도 부르지 않는다(update_starts가 비어 있음).
async def test_conversation_mode_drops_a_prepared_final_when_speech_resumes(live, db_session):
    stt = HeldFinal()
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token, mode="conversation")
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 1})
        assert stt.entered.wait(5)
        ws.send_json({"type": "resume", "id": 1})
        ws.send_bytes(SECOND)
        stt.release.set()
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        messages = until(ws, "final")
    assert [message["type"] for message in messages] == ["final"]
    assert messages[-1]["result"]["source_text"] == words(2 * len(SECOND))
    assert await records(db_session) == 1
    assert not stt.update_starts  # 자막용 인식은 한 번도 없었다


# 대화 모드는 자막을 보내지 않으므로 자막용 인식이 없는 일반 인식기로도 동작한다(docs/api.md).
async def test_conversation_mode_needs_no_live_recognition(live):
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=FakeSpeechToText(text="said"))
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token, mode="conversation")
        final = turn(ws, 1)
    assert final["result"]["translated_text"] == "[ko->en] said"


# 두 사람 대화: 첫 마디는 확신 1.0이라 한국어 → 영어. 둘째 마디는 확신 0.55(문턱 0.6 미만)라 직전
# 마디(한국어)의 반대인 영어로 처리하고 추측으로 표시한다. 화면이 둘째 마디의 방향을 한국어로 바꿨다는
# previous 메시지(마지막으로 저장한 마디 id 2)는 따르고, 더 오래된 마디 1에 대한 메시지는 무시하므로, 셋째
# 마디(확신 낮음)는 한국어의 반대인 영어가 된다.
async def test_dialog_mode_detects_each_turn_and_takes_an_unsure_one_as_the_other_person(live, db_session):
    stt = Turns([("ko", 1.0), ("ko", 0.55), ("ko", 0.55)])
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    with live.client.websocket_connect("/api/translate/live") as ws:
        start_dialog(ws, live.token)
        first = turn(ws, 1)
        second = turn(ws, 2)
        # 화면에서 마디 2의 방향을 바꿨다(HTTP 음성 API가 한국어로 다시 처리): 다음 확신 낮은 마디는 영어로
        # 본다. 더 오래된 마디에 대한 알림은 화면에서처럼 아무것도 바꾸지 않는다.
        ws.send_json({"type": "previous", "id": 2, "lang": "ko"})
        ws.send_json({"type": "previous", "id": 1, "lang": "en"})
        third = turn(ws, 3)
    assert [message["type"] for message in (first, second, third)] == ["final"] * 3
    first, second, third = first["result"], second["result"], third["result"]
    assert (first["source_lang"], first["target_lang"]) == ("ko", "en")
    assert (first["language_confidence"], first["language_guessed"]) == (1.0, False)
    assert first["translated_text"] == f"[ko->en] {words(len(SECOND))}"
    assert (second["source_lang"], second["target_lang"], second["language_guessed"]) == ("en", "ko", True)
    assert second["language_confidence"] == 0.55
    assert (third["source_lang"], third["language_guessed"]) == ("en", True)
    assert await records(db_session) == 3


# 마디는 끝난 순서대로 저장한다. 첫 마디가 아직 인식 중일 때 둘째 마디가 미리 처리되면, 둘째 마디는 확신이
# 낮으므로 저장할 때 앞 마디의 실제 언어(한국어)를 기준으로 다시 처리해 영어가 된다(판별은 다시 하지 않음,
# docs/api.md). HTTP 두 사람 대화에 previous_lang=ko를 준 것과 같은 결과다.
async def test_a_dialog_turn_prepared_before_the_turn_ahead_was_saved_is_redone_if_its_guess_changes(
    live, db_session
):
    stt = Turns([("ko", 1.0), ("ko", 0.55)], hold=True)
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    with live.client.websocket_connect("/api/translate/live") as ws:
        start_dialog(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        assert stt.entered.wait(5)  # 마디 2가 쉴 때 마디 1은 아직 인식 중이다
        ws.send_json({"type": "utterance", "id": 2})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 2})
        ws.send_json({"type": "end", "id": 2})
        stt.release.set()
        messages = until(ws, "final")
        messages += until(ws, "final")
    saved = [(message["id"], message["result"]["source_lang"]) for message in messages]
    assert saved == [(1, "ko"), (2, "en")]
    assert messages[1]["result"]["language_guessed"] is True
    assert await records(db_session) == 2


# 다시 연결할 때 start에 직전 마디(언어 en, id 5)를 넘기면 서버가 그것을 마지막 저장 마디로 기억한다. 그래서
# id 5의 방향을 바꾸는 previous 메시지(ko)를 따르고, 다음 확신 낮은 마디는 영어로 처리된다.
async def test_a_reconnected_dialog_keeps_the_previous_turn_it_is_given(live):
    stt = Turns([("ko", 0.55)])
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    with live.client.websocket_connect("/api/translate/live") as ws:
        start_dialog(ws, live.token, previous_lang="en", previous_id=5)
        ws.send_json({"type": "previous", "id": 5, "lang": "ko"})
        final = turn(ws, 6)["result"]
    assert (final["source_lang"], final["language_guessed"]) == ("en", True)


# 모드별 시작 메시지 검증(모두 4422): 모르는 모드, 대화 모드의 같은 언어끼리, 두 사람 대화의 지원하지 않는
# 직전 언어, previous_lang만 있고 previous_id 없음(둘은 함께 보내거나 함께 뺌), 문자열 id.
@pytest.mark.parametrize(
    "first",
    [
        {"mode": "chat", "source_lang": "ko", "target_lang": "en"},
        {"mode": "conversation", "source_lang": "ko", "target_lang": "ko"},
        {"mode": "dialog", "previous_lang": "ja", "previous_id": 1},
        {"mode": "dialog", "previous_lang": "ko"},
        {"mode": "dialog", "previous_lang": "ko", "previous_id": "1"},
    ],
)
async def test_a_bad_mode_start_closes_the_connection(live, first):
    with live.client.websocket_connect("/api/translate/live") as ws:
        ws.send_json({"type": "start", "token": live.token, **first})
        assert closed_with(ws) == 4422


# previous 메시지는 두 사람 대화에서만 쓰고, 다른 모드에서 오거나 언어가 틀리면 4422로 닫는다.
@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("live", {"type": "previous", "id": 1, "lang": "ko"}),
        ("conversation", {"type": "previous", "id": 1, "lang": "ko"}),
        ("dialog", {"type": "previous", "id": 1, "lang": "ja"}),
    ],
)
async def test_a_bad_previous_message_closes_the_connection(live, mode, message):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token, mode=mode)
        ws.send_json(message)
        assert closed_with(ws) == 4422


# 언어 판별(detect_language)이 없는 인식기로 두 사람 대화를 시작하면 4503으로 닫는다. 인식 호출에 이르면 안
# 되므로 transcribe는 불리면 실패하게 둔다.
async def test_dialog_mode_without_language_detection_is_unavailable(live):
    class NoDetection:
        model_name = "plain-stt"

        def transcribe(self, audio, language):
            raise AssertionError("not reached")

    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=NoDetection())
    with live.client.websocket_connect("/api/translate/live") as ws:
        ws.send_json({"type": "start", "token": live.token, "mode": "dialog"})
        assert closed_with(ws) == 4503
