"""The live subtitle WebSocket (T56) with fake models, through Starlette's test client."""

import asyncio
import logging
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import jwt
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import get_settings
from app.core.security import create_token
from app.db.session import ENGINE_OPTIONS
from app.dependencies import get_audio_store, get_models, get_sessions
from app.main import app
from app.models import Translation
from app.routers.live import LiveOptions, get_live_options
from app.services.audio_store import AudioStore
from app.services.interfaces import ModelError
from app.services.pipeline import PipelineModels
from tests.fakes import FakeLiveSpeechToText, FakeSpeechToText, FakeTextToSpeech, FakeTranslator

SECOND = b"\x01\x00" * 16_000  # one second of 16 kHz 16-bit audio: ten words to the fake
words = FakeLiveSpeechToText.words


@pytest.fixture
def live(migrated_database, user, tmp_path):
    # The test client runs the app on an event loop of its own; an engine without a pool shares no
    # connection between that loop and the tests'.
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


def start(ws, token, **changes):
    ws.send_json({"type": "start", "token": token, "source_lang": "ko", "target_lang": "en", **changes})
    assert ws.receive_json() == {"type": "ready"}


def until(ws, kind):
    """Every message up to and including the first of this type."""
    seen = [ws.receive_json()]
    while seen[-1]["type"] != kind:
        seen.append(ws.receive_json())
    return seen


def closed_with(ws) -> int:
    with pytest.raises(WebSocketDisconnect) as closed:
        while True:
            ws.receive_json()
    return closed.value.code


async def records(db_session) -> int:
    return await db_session.scalar(select(func.count()).select_from(Translation))


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
        # What the two results share is stable, down to the word.
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


async def test_an_end_without_a_pause_uses_all_the_audio(live, db_session):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 7})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "end", "id": 7})
        final = until(ws, "final")[-1]
    assert (final["id"], final["result"]["source_text"]) == (7, words(len(SECOND)))


async def test_cancelled_stale_and_stray_messages_leave_no_trace(live, db_session):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_bytes(SECOND)  # outside an utterance
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "cancel", "id": 1})
        ws.send_json({"type": "end", "id": 1})  # already dropped
        ws.send_json({"type": "utterance", "id": 2})
        ws.send_json({"type": "pause", "id": 9})  # not the current utterance
        ws.send_bytes(SECOND[:16_000])
        ws.send_json({"type": "end", "id": 2})
        messages = until(ws, "final")
    assert all(message["id"] != 1 or message["type"] in ("source", "translation") for message in messages)
    assert messages[-1]["id"] == 2
    assert messages[-1]["result"]["source_text"] == words(16_000)
    assert await records(db_session) == 1


async def test_an_utterance_without_speech_is_an_error_and_no_record(live, db_session):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND[:1000])  # shorter than one fake word
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        messages = until(ws, "error")
        # The connection goes on with the next utterance.
        ws.send_json({"type": "utterance", "id": 2})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "end", "id": 2})
        final = until(ws, "final")[-1]
    assert messages == [{"type": "error", "id": 1, "detail": "No speech was recognized"}]
    assert final["id"] == 2
    assert await records(db_session) == 1


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
    # Failed live updates show nothing; the failed final is an error for that utterance.
    assert messages == [{"type": "error", "id": 1, "detail": "Translation service is unavailable"}]
    assert "ModelError" in caplog.text and "secret" not in caplog.text
    assert await records(db_session) == 0


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


async def test_no_start_message_in_time_closes_the_connection(live):
    app.dependency_overrides[get_live_options] = lambda: LiveOptions(interval_s=0.0, start_timeout_s=0.1)
    with live.client.websocket_connect("/api/translate/live") as ws:
        assert closed_with(ws) == 4422


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


@pytest.mark.parametrize("stt", [None, FakeSpeechToText()])
async def test_without_live_recognition_the_service_is_unavailable(live, stt):
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    with live.client.websocket_connect("/api/translate/live") as ws:
        ws.send_json({"type": "start", "token": live.token, "source_lang": "ko", "target_lang": "en"})
        assert closed_with(ws) == 4503


@pytest.mark.parametrize(
    "message", ["not json", '{"type": "utterance", "id": "1"}', '{"type": "stop", "id": 1}']
)
async def test_a_bad_control_message_closes_the_connection(live, message):
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_text(message)
        assert closed_with(ws) == 4422


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


class HeldFinal(FakeLiveSpeechToText):
    """Holds the first final recognition in the model thread until released, and notes when each update
    starts (T60: the fast fakes above never catch the server in the middle of a call)."""

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


async def test_speech_while_the_final_is_prepared_drops_that_final(live, db_session):
    stt = HeldFinal()
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        ws.send_bytes(SECOND)
        ws.send_json({"type": "pause", "id": 1})
        assert stt.entered.wait(5)  # the prepared final is inside the recognition model
        ws.send_json({"type": "resume", "id": 1})
        ws.send_bytes(SECOND)
        stt.release.set()
        ws.send_json({"type": "pause", "id": 1})
        ws.send_json({"type": "end", "id": 1})
        final = until(ws, "final")[-1]
    # Whether the resume came before or after the held call returned, only the second pause counts.
    assert final["result"]["source_text"] == words(2 * len(SECOND))
    assert await records(db_session) == 1


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
    time.sleep(0.2)  # the server has seen the disconnect and cancelled the final
    stt.release.set()
    assert stt.left.wait(5)
    await asyncio.sleep(0.3)
    assert await records(db_session) == 0
    assert not list((tmp_path / "audio").glob("*.wav"))


async def test_updates_start_at_least_the_interval_apart(live):
    stt = HeldFinal(hold=False)
    app.dependency_overrides[get_models] = lambda: replace(live.models, stt=stt)
    app.dependency_overrides[get_live_options] = lambda: LiveOptions(interval_s=0.3)
    with live.client.websocket_connect("/api/translate/live") as ws:
        start(ws, live.token)
        ws.send_json({"type": "utterance", "id": 1})
        began = time.perf_counter()
        while time.perf_counter() - began < 1.6:
            ws.send_bytes(SECOND[:3200])  # 0.1 s of audio, faster than real time
            time.sleep(0.05)
        ws.send_json({"type": "end", "id": 1})
        until(ws, "final")
    starts = stt.update_starts
    assert len(starts) >= 4
    assert all(later - earlier >= 0.25 for earlier, later in zip(starts, starts[1:], strict=False))


async def test_a_failed_save_is_an_error_and_leaves_no_file(live, db_session, tmp_path, caplog):
    factory = app.dependency_overrides[get_sessions]()
    opened = []

    def sessions():
        session = factory()
        opened.append(session)
        if len(opened) > 1:  # the first session checks the user at the start

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
