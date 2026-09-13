import asyncio
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.models import AudioFile, Translation
from app.services import pipeline
from app.services.audio_store import AudioStore
from app.services.inference import run_model
from tests.fakes import FakeSpeechToText, FakeTextToSpeech, FakeTranslator, silent_wav


async def test_models_run_off_loop_and_queue_wait_is_excluded(db_session, user, tmp_path, monkeypatch):
    loop_thread = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    models = pipeline.PipelineModels(FakeSpeechToText(), FakeTranslator(), FakeTextToSpeech())
    for model, method in [
        (models.stt, "transcribe"),
        (models.translator, "translate"),
        (models.tts, "synthesize"),
    ]:
        original = getattr(model, method)

        def checked(*args, original=original, method=method):
            assert threading.get_ident() != loop_thread
            calls.append(method)
            return original(*args)

        monkeypatch.setattr(model, method, checked)

    def occupy_worker():
        entered.set()
        assert release.wait(5)

    blocker = asyncio.create_task(run_model(occupy_worker))
    while not entered.is_set():
        await asyncio.sleep(0.005)
    # Deterministic clock measures only inside each model call; waiting must not consume clock ticks.
    ticks = iter([1.0, 1.012, 2.0, 2.023, 3.0, 3.034])
    monkeypatch.setattr(pipeline, "perf_counter", lambda: next(ticks))
    task = asyncio.create_task(
        pipeline.translate(
            models=models,
            store=AudioStore(tmp_path),
            db=db_session,
            user_id=user.id,
            source="en",
            target="ko",
            audio=silent_wav(100),
        )
    )
    await asyncio.sleep(0.05)
    assert not task.done() and calls == []
    release.set()
    await blocker
    result = await task
    assert calls == ["transcribe", "translate", "synthesize"]
    assert (result.stt_ms, result.mt_ms, result.tts_ms) == (12, 23, 34)


async def test_event_loop_stays_responsive_during_model_call(db_session, user, tmp_path, monkeypatch):
    model = FakeTranslator()
    original = model.translate
    entered = threading.Event()

    def slow(*args):
        entered.set()
        time.sleep(0.2)
        return original(*args)

    monkeypatch.setattr(model, "translate", slow)
    task = asyncio.create_task(
        pipeline.translate(
            models=pipeline.PipelineModels(None, model),
            store=AudioStore(tmp_path),
            db=db_session,
            user_id=user.id,
            source="en",
            target="ko",
            text="hello",
        )
    )
    while not entered.is_set():
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.01)
    assert not task.done()
    await task


async def test_db_failure_removes_synthesized_file(db_session, user, tmp_path, monkeypatch):
    async def fail_commit():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(db_session, "commit", fail_commit)
    user_id = user.id
    with pytest.raises(RuntimeError, match="database unavailable"):
        await pipeline.translate(
            models=pipeline.PipelineModels(None, FakeTranslator(), FakeTextToSpeech()),
            store=AudioStore(tmp_path),
            db=db_session,
            user_id=user_id,
            source="en",
            target="ko",
            text="hello",
        )
    assert not list(tmp_path.iterdir())
    assert await db_session.scalar(select(func.count()).select_from(Translation)) == 0
    assert await db_session.scalar(select(func.count()).select_from(AudioFile)) == 0


async def test_audio_store_never_reads_or_deletes_outside_directory(tmp_path, caplog):
    outside = tmp_path / "private.wav"
    outside.write_bytes(b"private")
    store = AudioStore(tmp_path / "audio")
    for path in [str(outside), "../private.wav"]:
        assert await store.read(path) is None
        await store.delete(path)
        assert outside.read_bytes() == b"private"
    assert "outside AUDIO_DIR" in caplog.text


@pytest.mark.parametrize("failure_stage", ["write", "close"])
async def test_partial_write_is_removed(tmp_path, monkeypatch, failure_stage):
    store = AudioStore(tmp_path)
    original = Path.open

    class BrokenWriter:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
            if failure_stage == "close":
                raise OSError("disk full")

        def close(self):
            self.file.close()

        def write(self, data):
            self.file.write(data[:5])
            if failure_stage == "write":
                raise OSError("disk full")

    def broken_open(path, *args, **kwargs):
        return BrokenWriter(original(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", broken_open)
    with pytest.raises(OSError, match="disk full"):
        await store.save(uuid4(), b"a synthesized wav")
    assert not list(tmp_path.iterdir())
