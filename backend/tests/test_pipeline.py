import asyncio
import json
import threading
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import func, select

from app.models import AudioFile, Translation
from app.services import pipeline
from app.services.audio_store import AudioStore
from app.services.correction import OllamaCorrector
from app.services.inference import run_model
from app.services.interfaces import SynthesizedAudio
from tests.fakes import FakeCorrector, FakeSpeechToText, FakeTextToSpeech, FakeTranslator, silent_wav


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


async def translate_with(models, db_session, user, tmp_path, source="en", target="ko", **request):
    return await pipeline.translate(
        models=models,
        store=AudioStore(tmp_path),
        db=db_session,
        user_id=user.id,
        source=source,
        target=target,
        **request,
    )


async def test_typed_english_is_corrected_before_translation(db_session, user, tmp_path):
    corrector = FakeCorrector("I have a cat.")
    models = pipeline.PipelineModels(None, FakeTranslator(), corrector=corrector)
    result = await translate_with(models, db_session, user, tmp_path, text="I hvae a cat.")
    assert corrector.calls == [("I hvae a cat.", "en")]
    assert result.translated_text == "[en->ko] I have a cat."
    # The record keeps what the user typed, and names both models.
    assert result.source_text == "I hvae a cat."
    assert result.mt_model == "fake-mt + fake-corrector"


async def test_a_failed_correction_translates_the_text_as_typed(db_session, user, tmp_path):
    models = pipeline.PipelineModels(None, FakeTranslator(), corrector=FakeCorrector(None))
    result = await translate_with(models, db_session, user, tmp_path, text="I hvae a cat.")
    assert result.translated_text == "[en->ko] I hvae a cat."
    assert result.mt_model == "fake-mt"


TYPED, FIXED = "I hvae a cat.", "I have a cat."
FINISHED = {"done": True, "done_reason": "stop"}


@pytest.fixture
def ready_corrector():
    """Makes the real corrector, prepared, with only Ollama's HTTP answer to the typed text replaced (T38,
    T40), and closes each one after the test (T41)."""
    made: list[OllamaCorrector] = []

    def make(reply: httpx.Response) -> OllamaCorrector:
        def ollama(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if request.url.path == "/api/chat" and body["messages"][1]["content"] == TYPED:
                return reply
            return httpx.Response(200, json={"message": {"content": "Hello."}, **FINISHED})

        corrector = OllamaCorrector("qwen2.5:1.5b-instruct")
        corrector._client.close()
        corrector._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(ollama))
        made.append(corrector)
        corrector.start().join(5)
        assert corrector.corrects("en")
        return corrector

    yield make
    for corrector in made:
        corrector.close()


async def saved(db_session, result) -> tuple[str, str, str]:
    query = select(Translation.source_text, Translation.translated_text, Translation.mt_model)
    return tuple((await db_session.execute(query.where(Translation.id == result.id))).one())


async def test_a_real_correction_reaches_translation_and_the_record(
    db_session, user, tmp_path, ready_corrector
):
    corrector = ready_corrector(httpx.Response(200, json={"message": {"content": FIXED}, **FINISHED}))
    models = pipeline.PipelineModels(None, FakeTranslator(), corrector=corrector)
    result = await translate_with(models, db_session, user, tmp_path, text=TYPED)
    expected = (TYPED, f"[en->ko] {FIXED}", "fake-mt + ollama/qwen2.5:1.5b-instruct")
    assert (result.source_text, result.translated_text, result.mt_model) == expected
    assert await saved(db_session, result) == expected


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(500, json={"error": "model crashed"}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"message": {"content": "   "}, **FINISHED}),
        httpx.Response(200, json={"message": {"content": "I have a"}, "done": True, "done_reason": "length"}),
        httpx.Response(200, json={"message": {"content": FIXED}, "done": False}),
        httpx.Response(200, json={"message": {"content": FIXED}, "done": True, "done_reason": "load"}),
        httpx.Response(200, json={"message": {"content": "I cannot help with that request."}, **FINISHED}),
        httpx.Response(200, json={"message": {"content": "고양이가 있습니다."}, **FINISHED}),
    ],
    ids=["http-500", "not-json", "empty", "cut-off", "unfinished", "other-finish", "refusal", "korean"],
)
async def test_an_unusable_reply_from_ollama_leaves_the_typed_text(
    reply, db_session, user, tmp_path, caplog, ready_corrector
):
    models = pipeline.PipelineModels(None, FakeTranslator(), corrector=ready_corrector(reply))
    result = await translate_with(models, db_session, user, tmp_path, text=TYPED)
    expected = (TYPED, f"[en->ko] {TYPED}", "fake-mt")
    assert (result.source_text, result.translated_text, result.mt_model) == expected
    assert await saved(db_session, result) == expected
    assert "Typo correction" in caplog.text and TYPED not in caplog.text


async def test_korean_text_and_speech_are_not_corrected(db_session, user, tmp_path):
    corrector = FakeCorrector()
    models = pipeline.PipelineModels(FakeSpeechToText(), FakeTranslator(), corrector=corrector)
    korean = await translate_with(models, db_session, user, tmp_path, "ko", "en", text="안녕하세요")
    spoken = await translate_with(models, db_session, user, tmp_path, audio=silent_wav(100))
    assert corrector.calls == []
    assert korean.mt_model == spoken.mt_model == "fake-mt"


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
    # Logged as the exception's type and place, not its message (T54): the refusal in resolve().
    assert "ValueError (at audio_store.py" in caplog.text and "in resolve)" in caplog.text


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


async def test_a_save_cancelled_while_writing_leaves_neither_record_nor_file(db_session, user, tmp_path):
    """A live connection that closes mid-save cancels it while the worker thread writes the file (T59)."""
    entered, release = threading.Event(), threading.Event()
    store = AudioStore(tmp_path / "audio")
    resolve = store.resolve

    def held(name):
        entered.set()
        assert release.wait(5)
        return resolve(name)

    store.resolve = held
    speech = SynthesizedAudio(wav=silent_wav(100), sample_rate=16_000, duration_ms=100)
    prepared = pipeline.Prepared(
        mode="speech",
        source="ko",
        target="en",
        source_text="원문",
        translated_text="text",
        mt_model="fake-mt",
        mt_ms=1,
        speech=speech,
        tts_error=None,
    )
    saving = asyncio.create_task(
        pipeline.save(db=db_session, store=store, user_id=user.id, prepared=prepared)
    )
    assert await asyncio.to_thread(entered.wait, 5)
    saving.cancel()
    await asyncio.sleep(0)  # the cancellation reaches save() while the write is still held
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await saving
    assert not list((tmp_path / "audio").glob("*"))
    assert await db_session.scalar(select(func.count()).select_from(Translation)) == 0
