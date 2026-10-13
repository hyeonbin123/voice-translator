# [파일 역할] 번역 흐름 app/services/pipeline.py(음성 → 인식 → (영어 글자면 오타 교정) → 번역 → 합성 → 기록
# 저장)와 번역 음성 저장소 services/audio_store.py를 가짜 모델과 실제 테스트 DB로 검증한다.
# [무엇을 확인하나] 모델 호출이 이벤트 루프 밖 스레드에서 돌고 처리 시간에 대기 시간이 섞이지 않는지, 오타
# 교정이 모델 스레드를 막지 않고 한 번에 하나씩 돌며 실패하면 입력 그대로 번역하는지, DB·디스크 실패나 취소에
# 기록과 파일이 짝이 맞게 정리되는지, 저장소가 폴더 밖 경로를 읽거나 지우지 않는지, CPU 합성의 스레드 배치.
# [왜 중요한가] 이 파일이 지키는 것은 서버 전체의 응답성(한 사람의 느린 요청이 다른 사람을 막지 않음)과 데이터
# 정합성(기록 없는 음성 파일, 파일 없는 기록이 남지 않음)이다.

import asyncio
import json
import threading
import time
from pathlib import Path
from uuid import uuid4

# httpx: 실제 OllamaCorrector의 HTTP 응답을 MockTransport로 바꿀 때 쓴다. sqlalchemy의 select·func: 저장된
# 행과 개수를 DB에서 직접 확인한다.
import httpx
import pytest
from sqlalchemy import func, select

# run_model·run_synthesis는 모델 스레드와 합성 전용 스레드 실행기다. 가짜 모델들은 tests/fakes.py에 있다.
from app.models import AudioFile, Translation
from app.services import pipeline
from app.services.audio_store import AudioStore
from app.services.correction import OllamaCorrector
from app.services.inference import run_model, run_synthesis
from app.services.interfaces import SynthesizedAudio
from tests.fakes import FakeCorrector, FakeSpeechToText, FakeTextToSpeech, FakeTranslator, silent_wav


# 세 모델 호출(인식·번역·합성)이 모두 이벤트 루프 스레드가 아닌 곳에서, 이 순서로 불리는지 확인한다. 그리고
# 응답의 stt_ms·mt_ms·tts_ms가 모델 호출 안의 시간만 재고 모델 스레드 대기 시간은 넣지 않는지 본다.
# 방법: 하나뿐인 모델 스레드를 다른 작업으로 막아 두어 번역 요청이 줄을 서게 하고, perf_counter를 정해진 값을
# 차례로 내는 가짜 시계로 바꾼다. 대기 중에는 시계를 읽지 않으므로 결과가 정확히 12·23·34 ms여야 한다.
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


# 임시 폴더 저장소와 테스트 DB로 pipeline.translate를 부르는 도우미. request에는 text 또는 audio를 넘긴다.
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


# 글자로 입력한 영어는 번역 전에 교정기를 거치고, 번역기는 교정된 글자를 받는다. 기록의 원문은 사용자가 친
# 그대로이고, 모델 이름에는 번역기와 교정기가 함께 적히는지 확인한다.
async def test_typed_english_is_corrected_before_translation(db_session, user, tmp_path):
    corrector = FakeCorrector("I have a cat.")
    models = pipeline.PipelineModels(None, FakeTranslator(), corrector=corrector)
    result = await translate_with(models, db_session, user, tmp_path, text="I hvae a cat.")
    assert corrector.calls == [("I hvae a cat.", "en")]
    assert result.translated_text == "[en->ko] I have a cat."
    # The record keeps what the user typed, and names both models.
    assert result.source_text == "I hvae a cat."
    assert result.mt_model == "fake-mt + fake-corrector"


# 회귀 테스트(커밋 edf6e32): 교정은 Ollama HTTP 호출이라 오래 걸릴 수 있으므로 모델 스레드가 아니라 교정 전용
# 스레드에서 돈다. 교정이 멈춰 있는 동안 다른 사람의 음성 번역이 2초 안에 끝나고(교정은 아직 풀려나지 않음),
# 교정이 끝난 뒤 글자 요청도 정상 결과를 내는지 확인한다.
async def test_a_slow_correction_does_not_hold_the_model_thread():
    """The correction is an HTTP call to Ollama that can take up to CORRECTION_TIMEOUT_S: speech from other
    requests must not wait behind it on the model thread."""

    class BlockingCorrector(FakeCorrector):
        def __init__(self) -> None:
            super().__init__("I have a cat.")
            self.entered, self.release = threading.Event(), threading.Event()

        def correct(self, text, language):
            self.entered.set()
            self.release.wait(5)
            return super().correct(text, language)

    corrector = BlockingCorrector()
    typed = asyncio.create_task(
        pipeline.prepare(
            models=pipeline.PipelineModels(None, FakeTranslator(), corrector=corrector),
            source="en",
            target="ko",
            text="I hvae a cat.",
        )
    )
    try:
        assert await asyncio.to_thread(corrector.entered.wait, 5)
        speech = pipeline.prepare(
            models=pipeline.PipelineModels(FakeSpeechToText(), FakeTranslator()),
            source="ko",
            target="en",
            audio=silent_wav(100),
        )
        spoken = await asyncio.wait_for(speech, timeout=2)
        assert spoken.mt_model == "fake-mt" and not corrector.release.is_set()
    finally:
        corrector.release.set()
        result = await typed
    assert result.mt_model == "fake-mt + fake-corrector"
    assert result.source_text == "I hvae a cat."
    assert result.translated_text == "[en->ko] I have a cat."
    assert corrector.calls == [("I hvae a cat.", "en")]


# 풀어 줄 때까지 모든 교정을 붙잡아 두고, 동시에 실행 중인 교정 수의 최댓값(most)을 세는 가짜 교정기. 교정이
# 시작될 때마다 세마포어 허가를 하나 내어, 테스트가 '교정이 시작됐는가'를 시간 제한을 두고 기다릴 수 있게
# 한다.
class GatedCorrector(FakeCorrector):
    """Holds every correction until released and counts how many run at the same time."""

    def __init__(self) -> None:
        super().__init__("I have a cat.")
        self.started = threading.Semaphore(0)  # one permit per correction that has begun
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.running = self.most = 0

    def correct(self, text, language):
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
        self.started.release()
        try:
            self.release.wait(5)
            return super().correct(text, language)
        finally:
            with self.lock:
                self.running -= 1

    async def begins_within(self, seconds: float) -> bool:
        return await asyncio.to_thread(self.started.acquire, True, seconds)


# 교정기가 붙은 영어 글자 번역 준비(prepare)를 작업으로 띄우는 도우미.
def typed_request(corrector: FakeCorrector) -> asyncio.Task:
    return asyncio.create_task(
        pipeline.prepare(
            models=pipeline.PipelineModels(None, FakeTranslator(), corrector=corrector),
            source="en",
            target="ko",
            text="I hvae a cat.",
        )
    )


# 회귀 테스트(커밋 890e53a): 교정은 모델과 같은 GPU를 쓰는 Ollama 작업이므로, 글자 요청이 동시에 와도 교정은
# 하나씩만 돈다(교정 전용 스레드 1개). 첫 교정이 붙잡힌 동안 두 번째는 0.3초 안에 시작하지 않아야 하고, 동시
# 실행 최댓값은 1이다.
async def test_typed_requests_are_corrected_one_at_a_time():
    """Each correction is Ollama work on the GPU the models use: one at a time, as when it ran on the model
    thread, not one per typed request at once (T73)."""
    corrector = GatedCorrector()
    first, second = typed_request(corrector), typed_request(corrector)
    try:
        assert await corrector.begins_within(5)
        assert not await corrector.begins_within(0.3), "the second correction did not wait for the first"
    finally:
        corrector.release.set()
        results = await asyncio.gather(first, second)
    assert corrector.most == 1
    assert [result.translated_text for result in results] == ["[en->ko] I have a cat."] * 2


# 요청이 취소돼도 이미 시작한 스레드 속 Ollama 호출은 멈출 수 없으므로 끝까지 돈다. 그다음 요청의 교정이 그
# 호출과 겹치지 않고 기다리는지 확인한다.
async def test_the_next_correction_waits_for_a_cancelled_requests_correction():
    # A thread cannot be stopped: the cancelled request's call to Ollama runs to its end, and the next
    # correction must not overlap it.
    corrector = GatedCorrector()
    first = typed_request(corrector)
    assert await corrector.begins_within(5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = typed_request(corrector)
    try:
        assert not await corrector.begins_within(0.3), "the next correction overlapped the cancelled one"
    finally:
        corrector.release.set()
        result = await second
    assert corrector.most == 1
    assert result.translated_text == "[en->ko] I have a cat."
    assert len(corrector.calls) == 2


# 줄을 서서 기다리던 중 취소된 요청의 교정은 아예 실행되지 않는지 확인한다(inference._run이 대기 중인 작업을
# 비움). 쓸데없는 GPU 작업을 하지 않는다.
async def test_a_correction_cancelled_while_waiting_never_runs():
    corrector = GatedCorrector()
    first, waiting = typed_request(corrector), None
    try:
        assert await corrector.begins_within(5)
        waiting = typed_request(corrector)
        await asyncio.sleep(0.1)  # queued behind the first correction
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        corrector.release.set()
        await first
    assert not await corrector.begins_within(0.3)
    assert corrector.calls == [("I hvae a cat.", "en")]


# 교정기가 답을 주지 못하면(None, Ollama가 꺼진 경우와 같음) 입력 그대로 번역하고, 모델 이름에도 교정기를 적지
# 않는다. 교정은 보조 단계라 실패가 번역을 막으면 안 된다(services/correction.py docstring).
async def test_a_failed_correction_translates_the_text_as_typed(db_session, user, tmp_path):
    models = pipeline.PipelineModels(None, FakeTranslator(), corrector=FakeCorrector(None))
    result = await translate_with(models, db_session, user, tmp_path, text="I hvae a cat.")
    assert result.translated_text == "[en->ko] I hvae a cat."
    assert result.mt_model == "fake-mt"


# 오타가 있는 입력과 그 교정 결과, 그리고 Ollama가 정상적으로 끝낸 응답의 표시(done, done_reason stop).
TYPED, FIXED = "I hvae a cat.", "I have a cat."
FINISHED = {"done": True, "done_reason": "stop"}


# 실제 OllamaCorrector를 쓰되 HTTP만 MockTransport로 바꾼다. 준비 단계 호출에는 정상 답을, 시험할
# 입력(TYPED)에는 테스트가 정한 응답을 준다. start()로 준비를 끝내 corrects("en")이 참인 상태로 만들고,
# 테스트가 끝나면 만든 교정기를 모두 닫는다(백그라운드 스레드·HTTP 클라이언트 정리).
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


# DB에 실제로 저장된 원문·번역문·번역 모델 이름을 읽는다. 응답뿐 아니라 기록도 같은지 보려는 도우미다.
async def saved(db_session, result) -> tuple[str, str, str]:
    query = select(Translation.source_text, Translation.translated_text, Translation.mt_model)
    return tuple((await db_session.execute(query.where(Translation.id == result.id))).one())


# 실제 교정기 코드를 거친 정상 교정이 번역과 DB 기록까지 이어지는지 확인한다. 기록의 원문은 입력 그대로,
# 번역문은 교정된 글자의 번역, 모델 이름에 교정 모델이 붙는다.
async def test_a_real_correction_reaches_translation_and_the_record(
    db_session, user, tmp_path, ready_corrector
):
    corrector = ready_corrector(httpx.Response(200, json={"message": {"content": FIXED}, **FINISHED}))
    models = pipeline.PipelineModels(None, FakeTranslator(), corrector=corrector)
    result = await translate_with(models, db_session, user, tmp_path, text=TYPED)
    expected = (TYPED, f"[en->ko] {FIXED}", "fake-mt + ollama/qwen2.5:1.5b-instruct")
    assert (result.source_text, result.translated_text, result.mt_model) == expected
    assert await saved(db_session, result) == expected


# 쓸 수 없는 Ollama 응답 8가지(서버 오류, JSON 아님, 빈 답, 출력 상한에서 잘림, 끝나지 않음, 다른 이유로 끝남,
# 거절 문장, 한국어로 번역해 버림)는 모두 교정을 버리고 입력 그대로 번역하는지 확인한다. 거절·번역처럼 원문과
# 너무 다른 답은 교정이 아니라고 본다(correction.py의 looks_like_a_correction).
# 로그에는 교정 실패가 남지만 사용자가 입력한 글자는 남지 않아야 한다.
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


# 한국어 글자 입력과 음성 입력은 교정하지 않는다(README: 글자로 입력한 영어만 교정). 한국어는 교정이 깨끗한
# 문장을 망가뜨려 쓰지 않기로 했다.
async def test_korean_text_and_speech_are_not_corrected(db_session, user, tmp_path):
    corrector = FakeCorrector()
    models = pipeline.PipelineModels(FakeSpeechToText(), FakeTranslator(), corrector=corrector)
    korean = await translate_with(models, db_session, user, tmp_path, "ko", "en", text="안녕하세요")
    spoken = await translate_with(models, db_session, user, tmp_path, audio=silent_wav(100))
    assert corrector.calls == []
    assert korean.mt_model == spoken.mt_model == "fake-mt"


# 번역 모델이 0.2초 걸리는 동안에도 이벤트 루프가 다른 일(여기서는 asyncio.sleep)을 할 수 있는지 확인한다.
# 모델 호출을 루프에서 직접 부르면 번역이 끝날 때까지 루프가 멈춰, 확인 시점에 작업이 이미 끝나 있게 된다.
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


# 합성 음성을 파일로 쓴 뒤 DB commit이 실패하면 예외를 그대로 올리되, 이미 쓴 음성 파일을 지우고 번역·음성
# 행도 남기지 않는지 확인한다. 기록 없는 음성 파일이 디스크에 쌓이지 않게 한다.
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


# 보안 확인: 저장소 폴더 밖을 가리키는 경로(절대 경로, ../)로는 읽기(None)도 지우기도 되지 않는다(경로 조작
# 차단, resolve의 is_relative_to 확인). 거부는 예외 메시지 없이 종류와 위치만 로그에 남는다.
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


# 음성 파일을 쓰는 도중(write) 또는 닫는 순간(close) 디스크 오류가 나면, 반쯤 쓴 파일을 지우고 오류를 올리는지
# 확인한다. Path.open을 5바이트만 쓰고 실패하는 가짜 파일로 바꿔 흉내 낸다.
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


# 회귀 테스트(커밋 7fa678e): 동시통역 연결이 저장 도중 끊기면 저장이 취소되는데, 파일 쓰기는 작업자 스레드에서
# 계속된다. pipeline.save는 쓰기를 shield로 끝까지 기다린 뒤 그 파일을 지우고 취소를 다시 올리므로, 파일도
# 기록도 남지 않아야 한다.
# 방법: 저장소의 resolve를 붙잡아 쓰기가 진행 중일 때 취소를 보내고, 그 뒤에 쓰기를 풀어 준다.
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


# 합성이 어느 스레드(이름)에서 불렸는지 언어와 함께 기록하는 가짜 합성기.
class ThreadRecordingTTS(FakeTextToSpeech):
    def __init__(self) -> None:
        super().__init__()
        self.threads: list[tuple[str, str]] = []

    def synthesize(self, text, language):
        self.threads.append((language, threading.current_thread().name))
        return super().synthesize(text, language)


# synthesis_off_model_thread에 든 언어(CPU로 도는 한국어 Supertonic 설정)의 합성은 합성 전용 스레드에서, 그
# 밖의 언어(GPU의 영어 Kokoro)는 모델 스레드에서 도는지 스레드 이름으로 확인한다. CPU 합성이 GPU 모델들의 줄을
# 막지 않게 하려는 배치다(inference.run_synthesis docstring).
async def test_cpu_synthesis_runs_on_the_synthesis_thread_and_the_rest_on_the_model_thread():
    """Korean synthesis on the CPU (Supertonic, T78) leaves the model thread to the GPU models; English
    synthesis (Kokoro on the GPU) stays on it."""
    tts = ThreadRecordingTTS()
    bundle = pipeline.PipelineModels(
        FakeSpeechToText(), FakeTranslator(), tts=tts, synthesis_off_model_thread=frozenset({"ko"})
    )
    await pipeline.prepare(models=bundle, source="en", target="ko", text="Hello.")
    await pipeline.prepare(models=bundle, source="ko", target="en", text="안녕하세요.")
    assert [language for language, _ in tts.threads] == ["ko", "en"]
    assert tts.threads[0][1].startswith("synthesis") and tts.threads[1][1].startswith("model")


# 기본 설정(빈 집합)에서는 합성도 모델 스레드에서 돈다. 기본 서버(MeloTTS·Kokoro, GPU)의 동작은 바뀌지 않는다.
async def test_synthesis_stays_on_the_model_thread_by_default():
    tts = ThreadRecordingTTS()
    bundle = pipeline.PipelineModels(FakeSpeechToText(), FakeTranslator(), tts=tts)
    prepared = await pipeline.prepare(models=bundle, source="en", target="ko", text="Hello.")
    assert prepared.tts_error is None and prepared.tts_ms is not None
    assert tts.threads[0][1].startswith("model")


# 모델 스레드가 다른 일로 막혀 있어도 합성 전용 스레드의 작업은 기다리지 않고 바로 실행되는지(2초 안에 끝남)
# 확인한다. 두 실행기가 실제로 분리돼 있다는 확인이다.
async def test_the_synthesis_thread_does_not_wait_for_a_busy_model_thread():
    entered, release = threading.Event(), threading.Event()

    def hold() -> None:
        entered.set()
        release.wait(5)

    held = asyncio.create_task(run_model(hold))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        name = await asyncio.wait_for(run_synthesis(lambda: threading.current_thread().name), timeout=2)
        assert name.startswith("synthesis") and not release.is_set()
    finally:
        release.set()
        await held
