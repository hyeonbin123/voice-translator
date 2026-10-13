# [파일 역할] 두 가지를 검증한다. (1) 다른 테스트가 실제 모델 대신 쓰는 가짜 모델(tests/fakes.py)이 모델
# 인터페이스(app/services/interfaces.py)를 지키고 실제 모델처럼 실패할 수 있는지. (2) 모델 호출을 이벤트 루프
# 밖 전용 스레드에서 돌리는 app/services/inference.py가 루프를 막지 않고, 결과·예외를 그대로 돌려주며, 취소된
# 호출의 인수(음성)를 바로 놓는지.
# [왜 중요한가] 모델 호출은 CPU·GPU를 수백 ms씩 잡으므로 비동기 핸들러 안에서 직접 부르면 서버의 다른 요청이
# 모두 멈춘다(inference.py docstring). 가짜 모델이 인터페이스와 어긋나면 API 테스트가 실제와 다른 경로를
# 시험하게 된다.

import asyncio
import gc
import io
import threading
import time
import wave
import weakref

import pytest

# run_model: 모델 스레드 풀(MODEL_THREADS, 기본 1개)에서 실행. run_live_model: 동시통역 자막 갱신 전용
# 스레드에서 실행(최종 처리 뒤에 줄 서지 않게).
# 인터페이스는 isinstance로 확인할 수 있는 Protocol이고, 오류는 ModelError 아래에 '읽을 수 없는 음성'과
# '말소리 없음'이 나뉘어 있다.
from app.services.inference import run_live_model, run_model
from app.services.interfaces import (
    InvalidAudioError,
    ModelError,
    NoSpeechError,
    SpeechToText,
    TextToSpeech,
    Translator,
    UndecodableAudioError,
)
from tests.fakes import FakeSpeechToText, FakeTextToSpeech, FakeTranslator, silent_wav


# 가짜 인식·번역·합성기가 각 Protocol의 메서드를 갖췄는지 확인한다. 인터페이스에 메서드가 늘었는데 가짜에
# 빠지면 여기서 실패한다.
def test_fakes_satisfy_the_model_interfaces():
    assert isinstance(FakeSpeechToText(), SpeechToText)
    assert isinstance(FakeTranslator(), Translator)
    assert isinstance(FakeTextToSpeech(), TextToSpeech)


# 가짜 인식기가 정해 준 글자와 요청 언어를 돌려주고, 빈 음성은 UndecodableAudioError, text=""는
# NoSpeechError로 실패하는지 확인한다. API는 이 두 경우에 서로 다른 422 메시지를 주므로 가짜도 둘을 구분해야
# 한다.
def test_fake_speech_to_text_keeps_the_language_and_rejects_empty_audio():
    transcript = FakeSpeechToText(text="안녕하세요").transcribe(silent_wav(500), "ko")

    assert transcript.text == "안녕하세요"
    assert transcript.language == "ko"
    with pytest.raises(UndecodableAudioError):
        FakeSpeechToText().transcribe(b"", "ko")
    with pytest.raises(NoSpeechError):
        FakeSpeechToText(text="").transcribe(silent_wav(500), "ko")


# 음성 오류를 둘로 나눈 뒤에도(커밋 35ddbe4) 두 오류가 여전히 InvalidAudioError와 ModelError의 하위 클래스인지
# 확인한다. 부모 클래스로 잡던 기존 코드가 계속 동작하게 하려는 것이다.
def test_audio_errors_are_still_invalid_audio_and_model_errors():
    # Code that catches the parent classes keeps working after the split.
    for error in (UndecodableAudioError, NoSpeechError):
        assert issubclass(error, InvalidAudioError)
        assert issubclass(error, ModelError)


# error 인자를 주면 가짜 모델이 메모리 부족처럼 ModelError로 실패하는지 확인한다. API의 503 경로를 시험할 때
# 쓰는 기능이다.
def test_fakes_can_be_told_to_fail_like_a_crashed_model():
    with pytest.raises(ModelError, match="out of memory"):
        FakeSpeechToText(error=ModelError("out of memory")).transcribe(silent_wav(500), "ko")
    with pytest.raises(ModelError, match="out of memory"):
        FakeTranslator(error=ModelError("out of memory")).translate("hello", "en", "ko")


# 가짜 번역기는 "[en->ko] 원문"처럼 방향을 표시해, API 테스트가 올바른 방향으로 번역을 요청했는지 결과 글자만
# 보고 알 수 있게 한다. 같은 언어끼리의 번역은 거절한다.
def test_fake_translator_marks_the_direction():
    assert FakeTranslator().translate("hello", "en", "ko") == "[en->ko] hello"
    with pytest.raises(ModelError):
        FakeTranslator().translate("hello", "en", "en")


# 가짜 합성기의 출력이 실제로 열리는 WAV이고 표본화율·길이가 응답 값과 맞는지, fail=True면 ModelError인지
# 확인한다(합성이 실패해도 번역은 돌려주는 경로 시험용).
def test_fake_synthesis_returns_a_playable_wav_or_fails_on_request():
    audio = FakeTextToSpeech().synthesize("hello there", "en")

    with wave.open(io.BytesIO(audio.wav)) as clip:
        assert clip.getframerate() == audio.sample_rate
        assert clip.getnframes() * 1000 // clip.getframerate() == audio.duration_ms
    with pytest.raises(ModelError):
        FakeTextToSpeech(fail=True).synthesize("hello", "en")


# 모델 호출이 이벤트 루프를 막지 않는지 직접 잰다. 10 ms마다 깨어나는 heartbeat 작업을 돌리면서 0.3초 동안
# 스레드를 잡는 호출(time.sleep)을 run_model로 실행하고, heartbeat 간격이 한 번도 0.15초를 넘지 않았는지 본다.
# 같은 호출을 루프에서 직접 부르면 간격이 0.3초 이상으로 벌어진다.
async def test_model_calls_run_off_the_event_loop():
    gaps: list[float] = []
    done = asyncio.Event()

    async def heartbeat() -> None:
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    await run_model(time.sleep, 0.3)  # stands in for a model holding its thread
    done.set()
    await beat

    assert max(gaps) < 0.15


# 두 실행기 모두 반환값을 그대로 돌려주고, 스레드 안에서 난 예외(ZeroDivisionError)를 호출한 쪽에 그대로 다시
# 던지는지 확인한다.
async def test_model_calls_still_return_results_and_raise_errors():
    for run in (run_model, run_live_model):
        assert await run(sum, [1, 2, 3]) == 6
        with pytest.raises(ZeroDivisionError):
            await run(divmod, 1, 0)


# weakref를 걸 수 있는 빈 객체. 음성 바이트(bytes)는 약한 참조를 걸 수 없어 대신 쓴다.
class Audio:
    pass


# 회귀 테스트(커밋 bb3a56b): 대기열에서 기다리다 취소된 호출의 인수가 스레드가 그 차례에 이르기 전에 바로
# 풀려나는지 확인한다.
# 방법: 하나뿐인 모델 스레드를 gate.wait로 막아 두고, 그 뒤에 줄 선 호출을 취소한 다음, 인수 객체의 약한
# 참조가 비었는지 본다. inference._run은 취소되면 대기 중인 작업 칸(job[0])을 None으로 비워 인수를 놓는다.
# gate.set()은 finally에서 불러 테스트가 실패해도 스레드를 풀어 준다.
async def test_a_cancelled_waiting_call_lets_go_of_its_arguments():
    """A live pause queues a whole utterance's audio for the model thread, and the resume that follows
    cancels it; the audio must not stay in the thread's queue until the thread gets there."""
    gate = threading.Event()
    blocker = asyncio.create_task(run_model(gate.wait, 5))  # a model call holds the only thread
    await asyncio.sleep(0.05)
    try:
        audio = Audio()
        released = weakref.ref(audio)
        waiting = asyncio.create_task(run_model(id, audio))
        await asyncio.sleep(0)  # the call now waits in the thread's queue
        del audio
        waiting.cancel()
        await asyncio.sleep(0.05)
        assert waiting.cancelled()
        # 취소된 작업은 예외 traceback을 들고 있고 그 프레임이 호출 인수를 참조하므로, 작업 객체까지 지운 뒤
        # gc.collect()로 순환 참조를 정리하고 확인한다.
        del waiting  # a cancelled task keeps its traceback, and with it the call's arguments
        gc.collect()
        assert released() is None
    finally:
        gate.set()
        await blocker
