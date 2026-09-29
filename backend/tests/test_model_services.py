import asyncio
import gc
import io
import threading
import time
import wave
import weakref

import pytest

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


def test_fakes_satisfy_the_model_interfaces():
    assert isinstance(FakeSpeechToText(), SpeechToText)
    assert isinstance(FakeTranslator(), Translator)
    assert isinstance(FakeTextToSpeech(), TextToSpeech)


def test_fake_speech_to_text_keeps_the_language_and_rejects_empty_audio():
    transcript = FakeSpeechToText(text="안녕하세요").transcribe(silent_wav(500), "ko")

    assert transcript.text == "안녕하세요"
    assert transcript.language == "ko"
    with pytest.raises(UndecodableAudioError):
        FakeSpeechToText().transcribe(b"", "ko")
    with pytest.raises(NoSpeechError):
        FakeSpeechToText(text="").transcribe(silent_wav(500), "ko")


def test_audio_errors_are_still_invalid_audio_and_model_errors():
    # Code that catches the parent classes keeps working after the split.
    for error in (UndecodableAudioError, NoSpeechError):
        assert issubclass(error, InvalidAudioError)
        assert issubclass(error, ModelError)


def test_fakes_can_be_told_to_fail_like_a_crashed_model():
    with pytest.raises(ModelError, match="out of memory"):
        FakeSpeechToText(error=ModelError("out of memory")).transcribe(silent_wav(500), "ko")
    with pytest.raises(ModelError, match="out of memory"):
        FakeTranslator(error=ModelError("out of memory")).translate("hello", "en", "ko")


def test_fake_translator_marks_the_direction():
    assert FakeTranslator().translate("hello", "en", "ko") == "[en->ko] hello"
    with pytest.raises(ModelError):
        FakeTranslator().translate("hello", "en", "en")


def test_fake_synthesis_returns_a_playable_wav_or_fails_on_request():
    audio = FakeTextToSpeech().synthesize("hello there", "en")

    with wave.open(io.BytesIO(audio.wav)) as clip:
        assert clip.getframerate() == audio.sample_rate
        assert clip.getnframes() * 1000 // clip.getframerate() == audio.duration_ms
    with pytest.raises(ModelError):
        FakeTextToSpeech(fail=True).synthesize("hello", "en")


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


async def test_model_calls_still_return_results_and_raise_errors():
    for run in (run_model, run_live_model):
        assert await run(sum, [1, 2, 3]) == 6
        with pytest.raises(ZeroDivisionError):
            await run(divmod, 1, 0)


class Audio:
    pass


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
        del waiting  # a cancelled task keeps its traceback, and with it the call's arguments
        gc.collect()
        assert released() is None
    finally:
        gate.set()
        await blocker
