import asyncio
import io
import time
import wave

import pytest

from app.services.inference import run_model
from app.services.interfaces import (
    InvalidAudioError,
    ModelError,
    SpeechToText,
    TextToSpeech,
    Translator,
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
    with pytest.raises(InvalidAudioError):
        FakeSpeechToText().transcribe(b"", "ko")


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
