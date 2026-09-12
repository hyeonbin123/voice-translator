import io
import wave

import numpy as np
import pytest

from app.services.interfaces import ModelError
from app.services.tts import LanguageTextToSpeech, _check_text, _guarded, wav_bytes
from tests.fakes import FakeTextToSpeech


def test_wav_bytes_writes_16_bit_mono_pcm_and_clips():
    audio = wav_bytes(np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32), 16_000)
    with wave.open(io.BytesIO(audio.wav)) as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 16_000)
        samples = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2")
    assert samples.tolist() == [0, 16383, -16383, 32767, -32767]
    assert audio.sample_rate == 16_000
    assert audio.duration_ms == 0  # 5 samples at 16 kHz round to 0 ms


def test_wav_bytes_reports_the_duration():
    assert wav_bytes(np.zeros(24_000), 24_000).duration_ms == 1000


def test_empty_audio_is_a_model_error():
    with pytest.raises(ModelError):
        wav_bytes(np.zeros(0), 16_000)


def test_guarded_turns_engine_failures_into_model_errors():
    def crash():
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(ModelError, match="RuntimeError"):
        _guarded(crash)
    assert _guarded(lambda: 42) == 42


def test_text_and_language_are_checked():
    with pytest.raises(ModelError):
        _check_text("   ", "ko", ("ko",))
    with pytest.raises(ModelError, match="does not speak ko"):
        _check_text("안녕", "ko", ("en",))


def test_language_router_uses_each_model_and_rejects_missing_languages():
    router = LanguageTextToSpeech({"en": FakeTextToSpeech()})
    assert router.synthesize("hello there", "en").duration_ms == 200
    assert router.model_name == "en: fake-tts"
    with pytest.raises(ModelError, match="no speech synthesis model for ko"):
        router.synthesize("안녕", "ko")
