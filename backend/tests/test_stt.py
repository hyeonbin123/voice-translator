import gc
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from av.error import InvalidDataError
from faster_whisper.audio import decode_audio as faster_whisper_decode_audio

from app.services import stt
from app.services.interfaces import ModelError, NoSpeechError, UndecodableAudioError
from tests.fakes import silent_wav


class FakeWhisperModel:
    """Stands in for faster-whisper: returns fixed segment texts, or raises when decoding.

    segments_without_no_speech, when set, is what a pass with the no-speech check off returns.
    """

    segments: list[str] = []
    segments_without_no_speech: list[str] | None = None
    error: Exception | None = None
    calls: list[dict] = []
    inputs: list = []

    def __init__(self, model_size, device, compute_type):
        pass

    def transcribe(self, audio, language, **options):
        self.calls.append(options)
        self.inputs.append(audio)
        if self.error:
            raise self.error
        texts = self.segments
        if options["no_speech_threshold"] is None and self.segments_without_no_speech is not None:
            texts = self.segments_without_no_speech
        return (SimpleNamespace(text=text) for text in texts), SimpleNamespace(duration=1.5)


@pytest.fixture
def make_whisper(monkeypatch):
    monkeypatch.setattr(stt, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(stt, "add_cuda_dll_dirs", lambda: None)
    monkeypatch.setattr(FakeWhisperModel, "segments", [])
    monkeypatch.setattr(FakeWhisperModel, "segments_without_no_speech", None)
    monkeypatch.setattr(FakeWhisperModel, "error", None)
    monkeypatch.setattr(FakeWhisperModel, "calls", [])
    monkeypatch.setattr(FakeWhisperModel, "inputs", [])
    return lambda **options: stt.WhisperSpeechToText("tiny", device="cpu", compute_type="int8", **options)


@pytest.fixture
def whisper(make_whisper):
    return make_whisper()


def test_an_engine_error_on_the_call_becomes_a_model_error(make_whisper, monkeypatch):
    # T48: CTranslate2 raises RuntimeError (CUDA errors among others); the API answers 503, not 500.
    monkeypatch.setattr(FakeWhisperModel, "error", RuntimeError("CUDA failed on private words"))
    with pytest.raises(ModelError) as caught:
        make_whisper().transcribe(silent_wav(100), "en")
    assert "private words" not in str(caught.value)


def test_an_engine_error_while_reading_the_segments_becomes_a_model_error(make_whisper, monkeypatch):
    # faster-whisper returns a generator; the model runs, and fails, while the segments are read.
    def segments():
        yield SimpleNamespace(text="partial")
        raise RuntimeError("CUDA failed on private words")

    monkeypatch.setattr(
        FakeWhisperModel, "transcribe", lambda self, audio, language, **options: (segments(), None)
    )
    with pytest.raises(ModelError):
        make_whisper().transcribe(silent_wav(100), "en")


def test_joins_segments_and_reports_the_duration(whisper):
    FakeWhisperModel.segments = [" 안녕하세요. ", "반갑습니다."]
    transcript = whisper.transcribe(silent_wav(1500), "ko")
    assert transcript.text == "안녕하세요. 반갑습니다."
    assert transcript.duration_ms == 1500


def test_default_options_are_faster_whispers_defaults(whisper):
    FakeWhisperModel.segments = ["hello"]
    whisper.transcribe(silent_wav(1000), "en")
    assert FakeWhisperModel.calls == [{"beam_size": 5, "vad_filter": False, "no_speech_threshold": 0.6}]


def test_empty_upload_is_undecodable(whisper):
    with pytest.raises(UndecodableAudioError):
        whisper.transcribe(b"", "ko")


def test_audio_that_cannot_be_decoded_is_undecodable(whisper):
    FakeWhisperModel.error = InvalidDataError(1094995529, "Invalid data found when processing input")
    with pytest.raises(UndecodableAudioError):
        whisper.transcribe(b"not audio", "ko")


def test_audio_without_speech_is_reported_as_no_speech(whisper):
    FakeWhisperModel.segments = ["  "]
    with pytest.raises(NoSpeechError):
        whisper.transcribe(silent_wav(1000), "en")
    assert len(FakeWhisperModel.calls) == 1  # no second pass unless asked for


def test_empty_result_is_retried_without_the_no_speech_check(make_whisper):
    FakeWhisperModel.segments_without_no_speech = ["다시 들었습니다."]
    transcript = make_whisper(retry_without_no_speech=True).transcribe(silent_wav(1000), "ko")
    assert transcript.text == "다시 들었습니다."
    assert [call["no_speech_threshold"] for call in FakeWhisperModel.calls] == [0.6, None]


def test_retry_happens_only_when_the_first_pass_is_empty(make_whisper):
    FakeWhisperModel.segments = ["hello"]
    FakeWhisperModel.segments_without_no_speech = ["something else"]
    transcript = make_whisper(retry_without_no_speech=True).transcribe(silent_wav(1000), "en")
    assert transcript.text == "hello"
    assert len(FakeWhisperModel.calls) == 1


def test_retry_that_still_finds_nothing_is_no_speech(make_whisper):
    with pytest.raises(NoSpeechError):
        make_whisper(retry_without_no_speech=True).transcribe(silent_wav(1000), "ko")
    assert len(FakeWhisperModel.calls) == 2


def tone_wav(seconds: float = 1.5, rate: int = 8000) -> bytes:
    """A 440 Hz tone at a rate other than 16 kHz, so decoding has to resample."""
    t = np.arange(int(rate * seconds)) / rate
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def test_own_decode_gives_faster_whispers_samples_without_collecting_garbage(monkeypatch):
    audio = tone_wav()
    expected = faster_whisper_decode_audio(io.BytesIO(audio))
    collections = []
    monkeypatch.setattr(gc, "collect", lambda *args: collections.append(args) or 0)
    samples = stt.decode_audio(audio)
    assert samples.dtype == np.float32
    assert len(samples) == 24000  # 1.5 s at 16 kHz
    assert np.array_equal(samples, expected)
    assert collections == []


def test_own_decode_hands_faster_whisper_the_samples(make_whisper):
    FakeWhisperModel.segments = ["hello"]
    make_whisper(own_decode=True).transcribe(silent_wav(1000), "en")
    (model_input,) = FakeWhisperModel.inputs
    assert isinstance(model_input, np.ndarray)
    assert len(model_input) == 16000


def test_own_decode_reports_undecodable_audio(make_whisper):
    with pytest.raises(UndecodableAudioError):
        make_whisper(own_decode=True).transcribe(b"not audio", "en")
    assert FakeWhisperModel.inputs == []


def test_vad_and_threshold_options_reach_faster_whisper(make_whisper):
    FakeWhisperModel.segments = ["hello"]
    make_whisper(vad_filter=True, no_speech_threshold=None).transcribe(silent_wav(1000), "en")
    assert FakeWhisperModel.calls == [{"beam_size": 5, "vad_filter": True, "no_speech_threshold": None}]


def test_recognition_replicas_reach_faster_whisper_only_when_asked(monkeypatch):
    """STT_NUM_WORKERS (T61): the default leaves faster-whisper's own default of one replica."""
    from app.services import stt as stt_module

    made = []

    class Recording:
        def __init__(self, model_size, **options):
            made.append(options)

    monkeypatch.setattr(stt_module, "WhisperModel", Recording)
    stt_module.WhisperSpeechToText(device="cpu", compute_type="int8")
    stt_module.WhisperSpeechToText(device="cpu", compute_type="int8", num_workers=2)
    assert made == [
        {"device": "cpu", "compute_type": "int8"},
        {"device": "cpu", "compute_type": "int8", "num_workers": 2},
    ]
