from types import SimpleNamespace

import pytest
from av.error import InvalidDataError

from app.services import stt
from app.services.interfaces import NoSpeechError, UndecodableAudioError
from tests.fakes import silent_wav


class FakeWhisperModel:
    """Stands in for faster-whisper: returns fixed segment texts, or raises when decoding."""

    segments: list[str] = []
    error: Exception | None = None

    def __init__(self, model_size, device, compute_type):
        pass

    def transcribe(self, audio, language, beam_size):
        if self.error:
            raise self.error
        return (SimpleNamespace(text=text) for text in self.segments), SimpleNamespace(duration=1.5)


@pytest.fixture
def whisper(monkeypatch):
    monkeypatch.setattr(stt, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(stt, "add_cuda_dll_dirs", lambda: None)
    monkeypatch.setattr(FakeWhisperModel, "segments", [])
    monkeypatch.setattr(FakeWhisperModel, "error", None)
    return stt.WhisperSpeechToText("tiny", device="cpu", compute_type="int8")


def test_joins_segments_and_reports_the_duration(whisper):
    FakeWhisperModel.segments = [" 안녕하세요. ", "반갑습니다."]
    transcript = whisper.transcribe(silent_wav(1500), "ko")
    assert transcript.text == "안녕하세요. 반갑습니다."
    assert transcript.duration_ms == 1500


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
