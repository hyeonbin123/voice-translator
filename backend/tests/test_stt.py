from types import SimpleNamespace

import pytest
from av.error import InvalidDataError

from app.services import stt
from app.services.interfaces import NoSpeechError, UndecodableAudioError
from tests.fakes import silent_wav


class FakeWhisperModel:
    """Stands in for faster-whisper: returns fixed segment texts, or raises when decoding.

    segments_without_no_speech, when set, is what a pass with the no-speech check off returns.
    """

    segments: list[str] = []
    segments_without_no_speech: list[str] | None = None
    error: Exception | None = None
    calls: list[dict] = []

    def __init__(self, model_size, device, compute_type):
        pass

    def transcribe(self, audio, language, **options):
        self.calls.append(options)
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
    return lambda **options: stt.WhisperSpeechToText("tiny", device="cpu", compute_type="int8", **options)


@pytest.fixture
def whisper(make_whisper):
    return make_whisper()


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


def test_vad_and_threshold_options_reach_faster_whisper(make_whisper):
    FakeWhisperModel.segments = ["hello"]
    make_whisper(vad_filter=True, no_speech_threshold=None).transcribe(silent_wav(1000), "en")
    assert FakeWhisperModel.calls == [{"beam_size": 5, "vad_filter": True, "no_speech_threshold": None}]
