# [파일 역할] 음성 인식 서비스 app/services/stt.py의 WhisperSpeechToText를 가짜 faster-whisper 모델로
# 검증한다. 실제 Whisper 가중치 없이 돈다.
# [무엇을 확인하나] 엔진 예외를 ModelError로 바꾸는 경계(API가 500이 아니라 503으로 답하게), 조각 글자 잇기와
# 길이 보고, 기본 디코딩 옵션, 읽을 수 없는 음성과 말소리 없음의 구분(각각 다른 422), 빈 결과 재시도
# 옵션(docs/experiments.md 1-1절), gc.collect() 없는 자체 디코딩(5절 후보 C), 옵션 전달.
# [왜 중요한가] 이 클래스의 예외 종류가 곧 API의 응답 코드와 메시지를 정하고, 옵션은 측정으로 고른 설정이 실제
# 엔진에 그대로 들어가는지를 정한다.

import gc
import io
import wave
from types import SimpleNamespace

# 외부 라이브러리. InvalidDataError는 PyAV(FFmpeg)가 디코딩할 수 없는 입력에 던지는 예외(FFmpegError
# 하위)이고, faster_whisper_decode_audio는 자체 디코딩이 원래 구현과 같은 표본을 내는지 견줄 기준이다.
import numpy as np
import pytest
from av.error import InvalidDataError
from faster_whisper.audio import decode_audio as faster_whisper_decode_audio

from app.services import stt
from app.services.interfaces import ModelError, NoSpeechError, UndecodableAudioError
from tests.fakes import silent_wav


# faster-whisper WhisperModel의 대역. 클래스 속성으로 돌려줄 조각 글자(segments), 무음 판정을 끈 두 번째
# 시도의 결과, 던질 예외를 정하고, 받은 옵션(calls)과 입력(inputs)을 기록해 테스트가 무엇이 엔진에 넘어갔는지
# 확인하게 한다. 실제 엔진처럼 조각을 생성기로, 정보 객체(duration)와 함께 돌려준다.
class FakeWhisperModel:
    """faster-whisper 대신 쓰인다: 정해 둔 조각 글자를 돌려주거나, 디코딩할 때 예외를 던진다.

    segments_without_no_speech를 정해 두면, 무음 판정을 끈 인식이 그 값을 돌려준다.
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


# 가짜 모델로 인식기를 만드는 함수를 돌려준다. 클래스 속성은 테스트 사이에 공유되므로 monkeypatch로 매번
# 초기값을 다시 넣어(테스트가 끝나면 원래대로 돌아감) 앞 테스트의 설정·기록이 새지 않게 한다. Windows CUDA DLL
# 등록(add_cuda_dll_dirs)도 끈다.
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


# 기본 옵션의 인식기.
@pytest.fixture
def whisper(make_whisper):
    return make_whisper()


# CTranslate2가 transcribe 호출에서 RuntimeError(CUDA 오류 등)를 던지면 ModelError로 바뀌고, 원래
# 메시지("private words")는 새 예외 메시지에 들어가지 않는지 확인한다(커밋 544f8d6). 메시지에는 예외 종류만
# 남는다. ModelError로 바뀌어야 API가 500이 아니라 503으로 답한다.
def test_an_engine_error_on_the_call_becomes_a_model_error(make_whisper, monkeypatch):
    monkeypatch.setattr(FakeWhisperModel, "error", RuntimeError("CUDA failed on private words"))
    with pytest.raises(ModelError) as caught:
        make_whisper().transcribe(silent_wav(100), "en")
    assert "private words" not in str(caught.value)


# faster-whisper는 조각을 생성기로 돌려주고 실제 계산은 조각을 읽을 때 일어나므로, 읽는 도중 난 예외도
# ModelError로 바뀌는지 확인한다. 예외 경계가 호출만 감싸고 읽기를 빠뜨리는 회귀를 잡는다.
def test_an_engine_error_while_reading_the_segments_becomes_a_model_error(make_whisper, monkeypatch):
    def segments():
        yield SimpleNamespace(text="partial")
        raise RuntimeError("CUDA failed on private words")

    monkeypatch.setattr(
        FakeWhisperModel, "transcribe", lambda self, audio, language, **options: (segments(), None)
    )
    with pytest.raises(ModelError):
        make_whisper().transcribe(silent_wav(100), "en")


# 조각 글자의 앞뒤 공백을 지우고 한 칸으로 이어 붙이며, 음성 길이(초)를 ms로 보고하는지 확인한다.
def test_joins_segments_and_reports_the_duration(whisper):
    FakeWhisperModel.segments = [" 안녕하세요. ", "반갑습니다."]
    transcript = whisper.transcribe(silent_wav(1500), "ko")
    assert transcript.text == "안녕하세요. 반갑습니다."
    assert transcript.duration_ms == 1500


# 생성자 기본값이 faster-whisper 자체 기본값(beam 5, VAD 끔, no_speech_threshold 0.6)과 같은지 확인한다.
# 서비스는 이 위에 설정값으로 VAD를 켠다(services/models.py).
def test_default_options_are_faster_whispers_defaults(whisper):
    FakeWhisperModel.segments = ["hello"]
    whisper.transcribe(silent_wav(1000), "en")
    assert FakeWhisperModel.calls == [{"beam_size": 5, "vad_filter": False, "no_speech_threshold": 0.6}]


# 빈 업로드는 엔진에 넘기지 않고 '읽을 수 없는 음성'(API 422 "Audio could not be decoded")으로 처리한다.
def test_empty_upload_is_undecodable(whisper):
    with pytest.raises(UndecodableAudioError):
        whisper.transcribe(b"", "ko")


# 엔진 안에서 FFmpeg 디코딩 오류가 나면 일반 ModelError(503)가 아니라 UndecodableAudioError(422)로 구분하는지
# 확인한다. 사용자의 잘못된 파일을 서버 장애로 답하지 않게 하는 것이다.
def test_audio_that_cannot_be_decoded_is_undecodable(whisper):
    FakeWhisperModel.error = InvalidDataError(1094995529, "Invalid data found when processing input")
    with pytest.raises(UndecodableAudioError):
        whisper.transcribe(b"not audio", "ko")


# 공백뿐인 결과는 NoSpeechError(API 422 "No speech was recognized")다. 재시도 옵션이 없으면 엔진을 한 번만
# 부른다.
def test_audio_without_speech_is_reported_as_no_speech(whisper):
    FakeWhisperModel.segments = ["  "]
    with pytest.raises(NoSpeechError):
        whisper.transcribe(silent_wav(1000), "en")
    assert len(FakeWhisperModel.calls) == 1  # 요청하지 않으면 두 번째 인식은 없다


# retry_without_no_speech를 켜면 첫 결과가 비었을 때 무음 판정(no_speech_threshold)을 끄고 한 번 더
# 인식한다(docs/experiments.md 1-1절의 빈 결과 줄이기 후보). 두 호출의 문턱이 0.6 → None 순서인지 본다.
def test_empty_result_is_retried_without_the_no_speech_check(make_whisper):
    FakeWhisperModel.segments_without_no_speech = ["다시 들었습니다."]
    transcript = make_whisper(retry_without_no_speech=True).transcribe(silent_wav(1000), "ko")
    assert transcript.text == "다시 들었습니다."
    assert [call["no_speech_threshold"] for call in FakeWhisperModel.calls] == [0.6, None]


# 첫 결과가 있으면 재시도하지 않는다. 재시도는 빈 결과에만 드는 추가 비용이어야 한다.
def test_retry_happens_only_when_the_first_pass_is_empty(make_whisper):
    FakeWhisperModel.segments = ["hello"]
    FakeWhisperModel.segments_without_no_speech = ["something else"]
    transcript = make_whisper(retry_without_no_speech=True).transcribe(silent_wav(1000), "en")
    assert transcript.text == "hello"
    assert len(FakeWhisperModel.calls) == 1


# 재시도도 비면 NoSpeechError이고 엔진은 정확히 두 번 불린다.
def test_retry_that_still_finds_nothing_is_no_speech(make_whisper):
    with pytest.raises(NoSpeechError):
        make_whisper(retry_without_no_speech=True).transcribe(silent_wav(1000), "ko")
    assert len(FakeWhisperModel.calls) == 2


# 8 kHz 440 Hz 사인파 WAV. 16 kHz가 아니라서 디코딩할 때 리샘플링 단계까지 거치게 하려는 입력이다.
def tone_wav(seconds: float = 1.5, rate: int = 8000) -> bytes:
    """16 kHz가 아닌 표본화율의 440 Hz 사인파. 디코딩이 리샘플링을 거쳐야 한다."""
    t = np.arange(int(rate * seconds)) / rate
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


# 자체 decode_audio가 faster-whisper의 decode_audio와 표본 하나까지 같은 float32 배열(1.5초 × 16 kHz =
# 24000개)을 내면서, gc.collect()는 한 번도 부르지 않는지 확인한다. faster-whisper의 디코딩 뒤 전체 가비지
# 컬렉션이 모델을 올린 프로세스에서 GIL을 잡아 다른 요청을 막았던 문제(docs/experiments.md 5절, README 설계
# 판단)의 해결 후보다. gc.collect는 monkeypatch로 호출 기록만 남기게 바꾼다.
def test_own_decode_gives_faster_whispers_samples_without_collecting_garbage(monkeypatch):
    audio = tone_wav()
    expected = faster_whisper_decode_audio(io.BytesIO(audio))
    collections = []
    monkeypatch.setattr(gc, "collect", lambda *args: collections.append(args) or 0)
    samples = stt.decode_audio(audio)
    assert samples.dtype == np.float32
    assert len(samples) == 24000  # 16 kHz로 1.5초
    assert np.array_equal(samples, expected)
    assert collections == []


# own_decode를 켜면 엔진에 바이트가 아니라 이미 디코딩한 표본 배열(1초 = 16000개)을 넘기는지 확인한다.
def test_own_decode_hands_faster_whisper_the_samples(make_whisper):
    FakeWhisperModel.segments = ["hello"]
    make_whisper(own_decode=True).transcribe(silent_wav(1000), "en")
    (model_input,) = FakeWhisperModel.inputs
    assert isinstance(model_input, np.ndarray)
    assert len(model_input) == 16000


# 자체 디코딩에서 실패해도 UndecodableAudioError이고, 엔진은 아예 불리지 않는지 확인한다.
def test_own_decode_reports_undecodable_audio(make_whisper):
    with pytest.raises(UndecodableAudioError):
        make_whisper(own_decode=True).transcribe(b"not audio", "en")
    assert FakeWhisperModel.inputs == []


# VAD 켜기·무음 문턱 끄기 같은 측정용 옵션이 그대로 엔진 호출 인자로 전달되는지 확인한다.
def test_vad_and_threshold_options_reach_faster_whisper(make_whisper):
    FakeWhisperModel.segments = ["hello"]
    make_whisper(vad_filter=True, no_speech_threshold=None).transcribe(silent_wav(1000), "en")
    assert FakeWhisperModel.calls == [{"beam_size": 5, "vad_filter": True, "no_speech_threshold": None}]


# STT_NUM_WORKERS(인식 모델 복제 수)는 1이 아닐 때만 WhisperModel에 넘긴다. 기본값에서 인자를 넘기지 않아야
# faster-whisper의 기본(복제 하나)을 그대로 쓰고, num_workers를 받지 않는 가짜 모델들도 계속 동작한다(커밋
# 685f965: 이 인자 때문에 다른 테스트가 깨졌던 일을 고침).
def test_recognition_replicas_reach_faster_whisper_only_when_asked(monkeypatch):
    """STT_NUM_WORKERS(T61): 기본값이면 faster-whisper 자체 기본값인 복제 하나를 그대로 쓴다."""
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
