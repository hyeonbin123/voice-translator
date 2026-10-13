"""What the translation pipeline needs from the speech recognition, translation and
speech synthesis models.

The pipeline and the API are built and tested against these interfaces with the fakes
in tests/fakes.py, while the real models are still being chosen by measurement.
Implementations are synchronous and hold the CPU or GPU, so callers run them through
app.services.inference.run_model instead of calling them on the event loop.
"""

# [파일 역할] 번역 흐름이 모델에 기대하는 약속(Protocol)과 주고받는 값(Transcript, SynthesizedAudio), 모델
# 오류의 종류를 정의한다.
# [설계 이유] 모델이 측정으로 정해지기 전에 API·화면을 나란히 만들려고, 실제 모델 대신 이 인터페이스와 가짜
# 구현(tests/fakes.py)으로 먼저 만들고 테스트했다(docs/plan.md 9장 '두 가지 약속'). Protocol(구조적 타입)이라
# 구현 클래스가 상속하지 않아도 같은 메서드만 있으면 된다.
# [연결] 구현: services/stt.py(Whisper), translation.py(opus-mt, Hy-MT2), tts.py(MeloTTS, Kokoro, Supertonic),
# correction.py(Ollama). 사용: services/pipeline.py, models.py, routers.

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

# 지원 언어. 스키마·서비스·DB 제약이 모두 이 두 값만 허용한다.
Language = Literal["en", "ko"]


# 모델 오류의 뿌리. 라우터는 ModelError를 503 "Translation service is unavailable"로 바꾼다(docs/api.md). 그
# 아래 InvalidAudioError 갈래는 사용자의 음성 문제라 422로 구분해 답한다.
class ModelError(Exception):
    """The model could not produce a result for this input."""


class InvalidAudioError(ModelError):
    """The audio could not be decoded, or holds no sound to recognize.

    Raise one of the two subclasses; the API answers each with its own fixed message (docs/api.md).
    """


class UndecodableAudioError(InvalidAudioError):
    """The upload is empty or is not audio that can be decoded."""


class NoSpeechError(InvalidAudioError):
    """The audio decodes, but no speech was recognized in it."""


# 인식 결과. frozen=True라 만든 뒤 바뀌지 않는다. duration_ms는 인식한 음성의 길이(ms).
@dataclass(frozen=True)
class Transcript:
    text: str
    language: Language
    duration_ms: int  # length of the recognized audio


# 합성 결과: WAV 파일 바이트(헤더 포함), 표본 추출률, 길이(ms). wav는 그대로 디스크에 저장되고 GET
# /api/audio로 나간다.
@dataclass(frozen=True)
class SynthesizedAudio:
    wav: bytes
    sample_rate: int
    duration_ms: int


# 음성 인식. model_name은 기록·상태 확인에 남는 모델 식별자다. runtime_checkable이라 isinstance로 구현 여부를
# 확인할 수 있다(멤버가 있는지만 보고 시그니처는 보지 않는다).
@runtime_checkable
class SpeechToText(Protocol):
    model_name: str

    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        """Recognize speech in `audio` (an encoded file such as wav or webm) spoken in `language`.

        The caller passes the language the user picked, so implementations don't detect it.
        """
        ...


# 동시통역 자막용 인식(transcribe_live)이 더 있는 인식기. routers/health.py와 routers/live.py가 isinstance로
# 동시통역을 쓸 수 있는지 판단한다.
@runtime_checkable
class LiveSpeechToText(SpeechToText, Protocol):
    def transcribe_live(self, pcm: bytes, language: Language) -> str:
        """Recognize 16 kHz mono 16-bit little-endian PCM, an utterance so far, for live subtitles (T34).

        Returns an empty string where transcribe would raise NoSpeechError.
        """
        ...


# 두 사람 대화의 언어 판별(Whisper의 판별에서 한국어·영어만 비교). 반환값은 (언어, 두 확률 중 그 언어의 몫
# 0.5~1.0). routers/translate.py와 routers/live.py가 isinstance로 판별 기능이 있는지 확인한다.
@runtime_checkable
class LanguageDetector(Protocol):
    def detect_language(self, audio: bytes) -> tuple[Language, float]:
        """Korean or English, from an encoded audio file, and its share of the two probabilities (T35).

        Raises UndecodableAudioError or NoSpeechError as transcribe does.
        """
        ...


# 번역기와 음성 합성기. 둘 다 동기 함수이고 실패하면 ModelError를 낸다.
@runtime_checkable
class Translator(Protocol):
    model_name: str

    def translate(self, text: str, source: Language, target: Language) -> str: ...


@runtime_checkable
class TextToSpeech(Protocol):
    model_name: str

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio: ...


# 오타 교정기. corrects(language)는 그 언어를 지금 고칠 수 있는지 알려 준다(OllamaCorrector는 교정 대상
# 언어이고, 모델 준비가 끝났고, 닫히지 않았을 때만 True).
@runtime_checkable
class TypoCorrector(Protocol):
    """Fixes typing mistakes in typed text before translation (T32).

    correct() is blocking network I/O, not model work in this process, so the pipeline runs it with
    run_correction (a thread of its own, one call at a time) instead of run_model.
    """

    model_name: str

    def corrects(self, language: Language) -> bool: ...

    def correct(self, text: str, language: Language) -> str | None:
        """The corrected text, or None on any failure: the caller then translates the text as typed."""
        ...

    def close(self) -> None:
        """Called once when the app shuts down; must not block."""
        ...
