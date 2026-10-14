"""번역 흐름이 음성 인식·번역·음성 합성 모델에 기대하는 것.

실제 모델을 아직 측정으로 고르는 동안, 번역 흐름과 API를 이 인터페이스에 맞춰 tests/fakes.py의 가짜
구현으로 만들고 테스트했다. 구현은 동기 함수이고 CPU나 GPU를 쥐므로, 호출하는 쪽은 이벤트 루프에서 직접
부르지 않고 app.services.inference.run_model을 거쳐 실행한다.
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
    """모델이 이 입력에 대한 결과를 만들지 못했다."""


class InvalidAudioError(ModelError):
    """음성을 디코딩할 수 없거나, 인식할 소리가 없다.

    두 하위 클래스 중 하나를 raise한다. API는 각각을 고정된 문구로 답한다(docs/api.md).
    """


class UndecodableAudioError(InvalidAudioError):
    """업로드가 비었거나 디코딩할 수 있는 음성이 아니다."""


class NoSpeechError(InvalidAudioError):
    """음성은 디코딩되지만 그 안에서 말소리를 인식하지 못했다."""


# 인식 결과. frozen=True라 만든 뒤 바뀌지 않는다. duration_ms는 인식한 음성의 길이(ms).
@dataclass(frozen=True)
class Transcript:
    text: str
    language: Language
    duration_ms: int


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
        """`language`로 말한 `audio`(wav·webm 같은 인코딩된 파일)의 말소리를 인식한다.

        호출하는 쪽이 사용자가 고른 언어를 넘기므로 구현은 언어를 판별하지 않는다.
        """
        ...


# 동시통역 자막용 인식(transcribe_live)이 더 있는 인식기. routers/health.py와 routers/live.py가 isinstance로
# 동시통역을 쓸 수 있는지 판단한다.
@runtime_checkable
class LiveSpeechToText(SpeechToText, Protocol):
    def transcribe_live(self, pcm: bytes, language: Language) -> str:
        """동시통역 자막용으로, 지금까지의 한 마디(16kHz 모노 16비트 little-endian PCM)를 인식한다.

        transcribe라면 NoSpeechError를 낼 상황에서는 빈 문자열을 돌려준다.
        """
        ...


# 두 사람 대화의 언어 판별(Whisper의 판별에서 한국어·영어만 비교). 반환값은 (언어, 두 확률 중 그 언어의 몫
# 0.5~1.0). routers/translate.py와 routers/live.py가 isinstance로 판별 기능이 있는지 확인한다.
@runtime_checkable
class LanguageDetector(Protocol):
    def detect_language(self, audio: bytes) -> tuple[Language, float]:
        """인코딩된 음성 파일이 한국어인지 영어인지와, 두 확률 중 그 언어가 차지하는 몫을 돌려준다.

        transcribe처럼 UndecodableAudioError나 NoSpeechError를 낸다.
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
    """글자로 입력한 글의 오타를 번역 전에 고친다.

    correct()는 이 프로세스 안의 모델 작업이 아니라 막는(blocking) 네트워크 입출력이므로, 번역 흐름은
    run_model 대신 run_correction(전용 스레드, 한 번에 한 호출)으로 실행한다.
    """

    model_name: str

    def corrects(self, language: Language) -> bool: ...

    def correct(self, text: str, language: Language) -> str | None:
        """교정한 글을 돌려준다. 어떤 실패든 None이고, 그러면 호출한 쪽이 입력 그대로 번역한다."""
        ...

    def close(self) -> None:
        """앱이 종료될 때 한 번 불린다. 막으면(기다리면) 안 된다."""
        ...
