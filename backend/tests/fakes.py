"""Deterministic stand-ins for the models, for tests of the pipeline and the API.

They follow the interfaces in app.services.interfaces, need no GPU or downloads, and can
be told to fail, so the "no server audio" path of the API contract can be tested too.
"""

# [파일 역할] 실제 모델(Whisper, opus-mt, MeloTTS/Kokoro, Ollama 교정기) 대신 쓰는 가짜 구현 모음이다.
# app/services/interfaces.py의 Protocol(SpeechToText, LiveSpeechToText, LanguageDetector, Translator,
# TextToSpeech, TypoCorrector)과 같은 메서드를 갖춰, 파이프라인·API 테스트가 GPU나 모델 내려받기 없이
# 돈다(README: '모델은 가짜 구현으로 대체한다').
# [설계 이유] 결과가 입력에서 결정적으로 정해져(예: 번역은 '[ko->en] 원문') 테스트가 기대값을 정확히 쓸 수
# 있고, error·fail 인자로 일부러 실패시켜 오류 응답(503, tts_error 등)의 계약도 검증할 수 있다.
# [연결] test_pipeline.py, test_translate_api.py, test_dialog_api.py, test_live_api.py, test_health.py,
# test_model_services.py 등이 가져다 쓴다. tests/__init__.py가 tests를 일반 패키지로 만들어 'tests.fakes'가
# 다른 설치 패키지의 tests와 섞이지 않게 한다.

import io
import wave

# 실제 서비스와 같은 결과 형식(Transcript, SynthesizedAudio)과 예외 계층(ModelError, NoSpeechError,
# UndecodableAudioError)을 써서 API가 가짜와 진짜를 똑같이 다루게 한다.
from app.services.interfaces import (
    Language,
    ModelError,
    NoSpeechError,
    SynthesizedAudio,
    Transcript,
    UndecodableAudioError,
)

# 가짜 음성의 표본화율(16 kHz, 서비스 입력과 같음).
SAMPLE_RATE = 16_000


# 주어진 길이의 무음 WAV를 만든다. 업로드 테스트의 입력과 FakeTextToSpeech의 출력에 쓴다. 16비트 모노라 1ms당
# 32바이트다(16,000표본/초 × 2바이트).
def silent_wav(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """A valid 16-bit mono WAV file of silence."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(b"\x00\x00" * (sample_rate * duration_ms // 1000))
    return buffer.getvalue()


# 가짜 음성 인식기. 빈 음성 → UndecodableAudioError(디코딩 실패), text='' → NoSpeechError(말 없음), error를
# 주면 모든 호출이 그 예외로 실패한다. text를 주지 않으면 '<언어> speech of N bytes'를 돌려줘 어떤 입력이
# 들어왔는지 테스트가 확인할 수 있다.
# duration_ms는 바이트 수 ÷ 32로 대충 계산한다(16 kHz 16비트 모노 기준).
class FakeSpeechToText:
    """Empty audio is undecodable and `text=""` finds no speech; `error` makes every call fail with it."""

    model_name = "fake-stt"

    def __init__(self, text: str | None = None, error: ModelError | None = None) -> None:
        self.text = text
        self.error = error

    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        if self.error:
            raise self.error
        if not audio:
            raise UndecodableAudioError("empty audio")
        if self.text == "":
            raise NoSpeechError("no speech was recognized")
        # 16 kHz, 16-bit mono PCM is 32 bytes per millisecond; good enough for a fake.
        text = self.text if self.text is not None else f"{language} speech of {len(audio)} bytes"
        return Transcript(text=text, language=language, duration_ms=len(audio) // 32)

    # 두 사람 대화의 언어 판별(T35) 흉내. 테스트가 클래스 속성 language·confidence를 바꿔 '확신이 낮을 때'(0.6
    # 미만) 같은 경우를 만든다.
    # What detect_language answers (T35); tests set them.
    language: Language = "ko"
    confidence: float = 1.0

    def detect_language(self, audio: bytes) -> tuple[Language, float]:
        if self.error:
            raise self.error
        if not audio:
            raise UndecodableAudioError("empty audio")
        return self.language, self.confidence


# 동시통역 자막 테스트용 인식기. 16 kHz 16비트 소리 0.1초(3,200바이트)마다 단어 하나(w0, w1, ...)를 듣는다.
# 자막 갱신(transcribe_live)과 최종 인식(transcribe)이 같은 규칙이라, 한 마디의 최종 원문이 마지막 갱신 글과
# 같아진다. live_error로 자막 갱신만 실패시킬 수 있다.
class FakeLiveSpeechToText(FakeSpeechToText):
    """Hears one word per 0.1 s of 16 kHz 16-bit audio, in live updates and finals alike, so the final text
    of a clip is its last update's text. `live_error` makes the live updates fail."""

    def __init__(
        self, text: str | None = None, error: ModelError | None = None, live_error: ModelError | None = None
    ) -> None:
        super().__init__(text, error)
        self.live_error = live_error

    # PCM 바이트 수 → 그 길이만큼의 가짜 단어 열. test_live_api.py도 기대값을 만들 때 이 함수를 쓴다.
    @staticmethod
    def words(pcm_bytes: int) -> str:
        return " ".join(f"w{i}" for i in range(pcm_bytes // 3200))

    def transcribe_live(self, pcm: bytes, language: Language) -> str:
        if self.live_error:
            raise self.live_error
        return self.words(len(pcm))

    # text·error를 주었거나 빈 음성이면 부모 동작을 따른다. 아니면 WAV 머리(44바이트)를 뺀 PCM 길이로 단어를
    # 만들고, 단어가 하나도 없으면(0.1초 미만) 말 없음으로 본다.
    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        if self.text is not None or self.error or not audio:
            return super().transcribe(audio, language)
        text = self.words(len(audio) - 44)  # after the 44-byte WAV header
        if not text:
            raise NoSpeechError("no speech was recognized")
        return Transcript(text=text, language=language, duration_ms=(len(audio) - 44) // 32)


# 가짜 번역기: '[원문언어->대상언어] 원문'을 돌려준다. 같은 언어로 번역하라면 ModelError를 낸다(실제
# DirectionalTranslator도 모델이 없는 방향이면 ModelError를 낸다). error를 주면 모델이 죽었거나 메모리가
# 부족한 경우처럼 모든 호출이 실패한다.
class FakeTranslator:
    """`error` makes every call fail with it, like a model that crashed or ran out of memory."""

    model_name = "fake-mt"

    def __init__(self, error: ModelError | None = None) -> None:
        self.error = error

    def translate(self, text: str, source: Language, target: Language) -> str:
        if self.error:
            raise self.error
        if source == target:
            raise ModelError("source and target languages are the same")
        return f"[{source}->{target}] {text}"


# 가짜 오타 교정기. 서버 교정기처럼 영어만 교정 대상이다(6절 결정: 한국어 입력은 교정하지 않음). fixed가
# 답이고, None이면 Ollama가 꺼져 교정하지 못한 경우처럼 행동한다(파이프라인은 입력 그대로 번역해야 함).
# calls에 받은 입력을 쌓아 교정이 불렸는지·무엇으로 불렸는지 테스트가 확인한다.
class FakeCorrector:
    """Corrects English only, like the server's. `fixed` is its answer; None acts like Ollama being down."""

    model_name = "fake-corrector"

    def __init__(self, fixed: str | None = "corrected text") -> None:
        self.fixed = fixed
        self.calls: list[tuple[str, Language]] = []

    def corrects(self, language: Language) -> bool:
        return language == "en"

    def correct(self, text: str, language: Language) -> str | None:
        self.calls.append((text, language))
        return self.fixed

    def close(self) -> None:
        self.closed = True


# 가짜 음성 합성기: 단어 수 × 0.1초(최소 0.1초)의 무음 WAV. fail=True면 ModelError로 실패해, 번역은 돌려주고
# tts_error를 싣는 경로(README '음성 합성이 실패해도 번역은 돌려준다')를 테스트할 수 있다.
class FakeTextToSpeech:
    model_name = "fake-tts"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        if self.fail:
            raise ModelError("synthesis failed on purpose")
        duration_ms = 100 * max(1, len(text.split()))
        return SynthesizedAudio(wav=silent_wav(duration_ms), sample_rate=SAMPLE_RATE, duration_ms=duration_ms)
