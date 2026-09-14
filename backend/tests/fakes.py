"""Deterministic stand-ins for the models, for tests of the pipeline and the API.

They follow the interfaces in app.services.interfaces, need no GPU or downloads, and can
be told to fail, so the "no server audio" path of the API contract can be tested too.
"""

import io
import wave

from app.services.interfaces import (
    Language,
    ModelError,
    NoSpeechError,
    SynthesizedAudio,
    Transcript,
    UndecodableAudioError,
)

SAMPLE_RATE = 16_000


def silent_wav(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """A valid 16-bit mono WAV file of silence."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(b"\x00\x00" * (sample_rate * duration_ms // 1000))
    return buffer.getvalue()


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

    # What detect_language answers (T35); tests set them.
    language: Language = "ko"
    confidence: float = 1.0

    def detect_language(self, audio: bytes) -> tuple[Language, float]:
        if self.error:
            raise self.error
        if not audio:
            raise UndecodableAudioError("empty audio")
        return self.language, self.confidence


class FakeLiveSpeechToText(FakeSpeechToText):
    """Hears one word per 0.1 s of 16 kHz 16-bit audio, in live updates and finals alike, so the final text
    of a clip is its last update's text. `live_error` makes the live updates fail."""

    def __init__(
        self, text: str | None = None, error: ModelError | None = None, live_error: ModelError | None = None
    ) -> None:
        super().__init__(text, error)
        self.live_error = live_error

    @staticmethod
    def words(pcm_bytes: int) -> str:
        return " ".join(f"w{i}" for i in range(pcm_bytes // 3200))

    def transcribe_live(self, pcm: bytes, language: Language) -> str:
        if self.live_error:
            raise self.live_error
        return self.words(len(pcm))

    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        if self.text is not None or self.error or not audio:
            return super().transcribe(audio, language)
        text = self.words(len(audio) - 44)  # after the 44-byte WAV header
        if not text:
            raise NoSpeechError("no speech was recognized")
        return Transcript(text=text, language=language, duration_ms=(len(audio) - 44) // 32)


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


class FakeTextToSpeech:
    model_name = "fake-tts"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        if self.fail:
            raise ModelError("synthesis failed on purpose")
        duration_ms = 100 * max(1, len(text.split()))
        return SynthesizedAudio(wav=silent_wav(duration_ms), sample_rate=SAMPLE_RATE, duration_ms=duration_ms)
