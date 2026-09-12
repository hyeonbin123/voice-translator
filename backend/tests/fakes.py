"""Deterministic stand-ins for the models, for tests of the pipeline and the API.

They follow the interfaces in app.services.interfaces, need no GPU or downloads, and can
be told to fail, so the "no server audio" path of the API contract can be tested too.
"""

import io
import wave

from app.services.interfaces import (
    InvalidAudioError,
    Language,
    ModelError,
    SynthesizedAudio,
    Transcript,
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
    model_name = "fake-stt"

    def __init__(self, text: str | None = None) -> None:
        self.text = text

    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        if not audio:
            raise InvalidAudioError("empty audio")
        # 16 kHz, 16-bit mono PCM is 32 bytes per millisecond; good enough for a fake.
        text = self.text if self.text is not None else f"{language} speech of {len(audio)} bytes"
        return Transcript(text=text, language=language, duration_ms=len(audio) // 32)


class FakeTranslator:
    model_name = "fake-mt"

    def translate(self, text: str, source: Language, target: Language) -> str:
        if source == target:
            raise ModelError("source and target languages are the same")
        return f"[{source}->{target}] {text}"


class FakeTextToSpeech:
    model_name = "fake-tts"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        if self.fail:
            raise ModelError("synthesis failed on purpose")
        duration_ms = 100 * max(1, len(text.split()))
        return SynthesizedAudio(
            wav=silent_wav(duration_ms), sample_rate=SAMPLE_RATE, duration_ms=duration_ms
        )
