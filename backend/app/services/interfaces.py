"""What the translation pipeline needs from the speech recognition, translation and
speech synthesis models.

The pipeline and the API are built and tested against these interfaces with the fakes
in tests/fakes.py, while the real models are still being chosen by measurement.
Implementations are synchronous and hold the CPU or GPU, so callers run them through
app.services.inference.run_model instead of calling them on the event loop.
"""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Language = Literal["en", "ko"]


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


@dataclass(frozen=True)
class Transcript:
    text: str
    language: Language
    duration_ms: int  # length of the recognized audio


@dataclass(frozen=True)
class SynthesizedAudio:
    wav: bytes
    sample_rate: int
    duration_ms: int


@runtime_checkable
class SpeechToText(Protocol):
    model_name: str

    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        """Recognize speech in `audio` (an encoded file such as wav or webm) spoken in `language`.

        The caller passes the language the user picked, so implementations don't detect it.
        """
        ...


@runtime_checkable
class LiveSpeechToText(SpeechToText, Protocol):
    def transcribe_live(self, pcm: bytes, language: Language) -> str:
        """Recognize 16 kHz mono 16-bit little-endian PCM, an utterance so far, for live subtitles (T34).

        Returns an empty string where transcribe would raise NoSpeechError.
        """
        ...


@runtime_checkable
class Translator(Protocol):
    model_name: str

    def translate(self, text: str, source: Language, target: Language) -> str: ...


@runtime_checkable
class TextToSpeech(Protocol):
    model_name: str

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio: ...


@runtime_checkable
class TypoCorrector(Protocol):
    """Fixes typing mistakes in typed text before translation (T32)."""

    model_name: str

    def corrects(self, language: Language) -> bool: ...

    def correct(self, text: str, language: Language) -> str | None:
        """The corrected text, or None on any failure: the caller then translates the text as typed."""
        ...

    def close(self) -> None:
        """Called once when the app shuts down; must not block."""
        ...
