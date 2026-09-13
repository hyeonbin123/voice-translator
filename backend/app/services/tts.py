"""Speech synthesis (task T4).

The candidates and the selection rule are in docs/experiments.md. Every implementation returns 16-bit
mono PCM WAV (docs/api.md). Model libraries are imported inside the constructors, so importing this
module needs none of them.
"""

from __future__ import annotations

import io
import os
import sys
import types
import wave
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np

from app.services.interfaces import Language, ModelError, SynthesizedAudio, TextToSpeech

T = TypeVar("T")
MMS_REPOS = {"ko": "facebook/mms-tts-kor", "en": "facebook/mms-tts-eng"}
MELO_VOICES = {"ko": ("KR", "KR"), "en": ("EN", "EN-US")}  # language code and speaker in MeloTTS


def wav_bytes(samples: np.ndarray, sample_rate: int) -> SynthesizedAudio:
    """Float samples in [-1, 1] to a 16-bit mono PCM WAV file."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ModelError("the model returned no audio")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())
    duration_ms = round(samples.size * 1000 / sample_rate)
    return SynthesizedAudio(wav=buffer.getvalue(), sample_rate=sample_rate, duration_ms=duration_ms)


def _guarded(run: Callable[[], T]) -> T:
    """Any failure inside the model (CUDA errors, text front-end errors) becomes a ModelError."""
    try:
        return run()
    except ModelError:
        raise
    except Exception as exc:  # noqa: BLE001 - the boundary around third-party models
        raise ModelError(f"the speech synthesis model failed: {type(exc).__name__}") from exc


def _check_text(text: str, language: Language, languages: tuple[Language, ...]) -> None:
    if language not in languages:
        raise ModelError(f"this model does not speak {language}")
    if not text.strip():
        raise ModelError("text is empty")


class MmsTextToSpeech:
    """Meta's MMS-TTS (VITS), one model per language. CC-BY-NC-4.0.

    The Korean model was trained on romanized text, so its input goes through uroman first.
    """

    def __init__(self, language: Language, device: str = "cuda") -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.language = language
        self.model_name = f"transformers/{MMS_REPOS[language]}"
        self._torch = torch
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(MMS_REPOS[language])
        self._model = VitsModel.from_pretrained(MMS_REPOS[language]).to(device).eval()
        self._romanize = None
        if getattr(self._tokenizer, "is_uroman", False):
            from uroman import Uroman

            self._romanize = Uroman().romanize_string

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        _check_text(text, language, (self.language,))

        def run() -> np.ndarray:
            prepared = self._romanize(text) if self._romanize else text
            inputs = self._tokenizer(prepared, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                return self._model(**inputs).waveform[0].float().cpu().numpy()

        return wav_bytes(_guarded(run), self._model.config.sampling_rate)


def _prepare_melo_on_windows() -> None:
    """Work around two MeloTTS import problems on Windows (docs/experiments.md, T4 candidates).

    - melo always imports its Japanese front end, which needs mecab-python3. Its `MeCab` folder is the
      same folder as python-mecab-ko's `mecab` on a case-insensitive file system, and Korean needs the
      latter, so an empty Japanese module stands in (Japanese is never used here).
    - g2pkk looks for `eunjeon` on Windows; python-mecab-ko offers the same pos() call.
    """
    if sys.platform != "win32":
        return
    if "melo.text.japanese" not in sys.modules:
        stub = types.ModuleType("melo.text.japanese")

        def distribute_phone(n_phone: int, n_word: int) -> list[int]:
            # Copied from melo.text.japanese; the English front end imports it from there.
            phones_per_word = [0] * n_word
            for _ in range(n_phone):
                phones_per_word[phones_per_word.index(min(phones_per_word))] += 1
            return phones_per_word

        stub.distribute_phone = distribute_phone  # type: ignore[attr-defined]
        sys.modules["melo.text.japanese"] = stub
    import g2pkk.g2pkk
    import mecab

    g2pkk.g2pkk.G2p.check_mecab = lambda self: None
    g2pkk.g2pkk.G2p.get_mecab = lambda self: mecab.MeCab()


class MeloTextToSpeech:
    """MeloTTS, one model per language. MIT. Pins transformers 4.27, so it needs its own environment."""

    def __init__(self, language: Language, device: str = "cuda") -> None:
        _prepare_melo_on_windows()
        from melo.api import TTS

        code, speaker = MELO_VOICES[language]
        self.language = language
        self.model_name = f"melotts/{code}/{speaker}"
        self._model = TTS(language=code, device=device)
        self._speaker = self._model.hps.data.spk2id[speaker]
        self._sample_rate = self._model.hps.data.sampling_rate

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        _check_text(text, language, (self.language,))
        audio = _guarded(lambda: self._model.tts_to_file(text, self._speaker, None, speed=1.0, quiet=True))
        return wav_bytes(audio, self._sample_rate)


def _espeak_data_on_windows() -> None:
    """Kokoro's English front end falls back to espeak-ng, whose bundled library cannot open a data path
    with non-ASCII characters on Windows, and ends the whole process when it tries. When the data sits
    under such a path (this project's folder does) and ESPEAK_DATA_PATH isn't set, work from the data's
    parent folder and pass a relative ASCII path. This changes the process's working directory, which is
    safe here because the app uses absolute paths only. Linux containers need nothing.
    """
    if sys.platform != "win32" or os.environ.get("ESPEAK_DATA_PATH"):
        return
    import espeakng_loader

    data = Path(espeakng_loader.get_data_path())
    if str(data).isascii():
        return
    os.chdir(data.parent)
    os.environ["ESPEAK_DATA_PATH"] = data.name


class KokoroTextToSpeech:
    """Kokoro-82M, English only. Apache-2.0. See _espeak_data_on_windows for its one Windows workaround."""

    def __init__(self, voice: str = "af_heart", device: str = "cuda") -> None:
        _espeak_data_on_windows()
        from kokoro import KPipeline

        self.model_name = f"kokoro/Kokoro-82M/{voice}"
        self._voice = voice
        self._pipeline = KPipeline(lang_code="a", device=device)

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        _check_text(text, language, ("en",))

        def run() -> np.ndarray:
            chunks = [
                audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
                for _, _, audio in self._pipeline(text, voice=self._voice)
            ]
            return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

        return wav_bytes(_guarded(run), 24_000)


class LanguageTextToSpeech:
    """Routes each language to its own model, since the best model may differ by language."""

    def __init__(self, by_language: dict[Language, TextToSpeech]) -> None:
        self._by_language = by_language
        self.model_name = ", ".join(f"{lang}: {model.model_name}" for lang, model in by_language.items())

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        model = self._by_language.get(language)
        if model is None:
            raise ModelError(f"no speech synthesis model for {language}")
        return model.synthesize(text, language)
