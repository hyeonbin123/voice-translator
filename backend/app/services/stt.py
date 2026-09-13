"""Speech recognition with Whisper through faster-whisper (CTranslate2)."""

import io

import av
import numpy as np
from av.error import FFmpegError
from faster_whisper import WhisperModel
from faster_whisper.audio import _group_frames, _ignore_invalid_frames, _resample_frames

from app.services.cuda import add_cuda_dll_dirs
from app.services.interfaces import Language, NoSpeechError, Transcript, UndecodableAudioError

SAMPLE_RATE = 16_000


def decode_audio(audio: bytes) -> np.ndarray:
    """faster-whisper 1.2's decode_audio without its gc.collect() (docs/experiments.md 5, candidate C).

    faster-whisper runs a full garbage collection after every decode to free resampler objects (its issue
    390). In a process holding the models that collection holds the GIL for about 0.4 s, and the event loop
    stalls. Same steps otherwise: 16 kHz mono s16 through PyAV, then float32.
    """
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    raw = io.BytesIO()
    dtype = None
    with av.open(io.BytesIO(audio), mode="r", metadata_errors="ignore") as container:
        frames = _group_frames(_ignore_invalid_frames(container.decode(audio=0)), 500000)
        for frame in _resample_frames(frames, resampler):
            array = frame.to_ndarray()
            dtype = array.dtype
            raw.write(array)
    del resampler
    return np.frombuffer(raw.getbuffer(), dtype=dtype).astype(np.float32) / 32768.0


class WhisperSpeechToText:
    """Implements app.services.interfaces.SpeechToText.

    vad_filter and no_speech_threshold go to faster-whisper (its defaults: False, 0.6). With
    retry_without_no_speech, a result that comes back empty is recognized once more with the no-speech
    check off (the candidates in docs/experiments.md 1-1, T14). With own_decode, the upload is decoded by
    decode_audio above instead of inside faster-whisper (docs/experiments.md 5, candidate C).
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        vad_filter: bool = False,
        no_speech_threshold: float | None = 0.6,
        retry_without_no_speech: bool = False,
        own_decode: bool = False,
    ) -> None:
        add_cuda_dll_dirs()
        self.model_name = f"faster-whisper/{model_size}"
        self.options = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "no_speech_threshold": no_speech_threshold,
        }
        self.retry_without_no_speech = retry_without_no_speech
        self.own_decode = own_decode
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        if not audio:
            raise UndecodableAudioError("empty audio")
        source = self._decode(audio) if self.own_decode else audio
        text, duration = self._recognize(source, language, self.options)
        if not text and self.retry_without_no_speech:
            text, duration = self._recognize(source, language, {**self.options, "no_speech_threshold": None})
        if not text:
            raise NoSpeechError("no speech was recognized")
        return Transcript(text=text, language=language, duration_ms=round(duration * 1000))

    def _decode(self, audio: bytes) -> np.ndarray:
        try:
            samples = decode_audio(audio)
        except FFmpegError as exc:
            raise UndecodableAudioError("the audio could not be decoded") from exc
        if not samples.size:
            raise NoSpeechError("the audio has no samples")
        return samples

    def _recognize(self, source: bytes | np.ndarray, language: Language, options: dict) -> tuple[str, float]:
        model_input = io.BytesIO(source) if isinstance(source, bytes) else source
        try:
            segments, info = self._model.transcribe(model_input, language=language, **options)
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except FFmpegError as exc:
            raise UndecodableAudioError("the audio could not be decoded") from exc
        return text, info.duration
