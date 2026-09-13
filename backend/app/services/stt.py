"""Speech recognition with Whisper through faster-whisper (CTranslate2)."""

import io

from av.error import FFmpegError
from faster_whisper import WhisperModel

from app.services.cuda import add_cuda_dll_dirs
from app.services.interfaces import Language, NoSpeechError, Transcript, UndecodableAudioError


class WhisperSpeechToText:
    """Implements app.services.interfaces.SpeechToText.

    vad_filter and no_speech_threshold go to faster-whisper (its defaults: False, 0.6). With
    retry_without_no_speech, a result that comes back empty is recognized once more with the no-speech
    check off (the candidates in docs/experiments.md 1-1, T14).
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
    ) -> None:
        add_cuda_dll_dirs()
        self.model_name = f"faster-whisper/{model_size}"
        self.options = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "no_speech_threshold": no_speech_threshold,
        }
        self.retry_without_no_speech = retry_without_no_speech
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        if not audio:
            raise UndecodableAudioError("empty audio")
        text, duration = self._recognize(audio, language, self.options)
        if not text and self.retry_without_no_speech:
            text, duration = self._recognize(audio, language, {**self.options, "no_speech_threshold": None})
        if not text:
            raise NoSpeechError("no speech was recognized")
        return Transcript(text=text, language=language, duration_ms=round(duration * 1000))

    def _recognize(self, audio: bytes, language: Language, options: dict) -> tuple[str, float]:
        try:
            segments, info = self._model.transcribe(io.BytesIO(audio), language=language, **options)
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except FFmpegError as exc:
            raise UndecodableAudioError("the audio could not be decoded") from exc
        return text, info.duration
