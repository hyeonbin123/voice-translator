"""Speech recognition with Whisper through faster-whisper (CTranslate2)."""

import io

from av.error import FFmpegError
from faster_whisper import WhisperModel

from app.services.cuda import add_cuda_dll_dirs
from app.services.interfaces import Language, NoSpeechError, Transcript, UndecodableAudioError


class WhisperSpeechToText:
    """Implements app.services.interfaces.SpeechToText."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
    ) -> None:
        add_cuda_dll_dirs()
        self.model_name = f"faster-whisper/{model_size}"
        self.beam_size = beam_size
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        if not audio:
            raise UndecodableAudioError("empty audio")
        try:
            segments, info = self._model.transcribe(
                io.BytesIO(audio), language=language, beam_size=self.beam_size
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except FFmpegError as exc:
            raise UndecodableAudioError("the audio could not be decoded") from exc
        if not text:
            raise NoSpeechError("no speech was recognized")
        return Transcript(text=text, language=language, duration_ms=round(info.duration * 1000))
