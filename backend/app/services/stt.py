"""Speech recognition with Whisper through faster-whisper (CTranslate2)."""

import importlib.util
import io
import os
import sys

from av.error import FFmpegError
from faster_whisper import WhisperModel

from app.services.interfaces import InvalidAudioError, Language, Transcript

_CUDA_WHEELS = ("nvidia.cublas", "nvidia.cudnn")


def _add_cuda_dll_dirs() -> None:
    """Let CTranslate2 find the CUDA libraries installed from pip on Windows.

    CTranslate2 loads cuBLAS and cuDNN by name, and Windows doesn't search the folders
    that the nvidia-* wheels install into, so it fails with "cublas64_12.dll is not
    found". Linux resolves these through the wheels' rpath entries and needs nothing.
    """
    if sys.platform != "win32":
        return
    for package in _CUDA_WHEELS:
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue  # the gpu dependency group isn't installed; CPU use still works
        for root in spec.submodule_search_locations:
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


class WhisperSpeechToText:
    """Implements app.services.interfaces.SpeechToText."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
    ) -> None:
        _add_cuda_dll_dirs()
        self.model_name = f"faster-whisper/{model_size}"
        self.beam_size = beam_size
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: bytes, language: Language) -> Transcript:
        if not audio:
            raise InvalidAudioError("empty audio")
        try:
            segments, info = self._model.transcribe(
                io.BytesIO(audio), language=language, beam_size=self.beam_size
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except FFmpegError as exc:
            raise InvalidAudioError("the audio could not be decoded") from exc
        if not text:
            raise InvalidAudioError("no speech was recognized")
        return Transcript(text=text, language=language, duration_ms=round(info.duration * 1000))
