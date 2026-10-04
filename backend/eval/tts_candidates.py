"""Speech synthesis candidates for eval.tts_eval (tasks T4, T78 and T82). Each loader imports only its own
libraries, because the candidates run in different environments (docs/experiments.md)."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from app.services.interfaces import Language, TextToSpeech
from app.services.tts import KokoroTextToSpeech, MeloTextToSpeech, MmsTextToSpeech, SupertonicTextToSpeech
from eval.common import MODELS

# T78 (docs/experiments.md 12), fixed before any FLEURS run: ONNX Runtime on the CPU with 2 intra-op threads
# (the fastest of 2, 4, 6 and 12 in a smoke on made-up sentences in the API container, and it leaves 4 of the
# 6 cores to the server's other work), the reference code's speed 1.05 and voice F1.
SUPERTONIC_DIR = MODELS / "supertonic-3"
SUPERTONIC_THREADS = 2


def _kokoro(language: Language) -> TextToSpeech:
    import espeakng_loader

    # espeak-ng cannot open a path with Korean characters: work from the data's parent folder and pass a
    # relative ASCII path. Changing the working directory is fine in this one-off evaluation process;
    # the report and audio paths are absolute.
    data = Path(espeakng_loader.get_data_path())
    os.chdir(data.parent)
    os.environ["ESPEAK_DATA_PATH"] = data.name
    return KokoroTextToSpeech()


def _supertonic(steps: int, provider: str = "cpu") -> Callable[[Language], TextToSpeech]:
    def load(language: Language) -> TextToSpeech:
        return SupertonicTextToSpeech(
            SUPERTONIC_DIR, steps=steps, threads=SUPERTONIC_THREADS, languages=(language,), provider=provider
        )

    return load


# name: (how to load the model for one language, languages it speaks)
CANDIDATES: dict[str, tuple[Callable[[Language], TextToSpeech], list[Language]]] = {
    "mms": (MmsTextToSpeech, ["ko", "en"]),
    "melo": (MeloTextToSpeech, ["ko", "en"]),
    "kokoro": (_kokoro, ["en"]),
    # Supertonic 3 speaks English too, but the trial replaces only the Korean model (Kokoro keeps English).
    # S: the reference code's default 8 steps. S-fast: 2 steps, the fastest setting its documentation lists.
    "supertonic": (_supertonic(8), ["ko"]),
    "supertonic-fast": (_supertonic(2), ["ko"]),
    # T82 (docs/experiments.md 14): the same two on ONNX Runtime's CUDA provider, with the provider options
    # fixed in app/services/supertonic.py. Needs onnxruntime-gpu (the trial image, not the lock file).
    "supertonic-cuda": (_supertonic(8, "cuda"), ["ko"]),
    "supertonic-fast-cuda": (_supertonic(2, "cuda"), ["ko"]),
}
