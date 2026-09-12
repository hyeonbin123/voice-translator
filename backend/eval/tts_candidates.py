"""Speech synthesis candidates for eval.tts_eval (task T4). Each loader imports only its own libraries,
because the candidates run in different environments (docs/experiments.md)."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from app.services.interfaces import Language, TextToSpeech
from app.services.tts import KokoroTextToSpeech, MeloTextToSpeech, MmsTextToSpeech


def _kokoro(language: Language) -> TextToSpeech:
    import espeakng_loader

    # espeak-ng cannot open a path with Korean characters: work from the data's parent folder and pass a
    # relative ASCII path. Changing the working directory is fine in this one-off evaluation process;
    # the report and audio paths are absolute.
    data = Path(espeakng_loader.get_data_path())
    os.chdir(data.parent)
    os.environ["ESPEAK_DATA_PATH"] = data.name
    return KokoroTextToSpeech()


# name: (how to load the model for one language, languages it speaks)
CANDIDATES: dict[str, tuple[Callable[[Language], TextToSpeech], list[Language]]] = {
    "mms": (MmsTextToSpeech, ["ko", "en"]),
    "melo": (MeloTextToSpeech, ["ko", "en"]),
    "kokoro": (_kokoro, ["en"]),
}
