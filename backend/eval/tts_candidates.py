"""Speech synthesis candidates for eval.tts_eval (tasks T4, T78 and T82). Each loader imports only its own
libraries, because the candidates run in different environments (docs/experiments.md)."""

# [파일 역할] 음성 합성 측정 도구(eval/tts_eval.py)가 고르는 후보 표다. 후보 이름 → (언어 하나에 대한 합성기를
# 만드는 함수, 지원 언어 목록).
# [왜 따로 두나] 후보들은 의존성이 충돌해 서로 다른 환경에서 돈다(예: MeloTTS는 transformers 4.27.4 고정,
# docs/experiments.md 3절). 그래서 표를 별도 모듈로 두고, 합성기 클래스는 실제로 만들 때 그 후보의
# 라이브러리만 불러오게 했다(app/services/tts.py의 클래스들).
# [후보] T4: mms, melo, kokoro(3절, 한국어 MeloTTS·영어 Kokoro를 골랐다). T78:
# supertonic(8단계)·supertonic-fast(2단계)를 CPU로(12절). T82: 같은 두 설정을 ONNX Runtime CUDA
# 제공자로(14절). 두 시험 모두 MeloTTS 유지로 끝났다.
# [연결] eval/tts_eval.py의 synth가 CANDIDATES를 읽고, tests/test_tts_eval.py가 이 표를 가짜 후보로 바꿔 끼워
# 검증한다.

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

# 서버가 쓰는 합성기 구현을 그대로 가져와 측정한 코드와 서비스 코드가 같게 한다.
from app.services.interfaces import Language, TextToSpeech
from app.services.tts import KokoroTextToSpeech, MeloTextToSpeech, MmsTextToSpeech, SupertonicTextToSpeech
from eval.common import MODELS

# T78 (docs/experiments.md 12), fixed before any FLEURS run: ONNX Runtime on the CPU with 2 intra-op threads
# (the fastest of 2, 4, 6 and 12 in a smoke on made-up sentences in the API container, and it leaves 4 of the
# 6 cores to the server's other work), the reference code's speed 1.05 and voice F1.
# SUPERTONIC_DIR: eval/supertonic_download.py가 해시를 확인하며 받아 두는 모델 폴더. SUPERTONIC_THREADS: 위
# 영어 주석의 이유로 정한 intra-op 2스레드.
SUPERTONIC_DIR = MODELS / "supertonic-3"
SUPERTONIC_THREADS = 2


# Kokoro 영어 합성기를 만든다. 발음 보조로 쓰는 espeak-ng가 한글이 든 경로를 열지 못하므로, espeak 데이터
# 폴더의 부모로 작업 폴더를 옮기고 상대 경로를 넘긴다(3절 설치 확인). Linux 컨테이너에서는 필요 없는 Windows
# 평가용 우회다.
def _kokoro(language: Language) -> TextToSpeech:
    import espeakng_loader

    # espeak-ng cannot open a path with Korean characters: work from the data's parent folder and pass a
    # relative ASCII path. Changing the working directory is fine in this one-off evaluation process;
    # the report and audio paths are absolute.
    data = Path(espeakng_loader.get_data_path())
    os.chdir(data.parent)
    os.environ["ESPEAK_DATA_PATH"] = data.name
    return KokoroTextToSpeech()


# 단계 수와 실행 제공자(cpu·cuda)를 고정한 Supertonic 합성기 생성 함수를 돌려준다. CANDIDATES의 값은 '언어
# 하나를 받는 함수'여야 하므로 클로저로 감싼다. languages=(language,)로 요청 언어 하나만 맡게 한다.
def _supertonic(steps: int, provider: str = "cpu") -> Callable[[Language], TextToSpeech]:
    def load(language: Language) -> TextToSpeech:
        return SupertonicTextToSpeech(
            SUPERTONIC_DIR, steps=steps, threads=SUPERTONIC_THREADS, languages=(language,), provider=provider
        )

    return load


# 후보 이름은 tts_eval synth --candidate의 선택지다. 목소리 등 합성기 설정은 각 클래스 기본값을
# 따른다(MeloTTS·Kokoro는 서버도 같은 기본값으로 만든다, app/services/models.py).
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
