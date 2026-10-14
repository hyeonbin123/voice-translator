"""eval.tts_eval이 쓰는 음성 합성 후보(작업 T4, T78, T82). 후보마다 도는 환경이 달라서 각 로더는 자기
라이브러리만 불러온다(docs/experiments.md)."""

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

# SUPERTONIC_DIR: eval/supertonic_download.py가 해시를 확인하며 받아 두는 모델 폴더.
# SUPERTONIC_THREADS: T78(docs/experiments.md 12절)에서 FLEURS를 돌리기 전에 고정한 설정은 CPU의 ONNX Runtime
# intra-op 2스레드, 참조 코드의 속도 1.05, 목소리 F1이다. 2스레드는 API 컨테이너에서 지어낸 문장으로 짧게 돌려
# 본 2·4·6·12 가운데 가장 빨랐고, 코어 6개 중 4개를 서버의 다른 일에 남긴다.
SUPERTONIC_DIR = MODELS / "supertonic-3"
SUPERTONIC_THREADS = 2


# Kokoro 영어 합성기를 만든다. 발음 보조로 쓰는 espeak-ng가 한글이 든 경로를 열지 못하므로, espeak 데이터
# 폴더의 부모로 작업 폴더를 옮기고 상대 경로를 넘긴다(3절 설치 확인). Linux 컨테이너에서는 필요 없는 Windows
# 평가용 우회다.
def _kokoro(language: Language) -> TextToSpeech:
    import espeakng_loader

    # espeak-ng는 한글이 든 경로를 열지 못하므로 데이터 폴더의 부모에서 작업하고 ASCII 상대 경로를 넘긴다.
    # 작업 폴더를 바꿔도 되는 것은 이 프로세스가 한 번 쓰고 끝나는 평가용이고, 보고서·음성 경로가 모두 절대
    # 경로이기 때문이다.
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
# 형식: 이름: (언어 하나에 대한 모델을 만드는 함수, 그 후보가 말하는 언어 목록)
CANDIDATES: dict[str, tuple[Callable[[Language], TextToSpeech], list[Language]]] = {
    "mms": (MmsTextToSpeech, ["ko", "en"]),
    "melo": (MeloTextToSpeech, ["ko", "en"]),
    "kokoro": (_kokoro, ["en"]),
    # Supertonic 3은 영어도 말하지만, 이 시험은 한국어 모델만 바꾼다(영어는 Kokoro 유지).
    # S: 참조 코드의 기본 8단계. S-fast: 2단계, 그 문서에 적힌 가장 빠른 설정.
    "supertonic": (_supertonic(8), ["ko"]),
    "supertonic-fast": (_supertonic(2), ["ko"]),
    # T82(docs/experiments.md 14절): 같은 두 설정을 ONNX Runtime의 CUDA 제공자로, 제공자 옵션은
    # app/services/supertonic.py에 고정한 값으로 돌린다. onnxruntime-gpu가 필요하다(lock 파일이 아니라 시험용
    # 이미지에 들어 있음).
    "supertonic-cuda": (_supertonic(8, "cuda"), ["ko"]),
    "supertonic-fast-cuda": (_supertonic(2, "cuda"), ["ko"]),
}
