"""Paths and GPU measurement shared by the evaluation scripts."""

# [파일 역할] 측정 스크립트(backend/eval/)가 함께 쓰는 경로 상수와 GPU 메모리 측정 함수.
# [왜 필요한가] 데이터·모델·리포트 위치를 한곳에서 정해 스크립트마다 경로가 어긋나지 않게 한다. 측정 결과는
# eval/reports/에, 내려받은 데이터·모델은 커밋하지 않는 data/에 둔다.
# [연결] fleurs_download, stt_eval, mt_eval, tts_eval, e2e_eval, eos_eval, langid_eval, stream_eval,
# mt_convert, mt_decide, hymt_setup 등 대부분의 eval 스크립트가 가져다 쓴다.

from __future__ import annotations

from pathlib import Path

# PROJECT: 저장소 루트(eval/common.py → eval → backend → 루트). DATA·MODELS: 내려받은 평가 데이터와 변환한
# 모델(data/는 커밋하지 않음). REPORTS: 측정 리포트를 남기는 backend/eval/reports. FLEURS_CONFIG: 언어 코드 →
# FLEURS 데이터셋 설정 이름.
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
MODELS = DATA / "models"
REPORTS = Path(__file__).resolve().parent / "reports"
FLEURS_CONFIG = {"ko": "ko_kr", "en": "en_us"}


# GPU 이름과 장치 전체(모든 프로세스)의 사용 중 메모리(MB). 측정에서 모델을 올리기 전후의 차이로 VRAM 사용량을
# 기록하는 데 쓴다. pynvml(NVIDIA 관리 라이브러리)은 GPU 측정 때만 필요해 함수 안에서 import한다. 오래된
# pynvml은 이름을 bytes로 돌려주므로 문자열로 바꾼다.
def gpu_memory_mb() -> tuple[str, float]:
    """GPU name and memory in use on the whole device (all processes), in MB."""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
    return (name.decode() if isinstance(name, bytes) else name), used
