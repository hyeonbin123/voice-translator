# [파일 역할] app/services/cuda.py의 add_cuda_dll_dirs가 GPU 패키지(nvidia-cublas·nvidia-cudnn wheel)가 없는
# Windows에서도 서버 시작을 막지 않는지 확인하는 테스트 묶음이다.
# [왜 필요한가] 상위 패키지 `nvidia`조차 설치돼 있지 않으면 find_spec("nvidia.cublas")가 None이 아니라
# ModuleNotFoundError를 던진다. 예전에는 이 때문에 CPU만 쓰는 환경에서 음성 인식 생성자가 실패했다(커밋
# d2c532d). 이 테스트가 그 회귀를 막는다.
# [방법] 실제 GPU나 Windows가 없어도 돌도록 monkeypatch로 sys.platform·find_spec·os.add_dll_directory를 바꿔
# '패키지 없는 Windows'를 흉내 낸다.

import pytest

# cuda: 시험 대상 모듈. stt: add_cuda_dll_dirs를 생성자에서 부르는 실제 사용처(WhisperSpeechToText).
from app.services import cuda, stt


# 'GPU 패키지가 전혀 없는 Windows' 환경을 만드는 fixture. yield 값은 add_dll_directory에 넘어간 폴더 목록이다.
# 주의: add_cuda_dll_dirs는 functools.cache로 프로세스당 한 번만 실행되므로, 앞뒤로 cache_clear()를 불러 이
# 테스트에서 실제로 다시 실행되게 하고 다른 테스트에 결과가 남지 않게 한다.
@pytest.fixture
def windows_without_gpu_packages(monkeypatch):
    """Windows where not even the parent `nvidia` package is installed, so find_spec raises."""

    def missing(name):
        raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")

    # find_spec은 무조건 ModuleNotFoundError를 던지게 바꾸고, add_dll_directory는 호출 인자를 목록에 모으게
    # 바꾼다. Linux CI에는 os.add_dll_directory가 없으므로 raising=False로 없는 속성도 만들게 한다.
    added = []
    monkeypatch.setattr(cuda.sys, "platform", "win32")
    monkeypatch.setattr(cuda, "find_spec", missing)
    monkeypatch.setattr(cuda.os, "add_dll_directory", added.append, raising=False)
    cuda.add_cuda_dll_dirs.cache_clear()
    yield added
    cuda.add_cuda_dll_dirs.cache_clear()


# find_spec이 예외를 던져도 add_cuda_dll_dirs가 예외 없이 끝나고, DLL 폴더를 하나도 등록하지 않는지
# 확인한다(gpu 의존성 그룹이 없으면 조용히 넘어가야 CPU 실행이 가능하다).
def test_missing_gpu_packages_are_skipped(windows_without_gpu_packages):
    cuda.add_cuda_dll_dirs()
    assert windows_without_gpu_packages == []


# 실제 사용처 수준의 확인: 같은 '패키지 없는 Windows'에서 WhisperSpeechToText를 CPU·int8로 만들 때 생성자가
# 실패하지 않고, 받은 설정을 그대로 WhisperModel에 넘기는지 본다.
# WhisperModel은 가짜 클래스로 바꿔 모델 가중치를 내려받지 않게 하고, 전달된 인자만 기록한다.
@pytest.mark.usefixtures("windows_without_gpu_packages")
def test_speech_to_text_starts_on_cpu_without_gpu_packages(monkeypatch):
    created = {}

    class FakeWhisperModel:
        def __init__(self, model_size, device, compute_type):
            created.update(model_size=model_size, device=device, compute_type=compute_type)

    monkeypatch.setattr(stt, "WhisperModel", FakeWhisperModel)
    model = stt.WhisperSpeechToText("tiny", device="cpu", compute_type="int8")
    assert created == {"model_size": "tiny", "device": "cpu", "compute_type": "int8"}
    assert model.model_name == "faster-whisper/tiny"
