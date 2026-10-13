"""CUDA library lookup for the CTranslate2-based services (speech recognition and translation)."""

# [파일 역할] Windows에서 CTranslate2(faster-whisper 음성 인식, opus-mt 번역)가 pip로 설치된 CUDA
# 라이브러리(cuBLAS, cuDNN)를 찾게 해 주는 도우미 하나.
# [설계 이유] 아래 docstring대로, Windows는 nvidia-* wheel이 설치한 폴더를 DLL 검색 경로에 넣지 않아
# "cublas64_12.dll is not found"로 실패한다. CUDA Toolkit을 따로 설치하지 않고 uv의 gpu 의존성 그룹만으로 돌게
# 하려는 것이다. Linux(Docker 컨테이너)는 아무것도 하지 않는다.
# [연결] services/stt.py, services/translation.py가 모델을 올리기 전에 부르고,
# eval/eos_eval.py·langid_eval.py·stream_eval.py도 부른다. 테스트: tests/test_cuda.py.

import functools
import os
import sys
from importlib.util import find_spec

# CUDA DLL을 담은 pip 패키지(nvidia-cublas-*, nvidia-cudnn-*)의 import 이름.
_CUDA_WHEELS = ("nvidia.cublas", "nvidia.cudnn")


# functools.cache로 프로세스당 한 번만 실행한다(여러 모델이 불러도 PATH에 같은 폴더를 거듭 붙이지 않음).
# 테스트는 cache_clear()로 다시 실행한다.
@functools.cache
def add_cuda_dll_dirs() -> None:
    """Let CTranslate2 find the CUDA libraries installed from pip on Windows.

    CTranslate2 loads cuBLAS and cuDNN by name, and Windows doesn't search the folders
    that the nvidia-* wheels install into, so it fails with "cublas64_12.dll is not
    found". Linux resolves these through the wheels' rpath entries and needs nothing.
    Cached because every model service calls it and each call would prepend to PATH again.
    """
    if sys.platform != "win32":
        return
    # find_spec으로 패키지를 import하지 않고 설치 위치만 찾는다. 네임스페이스 패키지라 위치가 여러 개일 수
    # 있어 submodule_search_locations를 모두 돈다.
    for package in _CUDA_WHEELS:
        try:
            spec = find_spec(package)
        except ModuleNotFoundError:
            spec = None  # not even the parent "nvidia" package is installed
        if spec is None or not spec.submodule_search_locations:
            continue  # the gpu dependency group isn't installed; CPU use still works
        for root in spec.submodule_search_locations:
            # 두 가지 방법으로 등록한다: os.add_dll_directory(Python 3.8+의 Windows DLL 검색 경로)와 PATH 앞에
            # 붙이기. 라이브러리가 DLL을 어떤 방식으로 불러도 찾게 하려는 것이다.
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
