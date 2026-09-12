"""CUDA library lookup for the CTranslate2-based services (speech recognition and translation)."""

import functools
import os
import sys
from importlib.util import find_spec

_CUDA_WHEELS = ("nvidia.cublas", "nvidia.cudnn")


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
    for package in _CUDA_WHEELS:
        try:
            spec = find_spec(package)
        except ModuleNotFoundError:
            spec = None  # not even the parent "nvidia" package is installed
        if spec is None or not spec.submodule_search_locations:
            continue  # the gpu dependency group isn't installed; CPU use still works
        for root in spec.submodule_search_locations:
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
