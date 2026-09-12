import pytest

from app.services import cuda, stt


@pytest.fixture
def windows_without_gpu_packages(monkeypatch):
    """Windows where not even the parent `nvidia` package is installed, so find_spec raises."""

    def missing(name):
        raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")

    added = []
    monkeypatch.setattr(cuda.sys, "platform", "win32")
    monkeypatch.setattr(cuda, "find_spec", missing)
    monkeypatch.setattr(cuda.os, "add_dll_directory", added.append, raising=False)
    cuda.add_cuda_dll_dirs.cache_clear()
    yield added
    cuda.add_cuda_dll_dirs.cache_clear()


def test_missing_gpu_packages_are_skipped(windows_without_gpu_packages):
    cuda.add_cuda_dll_dirs()
    assert windows_without_gpu_packages == []


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
