"""Paths and GPU measurement shared by the evaluation scripts."""

from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
MODELS = DATA / "models"
REPORTS = Path(__file__).resolve().parent / "reports"
FLEURS_CONFIG = {"ko": "ko_kr", "en": "en_us"}


def gpu_memory_mb() -> tuple[str, float]:
    """GPU name and memory in use on the whole device (all processes), in MB."""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
    return (name.decode() if isinstance(name, bytes) else name), used
