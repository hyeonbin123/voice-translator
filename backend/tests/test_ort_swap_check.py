"""The check that swapping onnxruntime for onnxruntime-gpu leaves speech recognition as it was (T82,
docs/experiments.md 14). No model runs: the timestamps and texts are made up."""

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from eval import ort_swap_check

BACKEND = Path(__file__).resolve().parents[1]


def record(kind: str, clips: dict, no_speech: dict, package: str = "onnxruntime") -> dict:
    return {"kind": kind, "runtime": {"package": package}, "clips": clips, "no_speech": no_speech}


SILENT = {"ko_silence": "", "ko_noise": "", "en_silence": "", "en_noise": ""}


def test_the_tool_imports_without_the_eval_group():
    # The API container, where it runs, has neither pyarrow nor jiwer.
    code = (
        "import sys; sys.modules['pyarrow'] = None; sys.modules['pyarrow.parquet'] = None; "
        "sys.modules['jiwer'] = None; import eval.ort_swap_check"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_the_same_texts_and_no_speech_found_in_silence_pass():
    before = record("transcribe", {"en_1.wav": "Hello.", "ko_2.wav": "안녕하세요."}, SILENT)
    after = record("transcribe", {"en_1.wav": "Hello.", "ko_2.wav": "안녕하세요."}, SILENT, "onnxruntime-gpu")
    result = ort_swap_check.compare(before, after)
    assert result["pass"] is True
    assert (result["same"], result["clips"], result["differs"]) == (2, 2, [])
    assert result["no_speech_empty"] == {"before": 4, "after": 4, "of": 4}
    assert result["packages"] == ["onnxruntime", "onnxruntime-gpu"]


def test_one_changed_text_fails_and_is_named():
    before = record("transcribe", {"en_1.wav": "Hello.", "ko_2.wav": "안녕하세요."}, SILENT)
    after = record("transcribe", {"en_1.wav": "Hello!", "ko_2.wav": "안녕하세요."}, SILENT)
    result = ort_swap_check.compare(before, after)
    assert result["pass"] is False and result["differs"] == ["en_1.wav"] and result["same"] == 1


def test_speech_made_up_from_silence_after_the_swap_fails():
    # T14's silence check: all four no-speech inputs must come back empty (4/4).
    after = record("transcribe", {"en_1.wav": "Hello."}, {**SILENT, "en_noise": "Thank you."})
    result = ort_swap_check.compare(record("transcribe", {"en_1.wav": "Hello."}, SILENT), after)
    assert result["pass"] is False and result["no_speech_empty"]["after"] == 3


def test_vad_timestamps_must_match_exactly_including_the_no_speech_inputs():
    clips = {"en_1.wav": [[1600, 32000]], "ko_2.wav": [[0, 16000], [20000, 48000]]}
    quiet = {"silence": [], "noise": []}
    assert ort_swap_check.compare(record("vad", clips, quiet), record("vad", clips, quiet))["pass"] is True
    moved = {**clips, "ko_2.wav": [[0, 16000], [20512, 48000]]}
    result = ort_swap_check.compare(record("vad", clips, quiet), record("vad", moved, quiet))
    assert result["pass"] is False and result["differs"] == ["ko_2.wav"]
    noisy = {"silence": [], "noise": [[0, 512]]}
    assert ort_swap_check.compare(record("vad", clips, quiet), record("vad", clips, noisy))["pass"] is False


def test_records_of_different_kinds_or_clips_are_not_compared():
    with pytest.raises(ValueError, match="kind"):
        ort_swap_check.compare(record("vad", {}, {}), record("transcribe", {}, {}))
    with pytest.raises(ValueError, match="clips"):
        ort_swap_check.compare(
            record("vad", {"en_1.wav": []}, {}), record("vad", {"en_1.wav": [], "en_2.wav": []}, {})
        )


def test_clips_take_the_language_from_the_file_name(tmp_path):
    (tmp_path / "ko_12.wav").write_bytes(b"k")
    (tmp_path / "en_3.wav").write_bytes(b"e")
    (tmp_path / "notes.txt").write_text("x")
    assert ort_swap_check.clips(tmp_path) == [("en_3.wav", b"e", "en"), ("ko_12.wav", b"k", "ko")]


def test_the_no_speech_inputs_are_t14s():
    stt_eval = pytest.importorskip("eval.stt_eval", reason="needs the eval group (jiwer, pyarrow)")
    for kind in ("silence", "noise"):
        assert ort_swap_check.no_speech_wav(kind) == stt_eval.no_speech_wav(kind)


def test_vad_writes_one_record_with_the_runtime(tmp_path, monkeypatch):
    (tmp_path / "en_1.wav").write_bytes(b"one")
    (tmp_path / "ko_2.wav").write_bytes(b"two")
    seen = []

    def timestamps(audio: bytes) -> list[list[int]]:
        seen.append(audio)
        return [[0, len(audio)]]

    monkeypatch.setattr(ort_swap_check, "speech_timestamps", timestamps)
    monkeypatch.setattr(
        ort_swap_check, "runtime_info", lambda: {"package": "onnxruntime", "version": "1.30.0"}
    )
    out = tmp_path / "vad.json"
    ort_swap_check.vad(Namespace(clips=tmp_path, out=out))
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["kind"] == "vad" and written["runtime"]["package"] == "onnxruntime"
    assert written["clips"] == {"en_1.wav": [[0, 3]], "ko_2.wav": [[0, 3]]}
    assert set(written["no_speech"]) == {"silence", "noise"}
    assert seen[:2] == [b"one", b"two"]
