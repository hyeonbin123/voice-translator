"""The speech synthesis trial tools (T78, docs/experiments.md 12): sentence export for the container, the
recognition load, the digit-sentence sub-metric, the timing summary and the blind A/B pages. No model runs."""

import io
import json
import subprocess
import sys
import threading
import time
import wave
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from eval import tts_ab, tts_eval

BACKEND = Path(__file__).resolve().parents[1]


def wav(samples: np.ndarray, rate: int = 1000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as clip:
        data = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2").astype(np.float32) / 32767
        return data, clip.getframerate()


def test_the_tools_import_without_the_eval_group():
    # CI and the API container have neither pyarrow nor jiwer: they are needed to read FLEURS and to score.
    code = (
        "import sys; sys.modules['pyarrow'] = None; sys.modules['pyarrow.parquet'] = None; "
        "sys.modules['jiwer'] = None; import eval.tts_eval, eval.tts_ab, eval.tts_candidates"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_digit_sentences_are_those_with_arabic_digits():
    assert tts_eval.has_digits("13위에서 경기를 마쳤다.")
    assert not tts_eval.has_digits("열세 번째로 경기를 마쳤다.")


def test_exported_sentences_are_read_back_without_pyarrow(tmp_path, monkeypatch):
    rows = [
        {"id": 2, "raw_transcription": "둘.", "transcription": "둘"},
        {"id": 1, "raw_transcription": "하나.", "transcription": "하나"},
        {"id": 2, "raw_transcription": "둘.", "transcription": "둘"},
    ]
    monkeypatch.setattr(tts_eval, "_read_parquet", lambda language, split: rows)
    tts_eval.export(Namespace(split="validation", languages=["ko"], out=tmp_path))
    written = json.loads((tmp_path / "validation_ko.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in written] == [1, 2]

    monkeypatch.setattr(tts_eval, "_read_parquet", lambda language, split: pytest.fail("read the parquet"))
    assert tts_eval.sentences("ko", "validation", None, sentences_dir=tmp_path) == written
    assert [row["id"] for row in tts_eval.sentences("ko", "validation", 1, sentences_dir=tmp_path)] == [1]


def edits(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def corpus_cer(language: str, references: list[str], hypotheses: list[str]) -> float:
    """Total character edits over total reference length, as jiwer.cer (the eval group, not in CI)."""
    return sum(map(edits, references, hypotheses)) / sum(map(len, references))


def test_score_counts_failures_as_fully_wrong_and_reports_digit_sentences():
    rows = {
        1: {"id": 1, "raw_transcription": "가나다.", "transcription": "가나다"},
        2: {"id": 2, "raw_transcription": "3월에.", "transcription": "삼월에"},
        3: {"id": 3, "raw_transcription": "라마.", "transcription": "라마"},
    }
    values = {
        "model_name": "fake",
        "vram_mb": None,
        "failures": 1,
        "items": [
            {"id": 1, "synth_s": 0.2, "audio_s": 2.0, "per_audio_second": 0.1},
            {"id": 2, "synth_s": 0.4, "audio_s": 2.0, "per_audio_second": 0.2},
            {"id": 3, "failure": "ValueError: nothing"},
        ],
    }
    heard = {1: "가나다", 2: "삼월애"}
    result = tts_eval.score_language(
        "ko", values, rows, lambda item: heard[item["id"]], error_rate=corpus_cer
    )
    # 1 of 3 characters wrong in the digit sentence; the failed sentence's 2 characters are all missing.
    assert result["error"] == pytest.approx(3 / 8)
    assert (result["error_digits"], result["count_digits"]) == (pytest.approx(1 / 3), 1)
    assert result["failures"] == 1 and result["count"] == 3
    assert result["synth_s_p50"] == pytest.approx(0.3)


def test_score_reads_only_the_synthesis_records_of_its_own_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_eval, "REPORTS", tmp_path)
    for tag, candidate in (("t_dev", "melo"), ("t_dev", "supertonic"), ("t_dev_load", "supertonic")):
        record = {"tag": tag, "candidate": candidate, "split": "validation", "languages": {}}
        (tmp_path / f"tts_{tag}_{candidate}_synth.json").write_text(json.dumps(record), encoding="utf-8")
    assert [record["candidate"] for record in tts_eval.synth_records("t_dev")] == ["melo", "supertonic"]
    assert [record["tag"] for record in tts_eval.synth_records("t_dev_load")] == ["t_dev_load"]


class CountingStt:
    def __init__(self, fail_first: bool = False, fail_all: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_first, self.fail_all = fail_first, fail_all

    def transcribe(self, audio: bytes, language: str):
        self.calls.append(language)
        time.sleep(0.002)
        if self.fail_all or (self.fail_first and len(self.calls) == 1):
            raise RuntimeError("no GPU")
        return Namespace(text="ok")


def test_the_recognition_load_runs_back_to_back_until_stopped():
    stt = CountingStt()
    load = tts_eval.SttLoad(stt, [(b"a", "en"), (b"b", "ko")])
    load.start(timeout=5)
    assert stt.calls  # warm: at least one recognition finished before synthesis is timed
    time.sleep(0.05)
    summary = load.stop()
    assert summary["calls"] >= 4 and summary["errors"] == 0 and summary["call_s_p50"] > 0
    assert stt.calls[:4] == ["en", "ko", "en", "ko"]
    count = len(stt.calls)
    time.sleep(0.02)
    assert len(stt.calls) == count  # stopped for good
    assert not any(thread.name == "stt-load" for thread in threading.enumerate())


def test_a_recognition_load_that_cannot_run_stops_the_measurement():
    load = tts_eval.SttLoad(CountingStt(fail_all=True), [(b"a", "en")])
    with pytest.raises(RuntimeError, match="recognition load"):
        load.start(timeout=5)
    # A later failure is counted, not fatal.
    load = tts_eval.SttLoad(CountingStt(fail_first=False), [(b"a", "en")])
    load.start(timeout=5)
    assert load.stop()["errors"] == 0


def test_load_clips_take_the_language_from_the_file_name(tmp_path):
    (tmp_path / "ko_12.wav").write_bytes(b"k")
    (tmp_path / "en_3.wav").write_bytes(b"e")
    (tmp_path / "notes.txt").write_text("x")
    assert tts_eval.load_clips(tmp_path) == [(b"e", "en"), (b"k", "ko")]


def test_device_memory_falls_back_to_nvidia_smi():
    def run(*args, **kwargs):
        return Namespace(stdout="1234\n")

    assert tts_eval.nvidia_smi_memory_mb(run) == 1234.0

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert tts_eval.nvidia_smi_memory_mb(missing) is None


def test_timing_summarizes_speed_and_load_without_recognition(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_eval, "REPORTS", tmp_path)
    record = {
        "tag": "t_load",
        "candidate": "supertonic",
        "split": "validation",
        "load": {"ko": {"calls": 12, "errors": 0, "call_s_p50": 0.4}},
        "languages": {
            "ko": {
                "model_name": "supertonic-3/F1/8-step",
                "vram_mb": None,
                "failures": 0,
                "items": [
                    {"id": i, "synth_s": 0.1 * i, "audio_s": 1.0 * i, "per_audio_second": 0.1}
                    for i in (1, 2, 3)
                ],
            }
        },
    }
    (tmp_path / "tts_t_load_supertonic_synth.json").write_text(json.dumps(record), encoding="utf-8")
    tts_eval.timing(Namespace(tag="t_load"))
    (report,) = tmp_path.glob("tts_t_load_timing_*.json")
    (summary,) = json.loads(report.read_text(encoding="utf-8"))
    ko = summary["languages"]["ko"]
    assert (ko["speed_p50"], ko["synth_s_p50"], ko["failures"]) == (pytest.approx(0.1), pytest.approx(0.2), 0)
    assert summary["load"]["ko"]["calls"] == 12
    assert "| supertonic | ko | 0.100 |" in report.with_suffix(".md").read_text(encoding="utf-8")


FAILED = {"failure": "ModelError: the model returned no audio"}


def speeds(values: dict) -> tuple:
    return values["speed_p50"], values["speed_p95"], values["synth_s_p50"]


def test_timing_reports_a_candidate_whose_synthesis_mostly_failed(tmp_path, monkeypatch):
    """The failures are what the report must show: no success leaves the speed empty, one success is it
    (T79)."""
    monkeypatch.setattr(tts_eval, "REPORTS", tmp_path)
    done = {"id": 2, "synth_s": 0.6, "audio_s": 3.0, "per_audio_second": 0.2}
    for candidate, items in (
        ("none", [{"id": 1} | FAILED, {"id": 2} | FAILED]),
        ("one", [{"id": 1} | FAILED, done]),
    ):
        failures = sum("failure" in item for item in items)
        language = {"model_name": candidate, "vram_mb": None, "failures": failures, "items": items}
        record = {
            "tag": "t_fail",
            "candidate": candidate,
            "split": "validation",
            "languages": {"ko": language},
        }
        (tmp_path / f"tts_t_fail_{candidate}_synth.json").write_text(json.dumps(record), encoding="utf-8")
    tts_eval.timing(Namespace(tag="t_fail"))
    (report,) = tmp_path.glob("tts_t_fail_timing_*.json")
    summary = {
        result["candidate"]: result["languages"]["ko"] for result in json.loads(report.read_text("utf-8"))
    }
    assert speeds(summary["none"]) == (None, None, None)
    assert (summary["none"]["failures"], summary["none"]["count"]) == (2, 2)
    assert speeds(summary["one"]) == (pytest.approx(0.2), pytest.approx(0.2), pytest.approx(0.6))
    assert (summary["one"]["failures"], summary["one"]["count"]) == (1, 2)
    table = report.with_suffix(".md").read_text(encoding="utf-8")
    assert "| none | ko | - | - | - | 2 |" in table
    assert "| one | ko | 0.200 | 0.200 | 0.600초 | 1 |" in table


def test_score_reports_a_candidate_with_no_successful_synthesis(tmp_path, monkeypatch):
    rows = {1: {"id": 1, "raw_transcription": "가나다.", "transcription": "가나다"}}
    values = {"model_name": "fake", "vram_mb": None, "failures": 1, "items": [{"id": 1} | FAILED]}
    result = tts_eval.score_language(
        "ko", values, rows, lambda item: pytest.fail("nothing to hear"), error_rate=corpus_cer
    )
    assert result["error"] == pytest.approx(1.0) and result["failures"] == 1
    assert speeds(result) == (None, None, None)
    monkeypatch.setattr(tts_eval, "REPORTS", tmp_path)
    tts_eval.write_report([{"candidate": "fake", "languages": {"ko": result}}], Namespace(tag="t_fail"))
    (report,) = tmp_path.glob("tts_t_fail_*.md")
    assert "| fake | 100.00% | - | - | - | - | None | 1 |" in report.read_text(encoding="utf-8")


def tone(seconds: float, amplitude: float, rate: int = 1000, lead: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    body = amplitude * np.sin(2 * np.pi * 50 * t)
    silence = np.zeros(int(lead * rate))
    return np.concatenate([silence, body, silence]).astype(np.float32)


def make_audio(root, ids):
    for candidate, amplitude in (("alpha", 0.05), ("beta", 0.5)):
        folder = root / "t_dev" / candidate / "ko"
        folder.mkdir(parents=True)
        for sentence in ids:
            (folder / f"{sentence}.wav").write_bytes(wav(tone(1.0 + sentence / 10, amplitude)))


def test_blind_pairs_hide_the_models_match_loudness_and_keep_the_key_apart(tmp_path):
    ids = (11, 12, 13)
    make_audio(tmp_path / "audio", ids)
    out, key_path = tmp_path / "ab", tmp_path / "ab-key.json"
    texts = {11: "열하나.", 12: "열둘.", 13: "열셋."}
    tts_ab.build(tmp_path / "audio", "t_dev", "alpha", "beta", ids, texts, out, key_path)

    page = (out / "index.html").read_text(encoding="utf-8")
    assert "alpha" not in page and "beta" not in page and "t_dev" not in page
    assert all(text in page for text in texts.values())
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert sorted(pair["id"] for pair in key["pairs"]) == list(ids)
    assert all({pair["1"], pair["2"]} == {"alpha", "beta"} for pair in key["pairs"])
    assert not (out / key_path.name).exists()
    for pair in key["pairs"]:
        levels = []
        for side in ("1", "2"):
            samples, rate = read_wav(out / "clips" / f"{pair['pair']:02d}-{side}.wav")
            assert rate == 1000
            levels.append(20 * np.log10(np.sqrt(np.mean(samples[np.abs(samples) > 0.003] ** 2))))
            # Lead and tail silence is trimmed to about 0.1 s.
            assert np.abs(samples[: int(0.08 * rate)]).max() < 0.003
            assert len(samples) < (1.0 + pair["id"] / 10 + 0.25) * rate
        assert levels[0] == pytest.approx(levels[1], abs=0.5)
    # The same seed gives the same pages.
    again = tmp_path / "again-key.json"
    tts_ab.build(tmp_path / "audio", "t_dev", "alpha", "beta", ids, texts, tmp_path / "ab2", again)
    assert json.loads(again.read_text(encoding="utf-8"))["pairs"] == key["pairs"]


def test_unblinding_counts_preferences_and_unacceptable_clips_per_model(tmp_path):
    key = {
        "a": "alpha",
        "b": "beta",
        "pairs": [
            {"pair": 1, "id": 11, "1": "alpha", "2": "beta"},
            {"pair": 2, "id": 12, "1": "beta", "2": "alpha"},
            {"pair": 3, "id": 13, "1": "alpha", "2": "beta"},
            {"pair": 4, "id": 14, "1": "beta", "2": "alpha"},
        ],
    }
    answers = "01 pref=1 bad=-\n02 pref=1 bad=2\n03 pref=same bad=1,2\n04 pref=none bad=-\ncomment: 괜찮음\n"
    summary = tts_ab.unblind(key, answers)
    assert summary["preferred"] == {"alpha": 1, "beta": 1}
    assert (summary["same"], summary["unanswered"]) == (1, 1)
    assert summary["unacceptable"] == {"alpha": 2, "beta": 1}
    assert summary["comment"] == "괜찮음"
    with pytest.raises(ValueError, match="pair 05"):
        tts_ab.unblind(key, "05 pref=1 bad=-\n")
