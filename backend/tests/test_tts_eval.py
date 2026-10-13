"""The speech synthesis trial tools (T78, docs/experiments.md 12): sentence export for the container, the
recognition load, the digit-sentence sub-metric, the timing summary and the blind A/B pages. No model runs."""

# [파일 역할] 음성 합성 시험 도구 eval/tts_eval.py와 eval/tts_ab.py를 모델 없이 검증한다(docs/experiments.md
# 12·14절).
# [무엇을 확인하나] 컨테이너용 문장 내보내기와 다시 읽기, 재인식 채점(합성 실패는 완전히 틀린 것으로, 아라비아
# 숫자 문장은 따로), 태그별 기록 고르기, 합성 중 인식 모델을 계속 돌리는 부하(SttLoad), GPU 메모리 읽기, 속도
# 요약(성공이 0·1개여도), 적재·첫 호출 시간과 제공자 기록, 블라인드 A/B 청취 페이지 만들기와 풀기.
# [왜 중요한가] 이 도구들의 숫자와 표가 후보 채택 규칙의 판정 근거였다. 실패가 많은 후보의 보고서가 깨지거나,
# 블라인드 페이지에 모델 이름이 새면 판정 자체를 믿을 수 없게 된다.

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

# tts_eval: 합성·채점·속도 요약. tts_ab: 블라인드 청취 폴더 만들기(build)와 답 풀기(unblind). BACKEND는 자식
# 프로세스 import 확인의 작업 폴더다.
from eval import tts_ab, tts_eval

BACKEND = Path(__file__).resolve().parents[1]


# float 파형을 16비트 모노 WAV 바이트로 만드는 도우미(가짜 합성 결과와 청취용 음성 파일에 씀).
def wav(samples: np.ndarray, rate: int = 1000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


# WAV 파일을 float 파형과 표본화율로 읽는 도우미(청취 클립의 음량·길이 확인용).
def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as clip:
        data = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2").astype(np.float32) / 32767
        return data, clip.getframerate()


# CI와 API 컨테이너에는 pyarrow·jiwer가 없으므로, 두 모듈 import를 막은 새 프로세스에서도 세 도구를 불러올 수
# 있는지 확인한다. 합성은 컨테이너 안에서 돌기 때문이다.
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


# 측정 전에 정한 보조 지표의 대상: 아라비아 숫자로 쓴 문장(13위)만 '숫자 문장'이고, 한글로 쓴 수(열세 번째)는
# 아니다. 숫자·단위 읽기를 따로 보는 부지표로, 판정에는 쓰지 않고 적기만 한다(12절).
def test_digit_sentences_are_those_with_arabic_digits():
    assert tts_eval.has_digits("13위에서 경기를 마쳤다.")
    assert not tts_eval.has_digits("열세 번째로 경기를 마쳤다.")


# export가 FLEURS 문장을 id 순서로 중복 없이(같은 문장을 여러 사람이 읽어 id가 겹침) JSON으로 쓰고,
# sentences가 그 JSON을 읽을 때는 parquet를 전혀 읽지 않는지(읽으면 pytest.fail) 확인한다. limit은 앞에서부터
# 자른다.
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


# 두 문자열의 편집 거리(레벤시테인)를 한 줄짜리 DP 배열로 계산하는 시험용 도우미.
def edits(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


# jiwer 없이 말뭉치 CER(전체 편집 수 ÷ 전체 정답 길이)을 계산해 score_language에 넘길 대역 함수.
def corpus_cer(language: str, references: list[str], hypotheses: list[str]) -> float:
    """Total character edits over total reference length, as jiwer.cer (the eval group, not in CI)."""
    return sum(map(edits, references, hypotheses)) / sum(map(len, references))


# 채점 규칙: 합성에 실패한 문장은 빈 인식 결과로 보아 모든 글자가 틀린 것으로 센다(실패가 오류율을 낮추지
# 않게). 정답 8글자 중 숫자 문장의 1글자(에→애)와 실패 문장의 2글자로 3/8이고, 숫자 문장만의 CER은 1/3이다.
# 문장당 합성 시간 p50은 성공한 두 문장(0.2, 0.4)의 0.3초다.
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


# 태그 t_dev의 파일 패턴 tts_t_dev_*는 t_dev_load의 파일도 잡으므로, 파일 이름이 정확히
# tts_<태그>_<후보>_synth.json인 기록만 고르는지 확인한다. 부하 실행 기록이 일반 채점에 섞이는 것을 막는다.
def test_score_reads_only_the_synthesis_records_of_its_own_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_eval, "REPORTS", tmp_path)
    for tag, candidate in (("t_dev", "melo"), ("t_dev", "supertonic"), ("t_dev_load", "supertonic")):
        record = {"tag": tag, "candidate": candidate, "split": "validation", "languages": {}}
        (tmp_path / f"tts_{tag}_{candidate}_synth.json").write_text(json.dumps(record), encoding="utf-8")
    assert [record["candidate"] for record in tts_eval.synth_records("t_dev")] == ["melo", "supertonic"]
    assert [record["tag"] for record in tts_eval.synth_records("t_dev_load")] == ["t_dev_load"]


# 가짜 인식 모델: 호출 언어를 기록하고 2 ms씩 걸리며, 처음 한 번 또는 매번 실패하게 할 수 있다.
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


# SttLoad는 서버의 모델 스레드가 인식을 쉬지 않고 하는 상황을 흉내 내려고 다른 스레드에서 클립을 번갈아(en,
# ko, en, ko …) 계속 인식한다. start는 첫 인식이 끝난 뒤(모델이 데워진 뒤) 돌아오고, stop 뒤에는 호출이 더
# 늘지 않으며 스레드도 끝나는지 확인한다.
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


# 부하 인식이 첫 호출부터 실패하면(가짜 'no GPU' 오류) start가 RuntimeError로 측정을 멈추는지 확인한다. 두
# 번째 부분은 실패 없는 부하(fail_first=False)가 오류 0으로 끝나는지 확인한다.
def test_a_recognition_load_that_cannot_run_stops_the_measurement():
    load = tts_eval.SttLoad(CountingStt(fail_all=True), [(b"a", "en")])
    with pytest.raises(RuntimeError, match="recognition load"):
        load.start(timeout=5)
    # A later failure is counted, not fatal.
    load = tts_eval.SttLoad(CountingStt(fail_first=False), [(b"a", "en")])
    load.start(timeout=5)
    assert load.stop()["errors"] == 0


# 부하용 클립은 <언어>_<id>.wav 이름 순서로 읽고, 언어는 파일 이름 앞부분에서 얻으며 다른 파일은 무시한다.
def test_load_clips_take_the_language_from_the_file_name(tmp_path):
    (tmp_path / "ko_12.wav").write_bytes(b"k")
    (tmp_path / "en_3.wav").write_bytes(b"e")
    (tmp_path / "notes.txt").write_text("x")
    assert tts_eval.load_clips(tmp_path) == [(b"e", "en"), (b"k", "ko")]


# nvidia-ml-py가 없는 컨테이너에서는 nvidia-smi 출력(MB)을 읽고, nvidia-smi도 없으면 None을 돌려주는지
# 확인한다(실행 함수를 주입해 실제 명령은 부르지 않음).
def test_device_memory_falls_back_to_nvidia_smi():
    def run(*args, **kwargs):
        return Namespace(stdout="1234\n")

    assert tts_eval.nvidia_smi_memory_mb(run) == 1234.0

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert tts_eval.nvidia_smi_memory_mb(missing) is None


# 재인식 없이 속도만 요약하는 timing: 생성 음성 1초당 합성 시간 p50(0.1), 문장당 합성 시간 p50(0.2초), 부하
# 인식 호출 수(12)를 JSON과 마크다운 표에 적는지 확인한다.
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


# 가짜 합성 후보: 읽은 글자를 기록하고 0.5초짜리 WAV를 돌려주며, CUDA 시험처럼 모델 이름과 세션별 제공자를
# 가진다.
class FakeSpeech:
    model_name = "fake/2-step/cuda"
    providers = {"vocoder": ["CUDAExecutionProvider", "CPUExecutionProvider"]}

    def __init__(self, language) -> None:
        self.texts: list[str] = []

    def synthesize(self, text, language):
        self.texts.append(text)
        return Namespace(wav=wav(np.full(500, 0.1, np.float32)))


# 합성 단계 기록: 적재 시간과 첫 호출 시간을 문장 통계와 따로 재고, 세션의 제공자를 남기며, GPU 메모리를 적재
# 뒤(+500MB)뿐 아니라 마지막 문장 뒤(+700MB)에도 읽는지 확인한다. ONNX Runtime의 메모리 영역은 최고치를
# 유지하므로 실행 뒤 값이 최대치가 된다(docstring).
# 후보 목록·음성 폴더·보고서 폴더·GPU 메모리 읽기는 monkeypatch로 바꾼다. 메모리 값은 적재 전·적재 뒤·실행 뒤
# 순서로 1000·1500·1700을 준다. 이어서 timing 보고서에도 같은 값과 제공자가 옮겨지는지 본다.
def test_synth_reports_load_and_first_call_times_providers_and_memory_after_the_run(tmp_path, monkeypatch):
    """T82 (docs/experiments.md 14): the first call and the load are timed apart from the sentences, the
    sessions' providers are kept, and GPU memory is read after the run too (ONNX Runtime's arena keeps its
    high-water mark, so that is the peak)."""
    from eval import tts_candidates

    rows = [{"id": i, "raw_transcription": f"문장 {i}.", "transcription": f"문장 {i}"} for i in (1, 2, 3)]
    (tmp_path / "validation_ko.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(tts_candidates, "CANDIDATES", {"fake": (FakeSpeech, ["ko"])})
    monkeypatch.setattr(tts_eval, "AUDIO", tmp_path / "audio")
    monkeypatch.setattr(tts_eval, "REPORTS", tmp_path)
    readings = iter([1000.0, 1500.0, 1700.0])  # before the load, after it, after the last sentence
    monkeypatch.setattr(tts_eval, "gpu_memory_mb", lambda: next(readings))
    args = Namespace(
        candidate="fake",
        languages=["ko"],
        tag="t_gpu",
        split="validation",
        limit=None,
        sentences_dir=tmp_path,
        stt_load=None,
        stt_load_device="cuda",
    )
    tts_eval.synth(args)
    record = json.loads((tmp_path / "tts_t_gpu_fake_synth.json").read_text(encoding="utf-8"))
    ko = record["languages"]["ko"]
    assert (ko["vram_mb"], ko["vram_after_run_mb"]) == (500, 700)
    assert ko["providers"] == FakeSpeech.providers
    assert ko["load_s"] >= 0 and ko["first_call_s"] >= 0
    assert [item["id"] for item in ko["items"]] == [1, 2, 3] and ko["failures"] == 0
    assert (tmp_path / "audio" / "t_gpu" / "fake" / "ko" / "3.wav").exists()

    tts_eval.timing(Namespace(tag="t_gpu"))
    (report,) = tmp_path.glob("tts_t_gpu_timing_*.json")
    (summary,) = json.loads(report.read_text(encoding="utf-8"))
    values = summary["languages"]["ko"]
    assert (values["vram_mb"], values["vram_after_run_mb"]) == (500, 700)
    assert values["providers"] == FakeSpeech.providers
    assert values["first_call_s"] == ko["first_call_s"] and values["load_s"] == ko["load_s"]
    table = report.with_suffix(".md").read_text(encoding="utf-8")
    assert "| 500 | 700 | CUDAExecutionProvider |" in table


# 합성 실패 항목의 모양(모델이 빈 음성을 돌려준 경우).
FAILED = {"failure": "ModelError: the model returned no audio"}


# 속도 요약의 세 값(속도 p50·p95, 문장당 합성 p50)만 뽑는 도우미.
def speeds(values: dict) -> tuple:
    return values["speed_p50"], values["speed_p95"], values["synth_s_p50"]


# 회귀 테스트(커밋 fdfb911): 성공한 문장이 0개면 속도는 비어 있고(None, 표에는 '-'), 1개면 그 값이 p50이자
# p95다. 예전에는 성공 0·1개에서 통계 함수가 멈춰, 정작 보여야 할 실패 수가 보고서에 나오지 않았다.
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


# 모든 문장이 실패한 후보도 채점 보고서가 만들어지는지 확인한다. 오류율은 100%, 속도는 비고, 인식은 한 번도
# 부르지 않는다(부르면 pytest.fail).
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


# 앞뒤에 0.5초 무음이 붙은 50 Hz 사인파. 길이와 크기를 바꿔 두 후보의 음성을 흉내 낸다.
def tone(seconds: float, amplitude: float, rate: int = 1000, lead: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    body = amplitude * np.sin(2 * np.pi * 50 * t)
    silence = np.zeros(int(lead * rate))
    return np.concatenate([silence, body, silence]).astype(np.float32)


# 합성 음성 폴더 구조(<태그>/<후보>/<언어>/<id>.wav)로 두 후보의 음성을 만든다. alpha는 작게(0.05), beta는
# 크게(0.5) 만들어 음량 맞추기를 시험한다.
def make_audio(root, ids):
    for candidate, amplitude in (("alpha", 0.05), ("beta", 0.5)):
        folder = root / "t_dev" / candidate / "ko"
        folder.mkdir(parents=True)
        for sentence in ids:
            (folder / f"{sentence}.wav").write_bytes(wav(tone(1.0 + sentence / 10, amplitude)))


# 블라인드 청취 폴더 확인: 페이지에 모델 이름·태그가 없고 문장 글은 있으며, 답 열쇠에는 모든 문장이 있고
# 쌍마다 두 모델이 1·2에 하나씩 배정되며, 열쇠 파일은 청취 폴더 안에 없다.
# 클립마다 앞뒤 무음을 약 0.1초로 다듬고, 원래 10배 차이 나던 두 모델의 크기를 0.5 dB 안으로 맞추는지 본다.
# 길이나 크기로 모델을 알아채지 못하게 하려는 것이다(tts_ab 모듈 주석). 같은 시드로 다시 만들면 같은 쌍 배치가
# 나오는지도 확인한다.
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


# 답 풀기: 청취자가 고른 번호(1·2)를 열쇠로 모델 이름으로 바꿔 모델별 선호 수와 '쓰기 싫을 만큼 부자연스러움'
# 표시 수를 센다. 'same'은 비슷함, 'none'은 답 없음으로 세고, comment 줄은 느낀 점으로 남긴다. 열쇠에 없는 쌍
# 번호(05)는 잘못된 답으로 거부한다.
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
