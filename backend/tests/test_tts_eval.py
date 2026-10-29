"""음성 합성 시험 도구(T78, docs/experiments.md 12절)를 검증한다: 컨테이너용 문장 내보내기, 인식 부하,
숫자 문장 부지표, 시간 요약, 블라인드 A/B 페이지. 모델은 돌리지 않는다."""

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


# CI와 API 컨테이너에는 pyarrow·jiwer(FLEURS 읽기와 채점에 필요)가 없으므로, 두 모듈 import를 막은 새
# 프로세스에서도 세 도구를 불러올 수 있는지 확인한다. 합성은 컨테이너 안에서 돌기 때문이다.
def test_the_tools_import_without_the_eval_group():
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
    """전체 글자 편집 수 ÷ 전체 정답 길이. jiwer.cer(eval 그룹이라 CI에는 없음)와 같은 계산이다."""
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


# 가짜 인식 모델: 호출 언어를 기록하고 2 ms씩 걸리며, 첫 호출 뒤의 호출(fail_later) 또는 매번(fail_all)
# 실패하게 할 수 있다.
class CountingStt:
    def __init__(self, fail_later: bool = False, fail_all: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_later, self.fail_all = fail_later, fail_all

    def transcribe(self, audio: bytes, language: str):
        self.calls.append(language)
        time.sleep(0.002)
        if self.fail_all or (self.fail_later and len(self.calls) > 1):
            raise RuntimeError("no GPU")
        return Namespace(text="ok")


# SttLoad는 서버의 모델 스레드가 인식을 쉬지 않고 하는 상황을 흉내 내려고 다른 스레드에서 클립을 번갈아(en,
# ko, en, ko …) 계속 인식한다. start는 첫 인식이 끝난 뒤(모델이 데워진 뒤) 돌아오고, stop 뒤에는 호출이 더
# 늘지 않으며 스레드도 끝나는지 확인한다.
def test_the_recognition_load_runs_back_to_back_until_stopped():
    stt = CountingStt()
    load = tts_eval.SttLoad(stt, [(b"a", "en"), (b"b", "ko")])
    load.start(timeout=5)
    assert stt.calls  # 준비됨: 합성 시간을 재기 전에 인식이 한 번 이상 끝났다
    time.sleep(0.05)
    summary = load.stop()
    assert summary["calls"] >= 4 and summary["errors"] == 0 and summary["call_s_p50"] > 0
    assert stt.calls[:4] == ["en", "ko", "en", "ko"]
    count = len(stt.calls)
    time.sleep(0.02)
    assert len(stt.calls) == count  # 완전히 멈췄다
    assert not any(thread.name == "stt-load" for thread in threading.enumerate())


# 부하 인식이 첫 호출부터 실패하면(가짜 'no GPU' 오류) start가 RuntimeError로 측정을 멈추는지 확인한다. 두
# 번째 부분은 첫 인식이 성공한 뒤의 실패는 세기만 하고 부하를 멈추지 않는지 본다: 첫 호출만 성공하는
# 인식기로 start가 돌아오고, 실패가 세 번 쌓이도록(부하가 계속 돎) 기다린 뒤 stop의 요약이 성공 1번과
# 실패를 센다. 2026-10-13 전에는 실패 없는 인식기(fail_first=False)를 써서 이 경로를 확인하지 못했다.
def test_a_recognition_load_that_cannot_run_stops_the_measurement():
    load = tts_eval.SttLoad(CountingStt(fail_all=True), [(b"a", "en")])
    with pytest.raises(RuntimeError, match="recognition load"):
        load.start(timeout=5)
    # 첫 인식 뒤의 실패는 세기만 하고 측정을 멈추지 않는다.
    stt = CountingStt(fail_later=True)
    load = tts_eval.SttLoad(stt, [(b"a", "en")])
    load.start(timeout=5)
    deadline = time.monotonic() + 5
    while len(stt.calls) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    summary = load.stop()
    assert summary["errors"] >= 3 and summary["calls"] == summary["errors"] + 1


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
# 유지하므로 실행 뒤 값이 최대치가 된다(T82, docs/experiments.md 14절).
# 후보 목록·음성 폴더·보고서 폴더·GPU 메모리 읽기는 monkeypatch로 바꾼다. 메모리 값은 적재 전·적재 뒤·실행 뒤
# 순서로 1000·1500·1700을 준다. 이어서 timing 보고서에도 같은 값과 제공자가 옮겨지는지 본다.
def test_synth_reports_load_and_first_call_times_providers_and_memory_after_the_run(tmp_path, monkeypatch):
    """T82(docs/experiments.md 14절): 첫 호출과 적재는 문장과 따로 시간을 재고, 세션의 제공자를 남기며,
    GPU 메모리를 실행 뒤에도 읽는다(ONNX Runtime의 arena는 최고치를 유지하므로 그 값이 최대치다)."""
    from eval import tts_candidates

    rows = [{"id": i, "raw_transcription": f"문장 {i}.", "transcription": f"문장 {i}"} for i in (1, 2, 3)]
    (tmp_path / "validation_ko.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(tts_candidates, "CANDIDATES", {"fake": (FakeSpeech, ["ko"])})
    monkeypatch.setattr(tts_eval, "AUDIO", tmp_path / "audio")
    monkeypatch.setattr(tts_eval, "REPORTS", tmp_path)
    readings = iter([1000.0, 1500.0, 1700.0])  # 적재 전, 적재 뒤, 마지막 문장 뒤
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
        seed=None,
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
    """보고서가 보여야 할 것은 실패다: 성공이 없으면 속도는 비고, 성공이 하나면 그 값이 속도다."""
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
            # 앞뒤 무음은 약 0.1초로 잘린다.
            assert np.abs(samples[: int(0.08 * rate)]).max() < 0.003
            assert len(samples) < (1.0 + pair["id"] / 10 + 0.25) * rate
        assert levels[0] == pytest.approx(levels[1], abs=0.5)
    # 같은 시드면 같은 페이지가 나온다.
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


# 여기부터 melo-stages(docs/experiments.md 17절): MeloTTS 합성 한 번을 단계별로 나누는 도구를 가짜 모듈로
# 검증한다. 가짜 모듈은 고정 리비전 melo/api.py의 tts_to_file처럼 함수를 '모듈 이름으로' 찾아
# 부른다(split_sentence는 api의 전역, get_text_for_tts_infer는 utils 모듈, empty_cache는 torch.cuda, 잇기는
# TTS 클래스의 정적 메서드).
class FakeMeloModel:
    def infer(self, piece):
        time.sleep(0.002)
        return np.ones(len(piece), np.float32)


class FakeMeloTTS:
    def __init__(self):
        self.model = FakeMeloModel()

    @staticmethod
    def audio_numpy_concat(segments, sr, speed=1.0):
        return np.concatenate(segments)

    def tts_to_file(self, text, api, utils, torch):
        pieces = api.split_sentence(text, language_str="KR")
        audio = []
        for piece in pieces:
            utils.get_text_for_tts_infer(piece)
            audio.append(self.model.infer(piece))
        torch.cuda.empty_cache()
        return self.audio_numpy_concat(audio, sr=44100)


def fake_melo_modules():
    from types import SimpleNamespace

    api = SimpleNamespace(split_sentence=lambda text, language_str: text.split("."), TTS=FakeMeloTTS)
    utils = SimpleNamespace(get_text_for_tts_infer=lambda piece: time.sleep(0.001))
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    return api, utils, torch


# 감싸기를 켠 동안에는 다섯 단계의 시간과 조각 수가 모이고, 되돌리면 원래 함수가 모두 제자리에 있고 인스턴스에
# 덧붙인 infer도 사라진다. 결과(소리)는 감싸기 전후로 같다.
def test_melo_stages_wrap_each_step_and_undo_restores_the_originals():
    api, utils, torch = fake_melo_modules()
    originals = (api.split_sentence, utils.get_text_for_tts_infer, torch.cuda.empty_cache)
    concat = FakeMeloTTS.__dict__["audio_numpy_concat"]
    tts = FakeMeloTTS()
    plain = tts.tts_to_file("가나.다라마.바", api, utils, torch)
    clock, undo = tts_eval.instrument_melo(api, utils, tts, torch)
    traced = tts.tts_to_file("가나.다라마.바", api, utils, torch)
    undo()
    assert np.array_equal(plain, traced)
    assert clock.pieces == 3
    assert set(clock.seconds) == set(tts_eval.MELO_STAGES)
    assert clock.seconds["infer"] >= 0.005 and clock.seconds["text"] >= 0.002
    assert (api.split_sentence, utils.get_text_for_tts_infer, torch.cuda.empty_cache) == originals
    assert FakeMeloTTS.__dict__["audio_numpy_concat"] is concat
    assert "infer" not in vars(tts.model)


# 합성할 글은 서비스 측정 보고서의 영어 원문 마디가 받은 한국어 번역문이다. 두 모드에서 같은 번역문은 한 번만,
# 실패한 마디와 한국어 원문 마디는 뺀다.
def test_stage_texts_take_the_korean_translations_of_english_speech_once(tmp_path):
    def final(text, target="ko"):
        return {"outcome": "final", "result": {"translated_text": text, "target_lang": target}}

    live = {
        "details": {
            "en": [{"key": "1", **final("하나.")}, {"key": "2", "outcome": "error", "result": None}],
            "ko": [{"key": "3", **final("One.", "en")}],
        }
    }
    conversation = {"details": {"en": [{"key": "1", **final("하나.")}, {"key": "4", **final("넷. 다섯.")}]}}
    paths = []
    for name, report in (("live", live), ("conversation", conversation)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        paths.append(path)
    assert tts_eval.stage_texts(paths) == [{"id": "1", "text": "하나."}, {"id": "4", "text": "넷. 다섯."}]


# 요약은 문장별 단계 시간의 중앙값과 전체 합에서 단계가 차지하는 몫, 조각 2개 이상인 문장의 비율을 낸다.
def test_stage_summary_gives_shares_and_the_multi_piece_ratio():
    def row(total, infer, cache, pieces):
        stages = {"split": 1.0, "text": 10.0, "infer": infer, "empty_cache": cache, "concat": 4.0}
        return {"total_ms": total, "plain_ms": total - 5, "pieces": pieces, "stages_ms": stages}

    summary = tts_eval.summarize_stages([row(500, 400, 50, 1), row(700, 560, 70, 2), row(900, 720, 90, 3)])
    assert summary["total_p50_ms"] == 700 and summary["plain_p50_ms"] == 695
    assert summary["stages"]["empty_cache"]["share"] == pytest.approx(210 / 2100)
    assert summary["stages"]["infer"]["p50_ms"] == 560
    assert summary["multi_piece_ratio"] == pytest.approx(2 / 3)
    assert summary["pieces"] == {"1": 1, "2": 1, "3": 1}
    assert summary["other_p50_ms"] == pytest.approx(700 - (1 + 10 + 560 + 70 + 4))


# 여기부터 18절: melo-stages --regex-cache. off는 17절과 같고(캐시를 건드리지 않음), on은 서비스의
# MELO_REGEX_CACHE=true처럼 처음부터 켠 채로 재고, alternate는 문장마다 g2pkk 정규식 캐시를 켜고 끄며 같은
# 문장을 두 조건으로 잰다. 시작 조건도 문장마다 바꿔, 두 번째로 재는 쪽이 늘 같은 조건이 되지 않게 한다.
STAGE_TEXTS = [{"id": "1", "text": "가나.다"}, {"id": "2", "text": "라마"}, {"id": "3", "text": "바.사.아"}]


class FakeRegexSwitch:
    """app.services.tts._cache_g2pkk_regex 흉내. 켜면 (바꾼 모듈 이름, 되돌리기)를 돌려주고, 이미 켜져 있으면
    진짜 함수처럼 빈 이름 목록과 아무것도 하지 않는 되돌리기를 돌려준다."""

    def __init__(self, on: bool = False) -> None:
        self.on = on
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.on:
            return (), lambda: None
        self.on = True

        def restore():
            self.on = False

        return ("g2pkk.g2pkk",), restore


class FakeMeloService:
    """MeloTextToSpeech 흉내. synthesize가 가짜 melo 함수들(fake_melo_modules)을 지나며, 부를 때의 캐시
    상태(off/on)를 calls에 적는다. fail_call번째 호출은 합성 실패(예외)로 끝난다."""

    def __init__(self, switch: FakeRegexSwitch, fail_call: int | None = None) -> None:
        self.api, self.utils, self.torch = fake_melo_modules()
        self._model = FakeMeloTTS()
        self.switch = switch
        self.calls: list[tuple[str, str]] = []
        self.fail_call = fail_call

    def synthesize(self, text, language):
        from types import SimpleNamespace

        self.calls.append((text, "on" if self.switch.on else "off"))
        if len(self.calls) == self.fail_call:
            raise RuntimeError("synthesis failed")
        audio = self._model.tts_to_file(text, self.api, self.utils, self.torch)
        return SimpleNamespace(duration_ms=float(len(audio)))


def measure(service: FakeMeloService, mode: str) -> list[dict]:
    return tts_eval.measure_melo_stages(
        service, STAGE_TEXTS, mode, service.api, service.utils, service.torch, service.switch
    )


def test_regex_cache_orders_alternate_the_first_condition_every_sentence():
    assert tts_eval.regex_cache_orders("alternate", 3) == [("off", "on"), ("on", "off"), ("off", "on")]
    assert tts_eval.regex_cache_orders("off", 2) == [("off",), ("off",)]
    assert tts_eval.regex_cache_orders("on", 1) == [("on",)]


# alternate: 첫 문장으로 끔·켬을 한 번씩 데운 뒤(통계에서 뺌), 문장마다 두 조건을 정한 순서로 잰다. 한 조건의
# 두 합성(계측 없이 한 번, 계측을 켜고 한 번)은 모두 그 조건에서 불리고, 끝나면 캐시가 꺼져 있고 감쌌던 melo
# 함수들도 제자리다.
def test_alternate_measures_each_sentence_under_both_conditions_and_turns_the_cache_off_again():
    switch = FakeRegexSwitch()
    service = FakeMeloService(switch)
    originals = (
        service.api.split_sentence,
        service.utils.get_text_for_tts_infer,
        service.torch.cuda.empty_cache,
    )
    rows = measure(service, "alternate")
    assert [(row["id"], row["regex_cache"]) for row in rows] == [
        ("1", "off"),
        ("1", "on"),
        ("2", "on"),
        ("2", "off"),
        ("3", "off"),
        ("3", "on"),
    ]
    assert service.calls[:2] == [("가나.다", "off"), ("가나.다", "on")]
    texts = {item["id"]: item["text"] for item in STAGE_TEXTS}
    assert service.calls[2:] == [(texts[row["id"]], row["regex_cache"]) for row in rows for _ in range(2)]
    assert switch.on is False
    assert [row["pieces"] for row in rows] == [2, 2, 1, 1, 3, 3]
    assert all(row["stages_ms"]["infer"] > 0 for row in rows)
    assert (
        service.api.split_sentence,
        service.utils.get_text_for_tts_infer,
        service.torch.cuda.empty_cache,
    ) == (originals)


# off와 on은 측정 중에 캐시를 바꾸지 않는다: off는 17절 그대로 꺼진 채, on은 생성자가 켠 채로 모든 합성을
# 한다.
@pytest.mark.parametrize("mode", ["off", "on"])
def test_off_and_on_keep_the_cache_as_it_is_for_the_whole_run(mode):
    switch = FakeRegexSwitch(on=mode == "on")
    service = FakeMeloService(switch)
    rows = measure(service, mode)
    assert [row["regex_cache"] for row in rows] == [mode] * 3
    assert switch.calls == 0 and switch.on is (mode == "on")
    assert {state for _, state in service.calls} == {mode}
    assert len(service.calls) == 1 + 2 * 3  # 데우기 한 번 + 문장마다 두 번


# alternate는 캐시가 이미 켜져 있으면(끔 조건이 끔이 아니게 됨) 재지 않고 멈춘다. 켬 조건의 합성이 실패해도
# 캐시는 꺼지고 감싼 melo 함수는 제자리로 돌아간다(측정 뒤 상태가 남지 않음).
def test_alternate_refuses_a_cache_already_on_and_restores_the_state_after_a_failure():
    with pytest.raises(RuntimeError, match="already on"):
        measure(FakeMeloService(FakeRegexSwitch(on=True)), "alternate")
    switch = FakeRegexSwitch()
    service = FakeMeloService(switch, fail_call=8)  # 2번 문장 켬 조건의 계측 합성
    split = service.api.split_sentence
    with pytest.raises(RuntimeError, match="synthesis failed"):
        measure(service, "alternate")
    assert service.calls[-1] == ("라마", "on")
    assert switch.on is False and service.api.split_sentence is split


# alternate 요약: 조건마다 17절과 같은 요약(summarize_stages)을 내고, 같은 문장의 끔 − 켬 차이 중앙값을 전처리
# 단계와 합성 전체로 따로 낸다.
def test_the_alternate_summary_gives_each_condition_and_the_paired_saving():
    def row(id_, cache, text_ms):
        stages = {"split": 1.0, "text": text_ms, "infer": 100.0, "empty_cache": 2.0, "concat": 5.0}
        total = 110.0 + text_ms
        return {
            "id": id_,
            "regex_cache": cache,
            "total_ms": total,
            "plain_ms": total - 1,
            "pieces": 1,
            "stages_ms": stages,
        }

    rows = [
        row("1", "off", 300.0),
        row("1", "on", 40.0),
        row("2", "on", 50.0),
        row("2", "off", 420.0),
        row("3", "off", 350.0),
        row("3", "on", 30.0),
    ]
    summary = tts_eval.summarize_regex_cache(rows)
    assert summary["off"]["stages"]["text"]["p50_ms"] == 350.0
    assert summary["on"]["stages"]["text"]["p50_ms"] == 40.0
    assert summary["on"]["sentences"] == 3
    assert summary["text_saving_p50_ms"] == 320.0  # 문장별 260, 370, 320
    assert summary["total_saving_p50_ms"] == 320.0


# 보고서: alternate는 두 조건의 단계 표와 짝지은 차이를, off·on은 17절과 같은 표 하나를 쓴다. JSON에는 조건과
# g2pkk 정규식 캐시의 적중·실패 수가 함께 남는다.
def test_the_stage_report_names_the_regex_cache_condition(tmp_path):
    rows = [
        {
            "id": "1",
            "regex_cache": cache,
            "chars": 3,
            "pieces": 1,
            "audio_ms": 900.0,
            "plain_ms": 400.0,
            "total_ms": 401.0,
            "stages_ms": {"split": 0.1, "text": text, "infer": 150.0, "empty_cache": 1.0, "concat": 5.0},
        }
        for cache, text in (("off", 240.0), ("on", 20.0))
    ]
    record = {
        "tag": "t",
        "device": "cpu",
        "regex_cache": "alternate",
        "regex_cache_info": {
            "after_warm_up": {"hits": 799, "misses": 571, "currsize": 571},
            "end": {"hits": 9000, "misses": 571, "currsize": 571},
        },
        "summary": tts_eval.summarize_regex_cache(rows),
        "rows": rows,
    }
    md = tts_eval.write_stage_report(record, tmp_path / "tts_t_melo_stages_x")
    text = md.read_text(encoding="utf-8")
    assert (
        "정규식 캐시 alternate" in text
        and "| 전처리·BERT | 240.0ms |" in text
        and "| 전처리·BERT | 20.0ms |" in text
    )
    assert "220ms" in text and "데우기 뒤 새 컴파일 0" in text
    assert json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))["regex_cache"] == "alternate"
    single = {
        **record,
        "regex_cache": "off",
        "summary": tts_eval.summarize_stages(rows[:1]),
        "rows": rows[:1],
    }
    text = tts_eval.write_stage_report(single, tmp_path / "tts_t2_melo_stages_x").read_text(encoding="utf-8")
    assert "정규식 캐시 off" in text and text.count("| 전처리·BERT |") == 1


# 가짜 torch: synth --seed가 부르는 manual_seed만 흉내 내고, 받은 시드를 차례로(seeds)와 마지막
# 값(current)으로 남긴다. 실제 torch 없이 '어느 문장 합성 직전에 어떤 시드가 걸렸나'를 보려는 것이다.
class FakeTorch:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.current: int | None = None

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)
        self.current = seed


# 시드 시험의 공통 준비: 문장 ID가 정렬 순서와 다른 세 문장(7·3·12)을 export JSON으로 쓰고, 합성할 때 그
# 순간의 가짜 torch 시드를 글과 함께 heard에 적는 가짜 후보를 등록한다. 돌려주는 것은 main에 넘길 공통 인자다.
def _seed_setup(tmp_path, monkeypatch, fake_torch, heard: list) -> list[str]:
    from eval import tts_candidates

    class Speech(FakeSpeech):
        def synthesize(self, text, language):
            heard.append((text, getattr(fake_torch, "current", None)))
            return super().synthesize(text, language)

    rows = [{"id": i, "raw_transcription": f"문장 {i}.", "transcription": f"문장 {i}"} for i in (7, 3, 12)]
    (tmp_path / "validation_ko.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(tts_candidates, "CANDIDATES", {"fake": (Speech, ["ko"])})
    monkeypatch.setattr(tts_eval, "AUDIO", tmp_path / "audio")
    monkeypatch.setattr(tts_eval, "REPORTS", tmp_path)
    monkeypatch.setattr(tts_eval, "gpu_memory_mb", lambda: None)
    return ["synth", "--candidate", "fake", "--sentences-dir", str(tmp_path)]


# synth --seed <n>(docs/experiments.md 19절): 문장마다 합성 직전에 그 문장 ID에서 정한 시드로
# torch.manual_seed를 부르고, 기록에 기준 시드와 문장별 시드를 남긴다. 같은 문장은 몇 문장을 함께
# 돌리든(--limit) 같은 시드를 받고, 기준 시드가 다르면 모든 문장의 시드가 달라진다. 첫 준비 호출은 통계에서
# 빠지므로 시드와 상관없다.
def test_synth_seed_sets_the_seed_of_each_sentence_id_right_before_it(tmp_path, monkeypatch):
    fake_torch, heard = FakeTorch(), []
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    base = _seed_setup(tmp_path, monkeypatch, fake_torch, heard)
    tts_eval.main([*base, "--tag", "s5", "--seed", "5"])
    record = json.loads((tmp_path / "tts_s5_fake_synth.json").read_text(encoding="utf-8"))
    items = record["languages"]["ko"]["items"]
    expected = [(i, tts_eval.sentence_seed(5, i)) for i in (3, 7, 12)]  # sentences()는 ID 순서
    assert record["seed"] == 5
    assert [(item["id"], item["seed"]) for item in items] == expected
    assert heard[1:] == [(f"문장 {i}.", seed) for i, seed in expected]  # heard[0]은 준비 호출

    tts_eval.main([*base, "--tag", "s5_one", "--limit", "1", "--seed", "5"])
    one = json.loads((tmp_path / "tts_s5_one_fake_synth.json").read_text(encoding="utf-8"))
    assert [(item["id"], item["seed"]) for item in one["languages"]["ko"]["items"]] == expected[:1]

    tts_eval.main([*base, "--tag", "s6", "--seed", "6"])
    other = json.loads((tmp_path / "tts_s6_fake_synth.json").read_text(encoding="utf-8"))
    six = {item["id"]: item["seed"] for item in other["languages"]["ko"]["items"]}
    assert all(six[i] != seed for i, seed in expected)


# 기본값(--seed 없음)은 지금까지의 측정과 같다: torch를 import하지도 시드를 걸지도 않는다(sys.modules의
# torch를 None으로 두어 import하면 실패하게 함 — torch가 없는 후보 환경에서도 그대로 돈다). 기록의 seed는
# None이고 문장 항목에는 seed 칸이 없다.
def test_synth_without_seed_does_not_touch_the_random_state(tmp_path, monkeypatch):
    heard: list = []
    monkeypatch.setitem(sys.modules, "torch", None)
    base = _seed_setup(tmp_path, monkeypatch, None, heard)
    tts_eval.main([*base, "--tag", "plain"])
    record = json.loads((tmp_path / "tts_plain_fake_synth.json").read_text(encoding="utf-8"))
    assert record["seed"] is None
    assert all("seed" not in item for item in record["languages"]["ko"]["items"])
    assert [seed for _, seed in heard] == [None] * 4  # 준비 호출 + 세 문장


# 문장 시드 = 기준 시드 × 2^32 + 문장 ID: 같은 (기준, ID)는 늘 같은 값이고, FLEURS ID 범위(0~2999)와 기준
# 1·2·3에서 겹치는 값이 없으며 torch.manual_seed가 받는 범위(0 이상 2^64 미만) 안이다. 범위 밖 값과 정수가
# 아닌 값(bool 포함)은 ValueError, 명령줄의 범위 밖 --seed는 argparse 오류(종료 코드 2)다.
def test_sentence_seeds_are_fixed_by_the_id_and_never_collide():
    assert tts_eval.sentence_seed(1, 7) == tts_eval.sentence_seed(1, 7) == 2**32 + 7
    seeds = {tts_eval.sentence_seed(base, i) for base in (1, 2, 3) for i in range(3000)}
    assert len(seeds) == 3 * 3000 and all(0 <= seed < 2**64 for seed in seeds)
    for base, sentence_id in ((-1, 1), (2**32, 1), (True, 1), (1, -1), (1, 2**32), (1, "7"), (1, 7.0)):
        with pytest.raises(ValueError):
            tts_eval.sentence_seed(base, sentence_id)
    for bad in ("-1", str(2**32), "x"):
        with pytest.raises(SystemExit) as exit_info:
            tts_eval.main(["synth", "--candidate", "fake", "--seed", bad])
        assert exit_info.value.code == 2


# 관문 계산(19절): 비트까지 같음은 dtype·모양·바이트가 모두 같을 때만이다(-0.0과 0.0은 값이 같아도 비트가 달라
# 최대 절대 차 0이지만 같지 않음). 모양이 다르거나 한쪽에 NaN이 있으면 차를 무한대로 본다(관문에서 기각).
def test_array_gap_tells_bit_identical_from_close_and_from_mismatched():
    old = np.array([[0.5, -1.25], [3.0, 0.0]], dtype=np.float32)
    assert tts_eval.array_gap(old, old.copy()) == (True, 0.0)
    near = old.copy()
    near[1, 0] += np.float32(3e-5)
    same, gap = tts_eval.array_gap(old, near)
    assert not same and gap == pytest.approx(3e-5, rel=1e-2)
    negative_zero = old.copy()
    negative_zero[1, 1] = -0.0
    assert tts_eval.array_gap(old, negative_zero) == (False, 0.0)
    assert tts_eval.array_gap(old, old.astype(np.float64)) == (False, 0.0)
    assert tts_eval.array_gap(old, old[:1]) == (False, float("inf"))
    with_nan = old.copy()
    with_nan[0, 0] = np.nan
    assert tts_eval.array_gap(old, with_nan) == (False, float("inf"))
    assert tts_eval.array_gap(np.array([3, 1, 4]), np.array([3, 1, 4])) == (True, 0.0)  # input_ids 같은 정수


# 19절 판정: 정확히 같아야 하는 확인((a) 토큰·(b) clean_text·(d) Kokoro 음소·(e) 적재·합성) 중 하나라도
# 거짓이면 기각, 특징((c) BERT·(d) ALBERT)이 모두 비트까지 같으면 측정 없이 채택안, 가장 큰 차가 허용 폭 1e-4
# 이하(경계 포함)면 관문 2, 넘으면 기각. 확인이 하나도 없으면 판정하지 않는다(ValueError).
def test_the_gate_verdict_follows_the_registered_rule():
    exact = {"tokens": True, "clean_text": True, "load": True}
    same = {"bert": (True, 0.0), "albert": (True, 0.0)}
    assert tts_eval.GATE_TOLERANCE == 1e-4
    assert tts_eval.gate_verdict(exact, same) == "adopt"
    assert tts_eval.gate_verdict(exact, {**same, "bert": (False, 0.0)}) == "gate2"
    assert tts_eval.gate_verdict(exact, {**same, "bert": (False, 1e-4)}) == "gate2"
    assert tts_eval.gate_verdict(exact, {**same, "bert": (False, 1.0001e-4)}) == "reject"
    assert tts_eval.gate_verdict(exact, {**same, "albert": (False, float("inf"))}) == "reject"
    assert tts_eval.gate_verdict({**exact, "clean_text": False}, same) == "reject"
    for empty_exact, empty_features in (({}, same), (exact, {})):
        with pytest.raises(ValueError):
            tts_eval.gate_verdict(empty_exact, empty_features)
