"""The check that swapping onnxruntime for onnxruntime-gpu leaves speech recognition as it was (T82,
docs/experiments.md 14). No model runs: the timestamps and texts are made up."""

# [파일 역할] eval/ort_swap_check.py(onnxruntime을 onnxruntime-gpu로 바꾼 시험 이미지에서 서버 음성 인식이
# 그대로인지 보는 점검 도구, docs/experiments.md 14절)의 비교 규칙과 입출력을 검증한다. 모델은 돌리지 않고,
# 시각·글자는 지어낸 값이다.
# [왜 중요한가] 서버의 Silero VAD도 ONNX Runtime 위에서 돌기 때문에, 패키지를 바꾼 이미지로 합성을 재기 전에
# 인식이 바뀌지 않았다는 관문(60/60 클립 같음, 무음 입력은 빈 결과)을 통과해야 했다. 이 관문이 '아무것도
# 비교하지 않고 통과'하거나 '덜 모인 클립으로 통과'하는 구멍을 막은 수정(커밋 129e1f7, 18a43c2)을 테스트로
# 고정한다.

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

# BACKEND: 자식 프로세스 import 확인에서 작업 폴더로 쓰는 backend/ 경로.
from eval import ort_swap_check

BACKEND = Path(__file__).resolve().parents[1]


# 점검 도구가 쓰는 기록 JSON과 같은 모양의 사전을 만드는 도우미. kind는 'vad'(말소리 구간 시각) 또는
# 'transcribe'(인식 글자), clips는 클립별 결과, no_speech는 말소리 없는 입력별 결과다.
def record(kind: str, clips: dict, no_speech: dict, package: str = "onnxruntime") -> dict:
    return {"kind": kind, "runtime": {"package": package}, "clips": clips, "no_speech": no_speech}


# transcribe 기록의 무음 입력 네 개(한국어·영어 × 디지털 무음·작은 잡음)가 모두 빈 결과인 상태. 무음 입력은
# docs/experiments.md 1-1절의 점검 입력과 같다.
SILENT = {"ko_silence": "", "ko_noise": "", "en_silence": "", "en_noise": ""}


# 도구가 실제로 도는 API 컨테이너에는 pyarrow·jiwer가 없으므로, 두 모듈 import를 막은 새 프로세스에서도 불러올
# 수 있는지 확인한다.
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


# 통과 사례: 모든 클립의 글자가 같고 무음 입력 넷이 모두 비어 있으면 pass이고, 결과에 같은 클립 수, 무음 빈
# 결과 수(4/4), 비교한 두 패키지 이름이 담기는지 확인한다.
def test_the_same_texts_and_no_speech_found_in_silence_pass():
    before = record("transcribe", {"en_1.wav": "Hello.", "ko_2.wav": "안녕하세요."}, SILENT)
    after = record("transcribe", {"en_1.wav": "Hello.", "ko_2.wav": "안녕하세요."}, SILENT, "onnxruntime-gpu")
    result = ort_swap_check.compare(before, after)
    assert result["pass"] is True
    assert (result["same"], result["clips"], result["differs"]) == (2, 2, [])
    assert result["no_speech_empty"] == {"before": 4, "after": 4, "of": 4}
    assert result["packages"] == ["onnxruntime", "onnxruntime-gpu"]


# 문장부호 하나(. → !)만 달라도 실패하고, 다른 클립 이름을 결과에 적는지 확인한다. 교체가 인식 결과를 '똑같이'
# 유지해야 한다는 엄격한 규칙이다.
def test_one_changed_text_fails_and_is_named():
    before = record("transcribe", {"en_1.wav": "Hello.", "ko_2.wav": "안녕하세요."}, SILENT)
    after = record("transcribe", {"en_1.wav": "Hello!", "ko_2.wav": "안녕하세요."}, SILENT)
    result = ort_swap_check.compare(before, after)
    assert result["pass"] is False and result["differs"] == ["en_1.wav"] and result["same"] == 1


# 교체 뒤 잡음 입력에서 "Thank you." 같은 문장을 지어내면 실패한다. VAD를 켜기 전 서버가 무음에서 문장을
# 지어냈던 문제(README 설계 판단)가 교체로 되살아나지 않는지 보는 규칙이다.
def test_speech_made_up_from_silence_after_the_swap_fails():
    # T14's silence check: all four no-speech inputs must come back empty (4/4).
    after = record("transcribe", {"en_1.wav": "Hello."}, {**SILENT, "en_noise": "Thank you."})
    result = ort_swap_check.compare(record("transcribe", {"en_1.wav": "Hello."}, SILENT), after)
    assert result["pass"] is False and result["no_speech_empty"]["after"] == 3


# vad 기록은 말소리 구간 시각(표본 단위)이 정확히 같아야 한다. 한 구간의 시작이 512표본 밀리거나, 잡음
# 입력에서 새 구간이 잡히면 실패한다.
def test_vad_timestamps_must_match_exactly_including_the_no_speech_inputs():
    clips = {"en_1.wav": [[1600, 32000]], "ko_2.wav": [[0, 16000], [20000, 48000]]}
    quiet = {"silence": [], "noise": []}
    assert ort_swap_check.compare(record("vad", clips, quiet), record("vad", clips, quiet))["pass"] is True
    moved = {**clips, "ko_2.wav": [[0, 16000], [20512, 48000]]}
    result = ort_swap_check.compare(record("vad", clips, quiet), record("vad", moved, quiet))
    assert result["pass"] is False and result["differs"] == ["ko_2.wav"]
    noisy = {"silence": [], "noise": [[0, 512]]}
    assert ort_swap_check.compare(record("vad", clips, quiet), record("vad", clips, noisy))["pass"] is False


# 종류가 다른 기록이나 클립 목록이 다른 기록은 비교 자체를 거부한다(ValueError). 서로 다른 것을 견준 '통과'를
# 막는다.
def test_records_of_different_kinds_or_clips_are_not_compared():
    with pytest.raises(ValueError, match="kind"):
        ort_swap_check.compare(record("vad", {}, {}), record("transcribe", {}, {}))
    with pytest.raises(ValueError, match="clips"):
        ort_swap_check.compare(
            record("vad", {"en_1.wav": []}, {}), record("vad", {"en_1.wav": [], "en_2.wav": []}, {})
        )


# 두 기록 모두 클립이 0개면(폴더를 잘못 주거나 빈 폴더) '0/0 같음'으로 통과하지 않고 거부되는지 두 종류 모두
# 확인한다(커밋 129e1f7).
def test_records_without_clips_are_not_compared():
    # A wrong or empty clip folder in both runs must not pass the 60/60 gate with 0/0 (T84).
    for kind, quiet in (("vad", {"silence": [], "noise": []}), ("transcribe", SILENT)):
        with pytest.raises(ValueError, match="no clips"):
            ort_swap_check.compare(record(kind, {}, quiet), record(kind, {}, quiet))


# 두 기록이 똑같이 무음 입력 하나를 빠뜨려도 비교를 거부한다. 무음 입력은 종류별로 정해진 전부(transcribe 넷,
# vad 둘)가 있어야 한다.
def test_records_missing_a_no_speech_input_are_not_compared():
    # The no-speech inputs are T14's: both kinds must hold all of them, even when both lack the same one.
    clips = {"en_1.wav": "Hello."}
    three = {key: value for key, value in SILENT.items() if key != "en_noise"}
    with pytest.raises(ValueError, match="no-speech"):
        ort_swap_check.compare(record("transcribe", clips, three), record("transcribe", clips, three))
    timestamps = {"en_1.wav": [[0, 16000]]}
    with pytest.raises(ValueError, match="no-speech"):
        ort_swap_check.compare(
            record("vad", timestamps, {"silence": []}), record("vad", timestamps, {"silence": []})
        )


# 같은 불완전한 클립 집합(59개)으로 두 번 돌리면 서로는 같지만, 관문은 validation 60클립 모두(60/60)이므로
# expect_clips=60이면 실패해야 한다(커밋 18a43c2). 기대 수를 59로 주면 통과한다.
def test_records_holding_fewer_clips_than_expected_fail():
    # Two runs on the same incomplete clip set agree, but the gate is 60/60 (docs/experiments.md 14) (T89).
    clips = {f"en_{i}.wav": "Hello." for i in range(59)}
    before, after = record("transcribe", clips, SILENT), record("transcribe", clips, SILENT)
    result = ort_swap_check.compare(before, after, expect_clips=60)
    assert result["pass"] is False
    assert (result["clips"], result["same"], result["expected_clips"]) == (59, 59, 60)
    assert ort_swap_check.compare(before, after, expect_clips=59)["pass"] is True


# 명령줄 compare가 기본으로 60클립을 기대해 59클립 기록에는 종료 코드 1을, --expect-clips 59를 주면 0을 내는지
# 확인한다. 기록 파일은 tmp_path에 쓰고, sys.argv를 monkeypatch로 바꿔 main()을 직접 부른다. 출력 JSON도
# capsys로 읽어 기대 클립 수가 적혔는지 본다.
@pytest.mark.parametrize(("extra", "status"), [([], 1), (["--expect-clips", "59"], 0)])
def test_the_compare_command_expects_the_60_validation_clips(tmp_path, monkeypatch, capsys, extra, status):
    clips = {f"ko_{i}.wav": [[0, 16000]] for i in range(59)}
    paths = []
    for name in ("before", "after"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(record("vad", clips, {"silence": [], "noise": []})), encoding="utf-8")
        paths.append(str(path))
    monkeypatch.setattr(sys, "argv", ["ort_swap_check", "compare", *paths, *extra])
    with pytest.raises(SystemExit) as exited:
        ort_swap_check.main()
    assert exited.value.code == status
    printed = json.loads(capsys.readouterr().out)
    assert printed["expected_clips"] == (60 if not extra else 59) and printed["clips"] == 59


# <언어>_<id>.wav 파일이 없는 폴더(다른 파일만 있거나 아예 없는 경로)를 주면 기록을 만들기 전에 멈추는지
# 확인한다. 잘못된 경로로 만든 빈 기록 두 개가 서로 '같음'이 되는 일을 처음부터 막는다.
@pytest.mark.parametrize("folder", ["empty", "missing"])
def test_a_folder_without_clips_stops_the_run(tmp_path, folder):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="no clips"):
        ort_swap_check.clips(tmp_path / folder)


# 클립 목록은 파일 이름 순서이고, 언어는 파일 이름의 앞부분(ko_·en_)에서 얻으며, 형식이 다른 파일은 무시한다.
def test_clips_take_the_language_from_the_file_name(tmp_path):
    (tmp_path / "ko_12.wav").write_bytes(b"k")
    (tmp_path / "en_3.wav").write_bytes(b"e")
    (tmp_path / "notes.txt").write_text("x")
    assert ort_swap_check.clips(tmp_path) == [("en_3.wav", b"e", "en"), ("ko_12.wav", b"k", "ko")]


# 도구가 직접 만드는 무음·잡음 WAV가 음성 인식 평가(eval/stt_eval.py)의 것과 바이트까지 같은지 확인한다.
# stt_eval은 eval 그룹이 필요해 이 도구가 가져다 쓰지 못하고 같은 코드를 복사해 두었으므로, 둘이 어긋나지 않게
# 지킨다. eval 그룹이 없으면 건너뛴다.
def test_the_no_speech_inputs_are_t14s():
    stt_eval = pytest.importorskip("eval.stt_eval", reason="needs the eval group (jiwer, pyarrow)")
    for kind in ("silence", "noise"):
        assert ort_swap_check.no_speech_wav(kind) == stt_eval.no_speech_wav(kind)


# transcribe 단계의 가짜 인식기. 'quiet' 클립과 무음 입력(WAV)은 말 없음(NoSpeechError), 'broken' 클립은
# 디코딩 불가(UndecodableAudioError), 그 밖에는 바이트를 그대로 글자로 돌려준다.
class FakeRecognizer:
    model_name = "fake"
    options = {"beam_size": 5}

    def transcribe(self, audio: bytes, language: str):
        from app.services.interfaces import NoSpeechError, UndecodableAudioError

        if audio == b"broken":
            raise UndecodableAudioError("the audio could not be decoded")
        if audio == b"quiet" or audio.startswith(b"RIFF"):
            raise NoSpeechError("no speech was recognized")
        return Namespace(text=audio.decode())


# 말을 찾지 못한 클립과 무음 입력은 빈 결과로 기록된다(서버의 '말 없음'과 같음, T14 규칙을 세는 데 필요).
def test_transcribe_records_no_speech_as_an_empty_result(tmp_path, monkeypatch):
    (tmp_path / "en_1.wav").write_bytes(b"Hello.")
    (tmp_path / "ko_2.wav").write_bytes(b"quiet")
    monkeypatch.setattr(ort_swap_check, "server_recognizer", FakeRecognizer)
    monkeypatch.setattr(ort_swap_check, "runtime_info", lambda: {"package": "onnxruntime"})
    out = tmp_path / "stt.json"
    ort_swap_check.transcribe(Namespace(clips=tmp_path, out=out))
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["clips"] == {"en_1.wav": "Hello.", "ko_2.wav": ""}
    assert written["no_speech"] == SILENT


# 디코딩할 수 없는 클립은 서버가 '말 없음'과 다른 오류로 알리는 결과라 빈 결과로 기록하지 않고 실행을
# 멈춘다(기록도 쓰지 않음). 예전에는 두 오류의 상위 클래스를 잡아 빈 결과로 기록했고, 두 실행이 같은 클립을
# 똑같이 못 읽거나 한쪽은 못 읽고 한쪽은 말을 못 찾아도 '같음'이 되었다.
def test_transcribe_stops_on_audio_that_cannot_be_decoded(tmp_path, monkeypatch):
    from app.services.interfaces import UndecodableAudioError

    (tmp_path / "en_1.wav").write_bytes(b"broken")
    monkeypatch.setattr(ort_swap_check, "server_recognizer", FakeRecognizer)
    monkeypatch.setattr(ort_swap_check, "runtime_info", lambda: {"package": "onnxruntime"})
    out = tmp_path / "stt.json"
    with pytest.raises(UndecodableAudioError):
        ort_swap_check.transcribe(Namespace(clips=tmp_path, out=out))
    assert not out.exists()


# vad 단계가 클립마다 말소리 구간을, 무음 입력 둘에도 구간을 기록하고, 설치된 ONNX Runtime 정보(runtime)를
# 함께 남기는지 확인한다. 실제 VAD와 런타임 조회는 monkeypatch로 가짜 함수로 바꿔 모델 없이 돈다.
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
