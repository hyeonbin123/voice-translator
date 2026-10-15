"""명령 하나를 감싸 그동안의 GPU·CPU 사용률을 남기는 도구(eval/usage_log.py)를 GPU 없이 검증한다."""

# [파일 역할] docs/experiments.md 17절 4단계(MeloTTS 단계 측정, 컨테이너 안)에서 측정 도구가 스스로 사용률을
# 남길 수 없어 호스트에서 감싸 재는 도구를, 가짜 기록기로 확인한다. 기록기가 명령이 도는 구간 전체를 덮는지,
# 보고서에 요약·샘플·종료 코드가 남는지, 감싼 명령의 종료 코드를 그대로 돌려주는지, 프로그램 이름은 공개
# 보고서가 아니라 work/ 쪽 파일에만 가는지를 본다.

import json
import sys

from eval import usage_log


# 가짜 기록기: 들어갈 때·나올 때 시각을 적고, 그 사이를 샘플이 있는 것처럼 요약한다.
class FakeLog:
    made: list = []

    def __enter__(self):
        import time

        self.entered = time.time()
        self.samples = [(5.0, 1000.0), (15.0, 1200.0)]
        self.cpu = [10.0, 20.0]
        self.programs = ["some-program"]
        FakeLog.made.append(self)
        return self

    def __exit__(self, *exc):
        import time

        self.exited = time.time()
        return False

    def summary(self):
        return {"samples": len(self.samples), "max_util": 15.0}


def test_the_log_covers_the_command_and_the_exit_code_comes_back(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_log, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(usage_log, "PROJECT", tmp_path)
    (tmp_path / "reports").mkdir()
    marker = tmp_path / "ran.txt"
    # 자식 프로세스가 0.2초 뒤 벽시계 시각을 적고 종료 코드 3으로 끝난다(프로세스 사이라 time.time()으로
    # 견준다).
    script = f"import time, pathlib; time.sleep(0.2); pathlib.Path(r'{marker}').write_text(str(time.time()))"
    command = [sys.executable, "-c", script + "; raise SystemExit(3)"]
    FakeLog.made = []
    code = usage_log.run(command, "t", log_class=FakeLog)
    assert code == 3
    log = FakeLog.made[0]
    ran_at = float(marker.read_text())
    assert log.entered < ran_at < log.exited  # 명령이 도는 동안 기록기가 열려 있었다
    [path] = (tmp_path / "reports").glob("usage_t_*.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["exit_code"] == 3 and record["command"] == command
    assert record["summary"] == {"samples": 2, "max_util": 15.0}
    assert record["samples"] == [[5.0, 1000.0], [15.0, 1200.0]] and record["cpu"] == [10.0, 20.0]
    assert "some-program" not in path.read_text(encoding="utf-8")
    assert (tmp_path / "work" / "gpu_programs_t.txt").read_text(encoding="utf-8") == "some-program"


# 명령을 시작하지 못하면(없는 실행 파일) 예외를 올리되, 그 전에 기록기를 닫고 종료 코드 없는 보고서를 남긴다.
def test_a_command_that_cannot_start_still_leaves_a_report(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(usage_log, "REPORTS", tmp_path)
    monkeypatch.setattr(usage_log, "PROJECT", tmp_path)
    FakeLog.made = []
    with pytest.raises(OSError):
        usage_log.run([str(tmp_path / "no-such-program")], "missing", log_class=FakeLog)
    assert hasattr(FakeLog.made[0], "exited")
    [path] = tmp_path.glob("usage_missing_*.json")
    assert json.loads(path.read_text(encoding="utf-8"))["exit_code"] is None
