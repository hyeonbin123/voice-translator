"""명령 하나를 실행하는 동안 이 PC의 GPU·CPU 사용률을 0.5초마다 기록한다.

  uv run --no-sync python -m eval.usage_log --tag t95_dev_melo -- docker run --rm --gpus all ...

docs/experiments.md 17절의 측정 환경 규칙(지연 측정 중 0.5초마다 GPU·CPU 사용률을 남김)을, 측정 도구 자신이
기록할 수 없는 단계에 맞추려고 둔다. 17절 4단계의 MeloTTS 단계 측정(`eval.tts_eval melo-stages`)은 API 이미지
안에서 도는데, 그 안에는 nvidia-ml-py·psutil이 없어 stream_eval처럼 스스로 기록하지 못한다. 그래서 호스트에서
그 명령(docker run)을 감싸 같은 기록기(`eval.stream_eval.GpuLog`)로 잰다.
"""

# [파일 역할] 감싼 명령이 도는 동안의 사용률을 reports/usage_<태그>_<UTC 시각>.json에 남긴다: 명령, 종료 코드,
# 걸린 시간, 요약(샘플 수, GPU 사용률 평균·p90·최대, 메모리 최대, CPU 평균), 0.5초 샘플 전체. GPU를 쓰던
# 프로그램 이름은 공개 보고서에 넣지 않고 커밋하지 않는 work/gpu_programs_<태그>.txt에만 쓴다(stream_eval
# service와 같은 규칙). 같은 까닭으로 명령 속 저장소 절대 경로(측정 스크립트가 `docker run -v
# <저장소>/backend/eval:/app/eval`로 넘기는 것)는 `<project>`로 바꿔 적는다(hide_project). 종료 코드는
# 감싼 명령의 것을 그대로 돌려줘, 실행 스크립트가 실패를 알아챌 수 있게 한다.
# [왜 호스트에서 재나] 컨테이너 안에서 본 CPU는 Docker 가상 머신의 것이라 PC 전체의 부하(다른 프로그램)를
# 보지 못한다. 호스트의 기록기가 stream_eval과 같은 기준으로 PC 전체를 본다.

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import PurePath

from eval.common import PROJECT, REPORTS

# 보고서에 저장소 경로 대신 적는 자리표시. 보고서는 공개 저장소에 커밋하므로 이 PC의 폴더 구조(드라이브·사용자
# 폴더 이름)를 남기지 않고, 어느 폴더를 연결했는지는 저장소 기준 상대 위치(`<project>/backend/eval`)로 읽힌다.
PLACEHOLDER = "<project>"


def hide_project(command: list[str], project: PurePath | None = None) -> list[str]:
    """명령 인자 속 저장소 루트 경로(project, 기본 PROJECT)를 PLACEHOLDER로 바꾼 사본을 돌려준다.

    바꾸는 꼴은 셋이다: 운영체제 표기(Windows는 역슬래시), 슬래시 표기(`D:/…`, 측정 스크립트가 docker에
    넘기는 꼴), 드라이브 경로면 Git Bash(MSYS)가 넘기는 `/d/…` 꼴. Windows 경로는 대소문자를 가리지 않으므로
    대소문자를 무시하고 찾는다(Linux에서도 무시하지만, 대소문자만 다른 다른 폴더를 가리는 것은 보고서 칸이라
    해가 없다). 경로 바로 뒤가 이름에 쓰이는 글자(영문·숫자·`_`·`.`·`-`)이면 저장소 이름으로 시작할 뿐인 다른
    폴더(`voice-translator-old` 등)라 바꾸지 않는다. 경로 뒤의 나머지(`/backend/eval:/app/eval`)와 다른 인자는
    그대로 둔다. 실제 폴더를 읽지 않아 PurePath로도 시험할 수 있다(tests/test_usage_log.py).
    """
    project = PROJECT if project is None else project
    posix = project.as_posix()
    forms = {str(project), posix}
    if len(posix) >= 2 and posix[1] == ":" and posix[0].isalpha():
        forms.add(f"/{posix[0].lower()}{posix[2:]}")
    # 긴 꼴부터 맞춰 본다(한 꼴이 다른 꼴의 앞부분일 때 짧은 쪽이 먼저 잡혀 나머지가 남지 않게).
    alternatives = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
    pattern = re.compile(f"(?:{alternatives})(?![\\w.-])", re.IGNORECASE)
    return [pattern.sub(lambda _match: PLACEHOLDER, argument) for argument in command]


def run(command: list[str], tag: str, log_class=None) -> int:
    """command를 실행하는 동안 log_class(기본 stream_eval.GpuLog)로 기록하고, 보고서를 쓴 뒤 종료 코드를
    돌려준다.

    log_class를 받는 것은 테스트가 GPU 없이 가짜 기록기를 넣기 위해서다. 명령이 예외로 시작하지 못해도(파일
    없음 등) 기록기를 닫고 보고서를 남긴 뒤 예외를 올린다.
    """
    if log_class is None:
        from eval.stream_eval import GpuLog

        log_class = GpuLog
    start = time.perf_counter()
    code: int | None = None
    log = None  # 기록기가 시작하지 못하면(예: GPU 없음) None으로 남는다
    try:
        with log_class() as log:
            code = subprocess.call(command)
    finally:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        record = {
            "tag": tag,
            "command": hide_project(command),  # 저장소 경로는 자리표시로(위 hide_project)
            "exit_code": code,
            "seconds": round(time.perf_counter() - start, 1),
            "summary": log.summary() if log is not None else None,
            "samples": [list(sample) for sample in log.samples] if log is not None else [],
            "cpu": list(log.cpu) if log is not None else [],
        }
        path = REPORTS / f"usage_{tag}_{stamp}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        if log is not None:
            programs = PROJECT / "work" / f"gpu_programs_{tag}.txt"
            programs.parent.mkdir(parents=True, exist_ok=True)
            programs.write_text("\n".join(log.programs), encoding="utf-8")
        print(f"written to {path}", flush=True)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="명령 하나를 실행하는 동안 GPU·CPU 사용률을 0.5초마다 기록한다(docs/experiments.md 17절)",
    )
    parser.add_argument("--tag", required=True, help="보고서 이름 reports/usage_<tag>_<시각>.json")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- 뒤에 실행할 명령")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("실행할 명령을 -- 뒤에 준다")
    sys.exit(run(command, args.tag))


if __name__ == "__main__":
    main()
