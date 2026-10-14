"""onnxruntime을 onnxruntime-gpu로 바꾸면 서버의 음성 인식이 달라지는가?(docs/experiments.md 14절)

faster-whisper는 서버의 Silero VAD(STT_VAD_FILTER)를 ONNX Runtime의 CPU 제공자로 돌린다. Supertonic의
GPU 시험은 시험 이미지에서 onnxruntime 패키지를 onnxruntime-gpu로 바꾸므로, 그 이미지에서 서버를 재기
전에 이것으로 그 이미지를 바탕이 된 이미지와 비교한다:

    vad         CPU만: 클립마다, 그리고 1-1절의 말 없는 입력(3초 무음, 3초 작은 잡음)마다 faster-whisper의
                말소리 구간(서버의 VAD 옵션)
    transcribe  GPU: 같은 클립에 대한 서버의 인식(기본 설정)과 1-1절의 무음 확인 네 가지(무음과 잡음,
                한국어와 영어)
    compare     같은 종류의 두 기록: 모든 클립의 구간이나 전사가 같아야 하고, transcribe면 무음 확인 네
                가지가 모두 빈 결과(4/4)여야 한다는 1-1절 규칙. 기록에는 기대한 수의 클립이 있어야
                한다(--expect-clips, 기본은 14절의 validation 클립 60개)

클립은 e2e 클립(data/e2e/<split>/<language>_<id>.wav, eval/e2e_eval.py)이다. pyarrow나 jiwer가 없는 API
이미지 안에서 돈다. backend/(컨테이너에서는 /app)에서:

    python -m eval.ort_swap_check vad --clips ../data/e2e/validation --out ../work/t82/vad_base.json
    python -m eval.ort_swap_check compare ../work/t82/vad_base.json ../work/t82/vad_gpu.json
"""

# [파일 역할] onnxruntime(CPU 전용) 패키지를 onnxruntime-gpu로 바꾼 시험 이미지에서, 서버의 음성 인식 결과가
# 바꾸기 전 이미지와 똑같은지 확인하는 점검 도구다(T82, docs/experiments.md 14절).
# [왜 필요한가] T82는 한국어 합성 Supertonic 3를 ONNX Runtime의 CUDA 제공자로 돌려 보는 시험이었다. 두
# 패키지는 같은 onnxruntime 모듈을 깔아 함께 둘 수 없으므로 시험 이미지는 서버 전체의 onnxruntime을 바꾼다.
# 그런데 faster-whisper의 Silero VAD(STT_VAD_FILTER, T14)도 ONNX Runtime 위에서 돌기 때문에, 합성을 재기 전에
# 이 교체가 인식에 영향을 주지 않는지 먼저 확인해야 했다(14절 '준비 단계 점검'·'교체 점검').
# [흐름] vad(GPU 없이 말소리 구간만) 또는 transcribe(GPU로 서버 기본 설정의 인식)를 두 이미지에서 각각 돌려
# JSON 기록을 남긴다 → compare가 두 기록을 비교해 통과하면 종료 코드 0, 아니면 1을 낸다.
# [주의] API 컨테이너 안에서 돌기 때문에 eval 그룹 패키지(pyarrow, jiwer)를 가져오지 않는다.
# tests/test_ort_swap_check.py가 이 조건을 따로 검사한다.

from __future__ import annotations

import argparse
import io
import json
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path

# numpy: 무음·잡음 테스트 음성을 만드는 데만 쓴다. 무거운 모듈(onnxruntime, faster_whisper, app.*)은 쓰는 함수
# 안에서 가져와, compare만 쓸 때는 모델 관련 패키지가 없어도 돌게 한다.
import numpy as np

# SAMPLE_RATE: Whisper·Silero VAD가 받는 16 kHz.
# NO_SPEECH_INPUTS: T14 무음 확인에 쓴 두 입력(디지털 무음, 작은 백색 잡음).
# EXPECTED_CLIPS: e2e validation 클립 수 60. 14절 규칙이 60/60 일치를 요구하므로 compare의 기본 기대 개수로
# 쓴다(T89).
SAMPLE_RATE = 16_000
NO_SPEECH_INPUTS = ("silence", "noise")
EXPECTED_CLIPS = 60


def no_speech_wav(kind: str, seconds: float = 3.0) -> bytes:
    """1-1절과 같은 말이 없는 음성: 디지털 무음, 또는 작은 백색 잡음(최대 진폭 0.01, 고정 시드).

    16비트 모노 WAV 바이트로 만든다. 입력: 'silence' 또는 'noise', 길이(기본 3초). eval/stt_eval.py에도
    같은 생성이 있지만 그 파일은 eval 그룹 패키지가 필요해 컨테이너에서 가져올 수 없으므로 여기서 다시
    만든다. 잡음은 시드 0으로 고정해 두 이미지에서 똑같은 입력이 들어가게 한다. float 표본에 32767을 곱해
    리틀 엔디언 16비트 정수('<i2')로 바꿔 쓴다.
    """
    count = int(SAMPLE_RATE * seconds)
    samples = np.zeros(count) if kind == "silence" else np.random.default_rng(0).uniform(-0.01, 0.01, count)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def no_speech_names(kind: str) -> set[str]:
    """이 종류의 기록이 담는 무음 입력 이름: VAD는 두 가지, 인식은 그 두 가지를 언어마다.

    vad 기록은 언어와 상관없어 2개, transcribe 기록은 두 언어 × 2개 = 4개(1-1절 무음 확인 4번과 같음).
    compare가 기록이 완전한지 확인할 때 쓴다.
    """
    if kind == "vad":
        return set(NO_SPEECH_INPUTS)
    return {f"{language}_{name}" for language in ("ko", "en") for name in NO_SPEECH_INPUTS}


def clips(folder: Path) -> list[tuple[str, bytes, str]]:
    """폴더의 <언어>_<id>.wav마다 (파일 이름, 음성 바이트, 언어)를 이름순으로 돌려준다.

    언어는 파일 이름의 첫 '_' 앞부분이다. 클립이 하나도 없는 폴더(잘못되었거나 빈 경로)면 ValueError로
    멈춘다. 그런 두 실행이 '아무것도 없음'으로 일치해 통과해 버리지 않게 하려는 것이다(14절 덧붙임).
    """
    found = [
        (path.name, path.read_bytes(), path.name.split("_", 1)[0])
        for path in sorted(Path(folder).glob("*_*.wav"))
    ]
    if not found:
        raise ValueError(f"no clips (<language>_<id>.wav) in {folder}")
    return found


def runtime_info() -> dict:
    """어떤 ONNX Runtime 패키지가 설치돼 있는지, 그 빌드와 쓸 수 있는 실행 제공자.

    설치된 패키지 이름과 버전(onnxruntime, onnxruntime-gpu 중 있는 것), 모듈 버전, 빌드 정보, 실행 제공자
    목록을 기록에 남긴다. 두 기록이 정말 다른 패키지에서 나온 것인지 compare 결과의 packages로 확인할 수
    있게 한다.
    """
    from importlib import metadata

    import onnxruntime

    packages = []
    for name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            packages.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            pass
    return {
        "package": ", ".join(packages),
        "version": onnxruntime.__version__,
        "build_info": onnxruntime.get_build_info(),
        "available_providers": onnxruntime.get_available_providers(),
    }


def speech_timestamps(audio: bytes) -> list[list[int]]:
    """서버 설정에서 faster-whisper의 transcribe가 하는 말소리 구간 검출만 따로 돌린다.

    서버 설정은 vad_filter=True이고 vad_parameters를 주지 않는 것이다. 출력: [시작, 끝] 표본 위치 목록. 인식
    모델 없이 VAD만 돌리므로 GPU 없는 컨테이너에서도 패키지 교체가 VAD 결과를 바꾸는지 볼 수 있다(14절 준비
    단계 ②).
    """
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    samples = decode_audio(io.BytesIO(audio), sampling_rate=SAMPLE_RATE)
    return [[chunk["start"], chunk["end"]] for chunk in get_speech_timestamps(samples, VadOptions())]


# 기록에 작성 시각(UTC)을 붙여 JSON으로 저장한다. 출력 폴더가 없으면 만든다.
def _write(record: dict, out: Path) -> None:
    record["written"] = datetime.now(UTC).isoformat()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written to {out}")


# vad 단계: 모든 클립과 무음 입력 2개의 말소리 구간을 기록한다(CPU만 씀).
def vad(args: argparse.Namespace) -> None:
    record = {
        "kind": "vad",
        "runtime": runtime_info(),
        "clips": {name: speech_timestamps(audio) for name, audio, _ in clips(args.clips)},
        "no_speech": {kind: speech_timestamps(no_speech_wav(kind)) for kind in NO_SPEECH_INPUTS},
    }
    _write(record, args.out)


def server_recognizer():
    """app/services/models.py가 기본 설정으로 만드는 것과 똑같은 서버의 음성 인식기.

    load_models가 WhisperSpeechToText를 만드는 인자를 그대로 옮겼다. 기본 Settings()를 쓰므로 환경 변수를
    주지 않으면 서버 기본 설정의 인식이 된다(14절 '서버 기본 설정의 인식'). GPU면 float16, CPU면 int8로
    계산 형식을 고르는 것도 서버와 같다.
    """
    from app.config import Settings
    from app.services.stt import WhisperSpeechToText

    settings = Settings()
    device = settings.model_device
    return WhisperSpeechToText(
        settings.stt_model,
        vad_filter=settings.stt_vad_filter,
        own_decode=settings.stt_own_decode,
        live_beam_size=settings.live_beam_size,
        live_temperature_fallback=settings.live_temperature_fallback,
        num_workers=settings.stt_num_workers,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )


# transcribe 단계: 서버 인식기로 모든 클립과 두 언어의 무음 입력을 인식해 전사를 기록한다(GPU). 모델 이름과
# 디코딩 옵션도 함께 남겨 두 기록이 같은 설정인지 볼 수 있게 한다.
def transcribe(args: argparse.Namespace) -> None:
    from app.services.interfaces import NoSpeechError

    stt = server_recognizer()

    # 말을 찾지 못하면 인식기가 NoSpeechError를 낸다. 서버가 이를 '말 없음'으로 알리듯 여기서는 빈 문자열로
    # 기록해야 T14 규칙(무음 4번 모두 빈 결과)을 셀 수 있다. 디코딩할 수 없는 음성(UndecodableAudioError)은
    # 서버도 다른 오류(422 "Audio could not be decoded")로 알리는 다른 결과라 빈 결과에 섞지 않고, 기록을 쓰지
    # 않은 채 실행을 멈춘다. 2026-10-13 전에는 두 오류의 상위 클래스(InvalidAudioError)를 잡아 디코딩 실패도
    # 빈 결과로 기록했다(저장된 T82 기록에는 빈 결과인 클립이 없어 영향 없음).
    def text(audio: bytes, language: str) -> str:
        try:  # 말을 찾지 못하면 서버가 알리듯 빈 결과로 둔다
            return stt.transcribe(audio, language).text
        except NoSpeechError:
            return ""

    record = {
        "kind": "transcribe",
        "runtime": runtime_info(),
        "model_name": stt.model_name,
        "options": stt.options,
        "clips": {name: text(audio, language) for name, audio, language in clips(args.clips)},
        "no_speech": {
            f"{language}_{kind}": text(no_speech_wav(kind), language)
            for language in ("ko", "en")
            for kind in NO_SPEECH_INPUTS
        },
    }
    _write(record, args.out)


def compare(before: dict, after: dict, expect_clips: int | None = None) -> dict:
    """두 기록(교체 전·후)을 비교한 결과 dict를 돌려준다.

    pass가 True인 조건: 모든 클립의 결과(구간 또는 전사)가 같다. 무음 입력의 결과도 모두 같다.
    `expect_clips`를 주면 클립 수가 그 값과 같다(같은 불완전한 클립 집합으로 돈 두 실행도 서로 일치하지만,
    관문은 클립 수 60/60을 센다). transcribe 기록이면 교체 뒤 무음 입력이 모두 빈 결과다(1-1절 규칙 4/4).
    비교할 수 없는 기록은 통과/실패가 아니라 ValueError로 멈춘다: 종류가 다름, 클립 목록이 다름, 클립 0개,
    무음 입력이 빠짐. 아무것도 없는 것끼리 일치하는 것은 증거가 아니기 때문이다. 호출 위치: 이 파일의
    main(compare 단계), tests/test_ort_swap_check.py.
    """
    if before["kind"] != after["kind"]:
        raise ValueError(f"the records are of a different kind: {before['kind']} and {after['kind']}")
    if set(before["clips"]) != set(after["clips"]) or set(before["no_speech"]) != set(after["no_speech"]):
        raise ValueError("the two records do not cover the same clips")
    if not before["clips"]:
        raise ValueError("the records hold no clips")
    expected = no_speech_names(before["kind"])
    if set(before["no_speech"]) != expected:
        raise ValueError(
            f"the records' no-speech inputs are {sorted(before['no_speech'])}, not {sorted(expected)}"
        )
    # 결과가 다른 클립·무음 입력의 이름을 모은다. 이름을 남겨 두면 실패했을 때 어느 클립인지 바로 찾을 수
    # 있다.
    differs = sorted(name for name in before["clips"] if before["clips"][name] != after["clips"][name])
    no_speech_differs = sorted(
        name for name in before["no_speech"] if before["no_speech"][name] != after["no_speech"][name]
    )

    # 기록에서 빈 결과(구간 없음 또는 빈 전사)인 무음 입력의 수.
    def empty(record: dict) -> int:
        return sum(1 for value in record["no_speech"].values() if not value)

    result = {
        "kind": before["kind"],
        "packages": [before["runtime"].get("package"), after["runtime"].get("package")],
        "clips": len(before["clips"]),
        "same": len(before["clips"]) - len(differs),
        "differs": differs,
        "no_speech_differs": no_speech_differs,
        "no_speech_empty": {"before": empty(before), "after": empty(after), "of": len(after["no_speech"])},
    }
    # 기본 조건은 '차이 없음'이고, 기대 클립 수와 transcribe의 T14 무음 규칙을 차례로 더 건다.
    passed = not differs and not no_speech_differs
    if expect_clips is not None:
        result["expected_clips"] = expect_clips
        passed = passed and len(before["clips"]) == expect_clips
    if before["kind"] == "transcribe":
        passed = passed and empty(after) == len(after["no_speech"])
    result["pass"] = passed
    return result


# 명령줄 진입점. 하위 명령 vad·transcribe는 --clips 폴더와 --out 파일을 받고, compare는 두 기록 파일과
# --expect-clips(기본 60)를 받는다.
# compare는 결과를 JSON 한 줄로 출력하고 통과 여부를 종료 코드(0/1)로도 알려 스크립트에서 관문으로 쓸 수 있게
# 한다.
def main() -> None:
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("vad", "transcribe"):
        run = phases.add_parser(phase)
        run.add_argument("--clips", type=Path, required=True, help="<language>_<id>.wav 클립이 든 폴더")
        run.add_argument("--out", type=Path, required=True)
    check = phases.add_parser("compare", help="두 기록이 일치하지 않으면 종료 코드 1")
    check.add_argument("before", type=Path)
    check.add_argument("after", type=Path)
    check.add_argument(
        "--expect-clips",
        type=int,
        default=EXPECTED_CLIPS,
        help=f"두 기록이 담아야 할 클립 수(기본 {EXPECTED_CLIPS}, validation 클립)",
    )
    args = parser.parse_args()
    if args.phase == "compare":
        result = compare(
            *(json.loads(path.read_text(encoding="utf-8")) for path in (args.before, args.after)),
            expect_clips=args.expect_clips,
        )
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0 if result["pass"] else 1)
    {"vad": vad, "transcribe": transcribe}[args.phase](args)


if __name__ == "__main__":
    main()
