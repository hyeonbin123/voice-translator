"""uv.lock에 적힌 서버 환경(tts 그룹)의 torch가 2.6 이상인지 지키는 테스트다. 모델·DB 없이 잠금 파일만
읽는다."""

# [파일 역할] backend/uv.lock을 tomllib로 읽어, 음성 합성(tts 그룹)이 쓰는 torch가 2.6 아래로 내려가지
# 않았는지 확인한다.
# [왜 중요한가] 서버는 pickle 형식의 체크포인트를 둘 읽는다: MeloTTS-Korean의 checkpoint.pth(고정 MeloTTS의
# melo/download_utils.py가 `torch.load(ckpt_path, map_location=device)`로 읽음)와 한국어 BERT
# kykim/bert-kor-base의 pytorch_model.bin(MeloTTS가 고정한 transformers 4.27.4가
# `torch.load(checkpoint_file, map_location="cpu")`로 읽음). 둘 다 weights_only 인자를 주지 않으므로,
# 체크포인트 안의 임의 코드 실행을 막는 것은 torch 2.6부터 기본값이 된 `weights_only=True`다(Kokoro 0.9.4는
# 직접 weights_only=True를 준다). docs/security.md가 transformers 4.27.4의 역직렬화 권고를 받아들인 근거가 이
# 기본값이라, torch가 2.6 아래로 내려가면 그 근거가 사라진다. 잠금을 읽는 이유: 서버 이미지와 CI는
# `--locked`로 이 파일 그대로 설치하므로 잠금의 판이 곧 서버의 판이다.
# [주의] 잠금에 torch가 둘 있다(tts 그룹 2.11.0+cu128, eval 그룹 2.14.0). 둘 다 검사하되, tts 그룹이 torch를
# 가리키는지도 확인해 서버 환경이 검사 대상에서 빠지지 않게 한다. 판 문자열의 `+cu128` 같은 로컬 표시는 떼고
# 비교한다.

import tomllib
from pathlib import Path

# BACKEND: 이 파일 기준 backend/ 폴더. 테스트를 어느 작업 폴더에서 돌려도 같은 잠금 파일을 읽게 한다.
BACKEND = Path(__file__).resolve().parents[1]
LOCK_PATH = BACKEND / "uv.lock"
# torch.load의 weights_only 기본값이 True로 바뀐 첫 판.
WEIGHTS_ONLY_DEFAULT_SINCE = (2, 6)


def load_lock() -> dict:
    with LOCK_PATH.open("rb") as f:
        return tomllib.load(f)


# "2.11.0+cu128" → (2, 11). 로컬 표시(+ 뒤)는 떼고 앞 두 자리만 정수로 비교한다.
def major_minor(version: str) -> tuple[int, int]:
    numbers = version.split("+")[0].split(".")
    return int(numbers[0]), int(numbers[1])


def test_every_locked_torch_defaults_to_weights_only_loading():
    versions = {package["version"] for package in load_lock()["package"] if package["name"] == "torch"}
    assert versions, "uv.lock에 torch가 없다. tts 그룹이 torch를 잃었는지 확인한다"
    too_old = sorted(v for v in versions if major_minor(v) < WEIGHTS_ONLY_DEFAULT_SINCE)
    assert not too_old, (
        f"uv.lock의 torch {too_old}는 2.6보다 낮아 torch.load의 기본값이 weights_only=True가 아니다. MeloTTS "
        "checkpoint.pth와 BERT pytorch_model.bin(pickle)의 역직렬화 방어가 사라지므로(docs/security.md "
        "'의존성 취약점' transformers 행) 내리지 않는다"
    )


# 서버 환경인 tts 그룹이 torch를 가리키는지. 잠금의 프로젝트 항목은 uv 판에 따라 dependency-groups 또는
# dev-dependencies 키로 그룹을 적으므로 둘 다 본다.
def test_the_tts_group_still_pins_torch():
    project = next(p for p in load_lock()["package"] if p["name"] == "voice-translator-backend")
    groups = project.get("dependency-groups") or project.get("dev-dependencies") or {}
    names = {dependency["name"] for dependency in groups.get("tts", [])}
    assert "torch" in names, "tts 그룹에 torch가 없다. 서버 이미지의 torch 판을 잠금이 정하지 않게 된다"
