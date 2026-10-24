"""uv.lock에 서버가 쓰지 않는 MeloTTS의 부가 도구(gradio·tensorboard)가 다시 들어오지 않는지 지키는 테스트다.
모델·DB 없이 잠금 파일만 읽는다."""

# [파일 역할] backend/uv.lock을 tomllib로 읽어, 서버가 import하지 않는데 MeloTTS가 의존성으로 선언만 해 둔
# gradio(웹 데모 UI)와 tensorboard(학습 기록)가 잠금에 없는지 확인한다. pyproject.toml의 [tool.uv]
# exclude-dependencies가 둘을 빼고, 그 결과 둘만 쓰던 간접 의존성(pillow·brotli·pandas·orjson·grpcio·
# werkzeug 등 15개)도 잠금에서 함께 빠진다.
# [왜 중요한가] 2026-10-24에 서버 의존성(gpu + tts 그룹)의 pip-audit 권고 가운데 pillow 35건·brotli 2건이
# gradio가 끌어온 것이었다(docs/security.md '의존성 취약점' 2026-10-24 덧붙임). 서비스가 쓰는
# melo.api·melo.utils·melo.text는 둘을 import하지 않으므로(고정한 MeloTTS 리비전에서 gradio는 melo/app.py,
# tensorboard는 melo/train.py만 import) 잠금에서 빼는 것이 맞다. MeloTTS 리비전을 올리거나 exclude 설정을
# 지워 둘이 되돌아오면 CI의 이 테스트가 바로 잡는다. `uv lock --check`는 pyproject와 잠금이 어긋났는지만
# 보고, 잠금 안에 어떤 패키지가 있는지는 보지 않는다.
# [주의] 패키지를 빼는 방법은 `exclude-dependencies`여야 한다. override-dependencies에
# `sys_platform == 'never'` 같은 마커를 주는 방식(unidic에 쓴 선례)은 설치만 막고 패키지는 마커를 단 채
# 잠금에 그대로 남아, 잠금을 내보내 검사하는 pip-audit 목록과 이미지 빌드 입력에서 사라지지 않는다
# (2026-10-24에 잠금 사본으로 확인).

import tomllib
from pathlib import Path

import pytest

# BACKEND: 이 파일 기준 backend/ 폴더. 테스트를 어느 작업 폴더에서 돌려도 같은 잠금 파일을 읽게 한다.
BACKEND = Path(__file__).resolve().parents[1]
LOCK_PATH = BACKEND / "uv.lock"

# 잠금에 있으면 안 되는 패키지 → 고정한 MeloTTS에서 그 패키지를 import하는 유일한 파일(서버가 부르지 않음).
EXCLUDED = {
    "gradio": "melo/app.py (웹 데모 UI)",
    "tensorboard": "melo/train.py (학습 기록)",
}


# 잠금 파일의 [[package]] 이름 집합. 잠금은 TOML이라 표준 라이브러리 tomllib로 읽는다(글자 찾기보다 정확하다:
# 주석이나 다른 패키지의 dependencies 안 이름을 패키지로 잘못 세지 않는다).
def locked_package_names() -> set[str]:
    with LOCK_PATH.open("rb") as f:
        lock = tomllib.load(f)
    return {package["name"] for package in lock["package"]}


@pytest.mark.parametrize("name", sorted(EXCLUDED))
def test_the_lock_does_not_carry_melotts_tooling(name: str):
    names = locked_package_names()
    assert name not in names, (
        f"{name}이(가) uv.lock에 돌아왔다. 서버는 {EXCLUDED[name]}를 부르지 않으므로 pyproject.toml의 "
        "[tool.uv] exclude-dependencies에 남아 있어야 한다. 그 설정이 있는데도 들어왔다면 다른 의존성이 새로 "
        "끌어온 것이니 그 경로를 확인한다"
    )
