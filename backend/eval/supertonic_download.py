"""고정한 Supertonic 3 파일을 data/models/supertonic-3에 내려받는다(T78, docs/experiments.md 12절).

원 프로젝트는 보관(archive) 상태(2026-09-09)라, 모든 파일을 그 Hugging Face 저장소의 고정된 한 리비전에서
받고 app/services/supertonic.py에 적힌 SHA-256과 견준다. 파일은 약 400MB이며 커밋하지 않는다. 모델은
OpenRAIL-M 라이선스다: 쓰기 전에 내려받은 LICENSE(Attachment A)와 README의 사용 제한을 읽는다.
backend/에서:

    uv run python -m eval.supertonic_download
"""

# [파일 역할] 한국어 합성 비교용 Supertonic 3 모델 파일을 고정된 리비전으로 내려받고 SHA-256을 확인하는
# 스크립트다(T78, docs/experiments.md 12절). 받은 파일은 data/models/supertonic-3에 두고 커밋하지 않는다.
# [왜 필요한가] 만든 곳이 2026-09-09에 저장소를 보관 처리해 더는 수정·지원이 없다. 그래서 보관처 Hugging Face
# 저장소의 한 리비전과 파일별 해시를 서버 코드(app/services/supertonic.py)에 고정해 두고, 받을 때와 서버가
# 올릴 때 모두 같은 해시인지 확인한다(README 'Supertonic 3 이용 제한 고지').
# [주의] 모델 라이선스는 BigScience OpenRAIL-M이라 이용 제한(Attachment A)이 있다. 서버 기본값은 MeloTTS이고
# Supertonic은 KO_TTS=supertonic일 때만 쓰는 비교용 설정이다.

from __future__ import annotations

import argparse
from pathlib import Path

# 고정 목록(PINNED_FILES: 파일 이름 → SHA-256), 저장소 이름, 리비전, 해시 확인 함수를 서버 코드에서 가져온다.
# 목록을 한 곳에만 두어 내려받기와 서버 적재가 같은 기준을 쓰게 한다.
from app.services.supertonic import PINNED_FILES, REPO, REVISION, verify_files
from eval.common import MODELS


# 고정 파일을 하나씩 확인하고, 없거나 해시가 다를 때만 그 리비전에서 받는다. 마지막에 모든 파일을 다시 확인해
# 하나라도 다르면 verify_files가 예외를 내어 실패로 끝난다.
# huggingface_hub는 이 함수에서만 쓰므로 함수 안에서 가져온다.
def download(out: Path) -> None:
    from huggingface_hub import hf_hub_download

    for name, expected in PINNED_FILES.items():
        path = Path(out) / name
        # 이미 맞는 파일이 있으면 다시 받지 않는다. verify_files는 파일이 없으면 FileNotFoundError, 해시가
        # 다르면 ValueError를 낸다.
        try:
            verify_files(out, {name: expected})
            print(f"{name}: already there")
            continue
        except (FileNotFoundError, ValueError):
            pass
        hf_hub_download(REPO, name, revision=REVISION, local_dir=out)
        print(f"{name}: downloaded to {path}")
    verify_files(out, PINNED_FILES)
    print(f"all {len(PINNED_FILES)} files match their pinned SHA-256 ({REPO}@{REVISION[:12]})")


# 명령줄 진입점. --out으로 받을 폴더를 바꿀 수 있다(기본 data/models/supertonic-3).
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=MODELS / "supertonic-3")
    download(parser.parse_args().out)


if __name__ == "__main__":
    main()
