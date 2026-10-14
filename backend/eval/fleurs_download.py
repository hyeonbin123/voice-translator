"""평가가 읽는 FLEURS parquet 파일을 내려받는다.

backend/에서 쓰는 법(주 의존성인 httpx만 필요):
    uv run python -m eval.fleurs_download            # validation과 test, 한국어와 영어, 약 1GB
    uv run python -m eval.fleurs_download --dry-run  # 원격 파일 크기를 로컬 파일과 비교

data/fleurs/<config>/<split>.parquet에 쓴다(git이 무시함). 파일은 Hugging Face가 google/fleurs(CC-BY-4.0)에
대해 제공하는 parquet 변환본으로, docs/experiments.md의 측정에 쓴 것과 같은 파일이다.
"""

# [파일 역할] 모든 측정이 읽는 Google FLEURS(한국어·영어, validation·test) parquet 파일을 받는 스크립트.
# [설계 이유] FLEURS는 두 언어의 문장이 서로 번역 관계(FLORES 문장)라 음성 인식·번역·전체 흐름 평가에 함께 쓸
# 수 있다(docs/plan.md 6장). validation은 고르기, test는 고른 설정을 한 번 확인하는 데만 쓴다. Hugging Face가
# 제공하는 parquet 변환본을 직접 받아 datasets 라이브러리 없이 httpx만으로 돈다.
# [연결] README '측정 다시 하기'와 번역 모델 변환 뒤 토큰 확인(eval.mt_convert) 전에 실행한다. 받은 파일은
# stt_eval·mt_eval·e2e_eval·eos_eval·langid_eval 등이 읽는다.

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from eval.common import DATA, FLEURS_CONFIG

# Hugging Face 데이터셋 API의 parquet 변환본 주소(설정·분할마다 첫 파일 0.parquet).
URL = "https://huggingface.co/api/datasets/google/fleurs/parquet/{config}/{split}/0.parquet"
SPLITS = ("validation", "test")


# 1MiB씩 스트리밍으로 받아 .part 임시 파일에 쓰고, 다 받은 뒤에만 원래 이름으로 바꾼다. 중간에 끊기면 반쯤
# 받은 파일이 정상 파일처럼 남지 않는다.
def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as out:
            for chunk in response.iter_bytes(1 << 20):
                out.write(chunk)
    partial.replace(target)


# --dry-run용: HEAD 요청으로 원격 파일 크기를 받아 로컬 파일 크기와 비교한 문장을 돌려준다(받지 않음).
def compare(url: str, target: Path) -> str:
    response = httpx.head(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    remote = int(response.headers.get("content-length", 0))
    if not target.exists():
        return f"remote {remote:,} bytes, no local file"
    local = target.stat().st_size
    verdict = "same size" if local == remote else "DIFFERENT size"
    return f"remote {remote:,} bytes, local {local:,} bytes: {verdict}"


# 언어 × 분할마다 data/fleurs/<설정>/<분할>.parquet을 받는다. 이미 있으면 --force가 없는 한 건너뛴다.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", choices=list(FLEURS_CONFIG), default=list(FLEURS_CONFIG))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--dry-run", action="store_true", help="크기만 비교하고 아무것도 받지 않는다")
    parser.add_argument("--force", action="store_true", help="파일이 이미 있어도 다시 받는다")
    args = parser.parse_args()

    for language in args.languages:
        config = FLEURS_CONFIG[language]
        for split in args.splits:
            url = URL.format(config=config, split=split)
            target = DATA / "fleurs" / config / f"{split}.parquet"
            if args.dry_run:
                print(f"{config}/{split}: {compare(url, target)}")
            elif target.exists() and not args.force:
                print(f"{config}/{split}: {target} exists, skipped (--force downloads it again)")
            else:
                print(f"{config}/{split}: downloading ...", flush=True)
                download(url, target)
                print(f"{config}/{split}: {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
