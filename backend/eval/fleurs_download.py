"""Download the FLEURS parquet files the evaluations read (task T2).

Usage, from backend/ (needs only httpx, a main dependency):
    uv run python -m eval.fleurs_download            # validation and test, Korean and English, about 1 GB
    uv run python -m eval.fleurs_download --dry-run  # compare the remote sizes with the local files

Writes data/fleurs/<config>/<split>.parquet (git-ignored). The files are the parquet conversion Hugging Face
serves for google/fleurs (CC-BY-4.0), the same files docs/experiments.md was measured on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from eval.common import DATA, FLEURS_CONFIG

URL = "https://huggingface.co/api/datasets/google/fleurs/parquet/{config}/{split}/0.parquet"
SPLITS = ("validation", "test")


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as out:
            for chunk in response.iter_bytes(1 << 20):
                out.write(chunk)
    partial.replace(target)


def compare(url: str, target: Path) -> str:
    response = httpx.head(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    remote = int(response.headers.get("content-length", 0))
    if not target.exists():
        return f"remote {remote:,} bytes, no local file"
    local = target.stat().st_size
    verdict = "same size" if local == remote else "DIFFERENT size"
    return f"remote {remote:,} bytes, local {local:,} bytes: {verdict}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", choices=list(FLEURS_CONFIG), default=list(FLEURS_CONFIG))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--dry-run", action="store_true", help="compare sizes only, download nothing")
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
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
