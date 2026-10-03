"""Download the pinned Supertonic 3 files into data/models/supertonic-3 (T78, docs/experiments.md 12).

The upstream project is archived (2026-09-09), so every file is fetched at one fixed revision of the archive's
Hugging Face repository and checked against its SHA-256 in app/services/supertonic.py. The files are about
400 MB and are not committed. The model is under OpenRAIL-M: read the use restrictions in the downloaded
LICENSE (Attachment A) and in the README before using it. From backend/:

    uv run python -m eval.supertonic_download
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.services.supertonic import PINNED_FILES, REPO, REVISION, verify_files
from eval.common import MODELS


def download(out: Path) -> None:
    from huggingface_hub import hf_hub_download

    for name, expected in PINNED_FILES.items():
        path = Path(out) / name
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=MODELS / "supertonic-3")
    download(parser.parse_args().out)


if __name__ == "__main__":
    main()
