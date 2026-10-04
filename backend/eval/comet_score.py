"""COMET-22 segment scores for the re-analysis of recorded decisions (docs/experiments.md 13, task T80).

This script runs in its own environment, not backend/.venv: unbabel-comet 2.2.7 needs transformers<5 and
numpy<2 while the eval group has transformers 5 and numpy 2. It imports only the standard library at the top
so ruff and the backend tests can read it. From backend/:

    uv venv ../work/comet-venv --python 3.11
    set PY=../work/comet-venv/Scripts/python.exe
    uv pip sync --python %PY% --torch-backend cu128 eval/comet-requirements.txt
    %PY% eval/comet_score.py download
    %PY% eval/comet_score.py score --input IN.jsonl --output OUT.json --device cuda --precision fp16

The input is what `python -m eval.reanalysis comet-input` writes: one JSON object per line with `key`, `src`,
`mt` and `ref`. The output maps each key to its score and records the model revision, the SHA-256 of the
checkpoint as hashed when it was loaded for scoring, device, precision and package versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

MODEL = "Unbabel/wmt22-comet-da"  # Apache-2.0
REVISION = "2760a223ac957f30acfb18c8aa649b01cf1d75f2"
CHECKPOINT = "checkpoints/model.ckpt"
CHECKPOINT_SHA256 = "e213091cde220f97b89f8bdfa750c458cfea741ad62affb455b59900210ff2af"
# The checkpoint holds the encoder weights; COMET still reads the tokenizer and config of the encoder it
# was trained on (`pretrained_model: xlm-roberta-large` in hparams.yaml). They are loaded from this pinned
# snapshot instead of whatever "main" is when the script runs.
ENCODER = "FacebookAI/xlm-roberta-large"  # MIT
ENCODER_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
ENCODER_FILES = ["config.json", "sentencepiece.bpe.model", "tokenizer.json", "tokenizer_config.json"]
BATCH_SIZE = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshots(offline: bool) -> tuple[Path, Path]:
    """Local folders of the pinned COMET model and encoder files (downloaded unless offline)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.file_download import are_symlinks_supported

    # Without symlink rights (Windows, no developer mode) the parallel download threads race on this check:
    # the first marks the cache as symlink-capable before testing it, and another then fails with
    # WinError 1314. Settling it once here, before any download thread starts, avoids that.
    are_symlinks_supported()
    model = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=offline))
    encoder = Path(
        snapshot_download(
            ENCODER, revision=ENCODER_REVISION, allow_patterns=ENCODER_FILES, local_files_only=offline
        )
    )
    return model, encoder


def check_checkpoint(model_dir: Path) -> str:
    """SHA-256 of the checkpoint file in `model_dir`; stops unless it is the pinned one."""
    actual = sha256(model_dir / CHECKPOINT)
    if actual != CHECKPOINT_SHA256:
        raise SystemExit(f"{CHECKPOINT} SHA-256 {actual} is not the pinned {CHECKPOINT_SHA256}")
    return actual


def download() -> None:
    model, encoder = snapshots(offline=False)
    check_checkpoint(model)
    missing = [name for name in ENCODER_FILES if not (encoder / name).is_file()]
    if missing:
        raise SystemExit(f"encoder files missing: {missing}")
    print(f"ok: {model} (SHA-256 checked), {encoder}")


def load(device: str, precision: str):
    """The pinned COMET-22 model in eval mode on `device`, loaded from local files only, and the SHA-256 of
    the checkpoint file it was loaded from. The file is hashed here, before the model is built, so a cache
    that changed since `download` is refused and the scores' metadata records what was actually read."""
    import torch
    import yaml
    from comet.models import str2model

    model_dir, encoder_dir = snapshots(offline=True)
    checkpoint_sha256 = check_checkpoint(model_dir)
    hparams = yaml.safe_load((model_dir / "hparams.yaml").read_text(encoding="utf-8"))
    if (
        hparams.get("layer_transformation") == "sparsemax_patch"
    ):  # comet's own load_from_checkpoint workaround
        raise SystemExit("unexpected layer_transformation sparsemax_patch; use comet.load_from_checkpoint")
    model = str2model[hparams["class_identifier"]].load_from_checkpoint(
        checkpoint_path=model_dir / CHECKPOINT,
        load_pretrained_weights=False,
        map_location=torch.device("cpu"),
        strict=False,
        local_files_only=True,
        pretrained_model=str(encoder_dir),
    )
    model.eval()
    model.to(device)
    if precision == "fp16":
        model.half()
    return model, checkpoint_sha256


def score(model, samples: list[dict], device: str, batch_size: int = BATCH_SIZE) -> list[float]:
    """Segment scores in input order. Batches are formed in source-length order, as comet's predict does."""
    import torch

    order = sorted(range(len(samples)), key=lambda index: len(samples[index]["src"]))
    scores = [0.0] * len(samples)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            picked = order[start : start + batch_size]
            batch = [{key: samples[index][key] for key in ("src", "mt", "ref")} for index in picked]
            inputs = {name: tensor.to(device) for name, tensor in model.prepare_for_inference(batch).items()}
            values = model(**inputs).score.float().cpu().tolist()
            for index, value in zip(picked, values, strict=True):
                scores[index] = value
    return scores


def run(args: argparse.Namespace) -> None:
    import comet
    import torch
    import transformers

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.device == "cpu" and args.precision == "fp16":
        raise SystemExit("fp16 is for the GPU; score on the CPU in fp32")
    os.environ["HF_HUB_OFFLINE"] = "1"
    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in lines if line.strip()]
    keys = [sample["key"] for sample in samples]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate keys in the input")
    if args.limit:
        samples = samples[: args.limit]

    started = time.perf_counter()
    model, checkpoint_sha256 = load(args.device, args.precision)  # load_s includes hashing the checkpoint
    loaded = time.perf_counter()
    values = score(model, samples, args.device, args.batch_size)
    finished = time.perf_counter()
    broken = [
        sample["key"] for sample, value in zip(samples, values, strict=True) if not math.isfinite(value)
    ]
    if broken:
        raise SystemExit(f"{len(broken)} scores are not finite numbers (first {broken[0]}); nothing written")

    meta = {
        "model": MODEL,
        "revision": REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "encoder": ENCODER,
        "encoder_revision": ENCODER_REVISION,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "segments": len(samples),
        "load_s": round(loaded - started, 1),
        "score_s": round(finished - loaded, 1),
        "versions": {
            "unbabel-comet": comet.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "input": str(args.input),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    result = {
        "meta": meta,
        "scores": {sample["key"]: value for sample, value in zip(samples, values, strict=True)},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(samples)} segments in {meta['score_s']} s -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="fetch the pinned model and encoder files and check the checkpoint")
    scoring = commands.add_parser("score", help="score segments offline")
    scoring.add_argument("--input", required=True)
    scoring.add_argument("--output", required=True)
    scoring.add_argument("--device", choices=["cuda", "cpu"], required=True)
    scoring.add_argument("--precision", choices=["fp16", "fp32"], required=True)
    scoring.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    scoring.add_argument("--limit", type=int, help="score only the first N segments (checks)")
    args = parser.parse_args()
    if args.command == "download":
        download()
    else:
        run(args)


if __name__ == "__main__":
    main()
