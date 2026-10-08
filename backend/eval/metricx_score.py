"""MetricX-24 reference-free (QE) segment scores for the contamination check of T83 (docs/experiments.md 15).

MetricX-24-Hybrid-Large (google/metricx-24-hybrid-large-v2p6, Apache-2.0) is an mT5-Large regression model.
Its weights are a 4.9 GB pickled `pytorch_model.bin`, read here with `torch.load(weights_only=True)` after
the file's SHA-256 is checked. It runs in fp32: mT5 overflows in fp16, and the RTX 2080 Ti has no bf16.

This script runs in work/comet-venv (transformers 4.57, torch 2.11 + cu128; see eval/comet_score.py), not in
backend/.venv, and imports only the standard library at the top so ruff and the backend tests can read it.
The forward pass follows google-research/metricx at fc4978e (metricx24/models.py and predict.py, Apache-2.0):
the input is "source: <src> candidate: <mt>" for QE, tokenized with the mT5 tokenizer, at most 1536 tokens,
with the final end-of-sequence token removed; the decoder gets one step of token id 0, and the score is the
logit of token 250089 (<extra_id_10>) at that step, clipped to [0, 25]. MetricX's own MT5ForRegression has
the same layers as transformers' MT5ForConditionalGeneration, so the weights load into that class. Scores
are error scores: lower is better. From backend/:

    set PY=../work/comet-venv/Scripts/python.exe
    %PY% eval/metricx_score.py download
    %PY% eval/metricx_score.py score --input IN.jsonl --output OUT.json --device cpu

The input has one JSON object per line with `key`, `src` and `mt`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

MODEL = "google/metricx-24-hybrid-large-v2p6"  # Apache-2.0
REVISION = "51e875ba5c525c81627cfd135ee10f43c87dce00"
WEIGHTS = "pytorch_model.bin"
WEIGHTS_SHA256 = "f584e231b25a20d7766f03fa464d0b990e235b8514a3de2034002132c04a2878"
TOKENIZER = "google/mt5-large"  # Apache-2.0; MetricX-24 uses the mT5 tokenizer
TOKENIZER_REVISION = "50b7223e98fcd124b0cabb1ec81bc6324c7df107"
TOKENIZER_FILES = ["spiece.model", "special_tokens_map.json", "tokenizer_config.json"]
SPIECE_SHA256 = "ef78f86560d809067d12bac6c09f19a462cb3af3f54d2b8acbba26e1433125d6"
MAX_INPUT_LENGTH = 1536
SCORE_TOKEN = 250089  # <extra_id_10>
CODE = "google-research/metricx@fc4978eb064670f7cc33e93ea4f52d38396b8ae6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qe_input(source: str, translation: str) -> str:
    """MetricX-24's reference-free input (predict.py with --qe)."""
    return "source: " + source + " candidate: " + translation


def snapshots(offline: bool) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download
    from huggingface_hub.file_download import are_symlinks_supported

    are_symlinks_supported()
    # One download thread: without symlink rights (Windows, no developer mode) parallel threads race on the
    # symlink check of each new repository folder, and one then fails with WinError 1314 (comet_score.py).
    model = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=offline, max_workers=1))
    tokenizer = Path(
        snapshot_download(
            TOKENIZER,
            revision=TOKENIZER_REVISION,
            allow_patterns=TOKENIZER_FILES,
            local_files_only=offline,
            max_workers=1,
        )
    )
    return model, tokenizer


def check_files(model_dir: Path, tokenizer_dir: Path) -> str:
    """SHA-256 of the weights; stops unless the weights and the tokenizer model are the pinned files."""
    actual = sha256(model_dir / WEIGHTS)
    if actual != WEIGHTS_SHA256:
        raise SystemExit(f"{WEIGHTS} SHA-256 {actual} is not the pinned {WEIGHTS_SHA256}")
    spiece = sha256(tokenizer_dir / "spiece.model")
    if spiece != SPIECE_SHA256:
        raise SystemExit(f"spiece.model SHA-256 {spiece} is not the pinned {SPIECE_SHA256}")
    return actual


def download() -> None:
    model, tokenizer = snapshots(offline=False)
    check_files(model, tokenizer)
    print(f"ok: {model} and {tokenizer} (SHA-256 checked)")


def load(device: str):
    import torch
    import transformers

    model_dir, tokenizer_dir = snapshots(offline=True)
    weights_sha256 = check_files(model_dir, tokenizer_dir)
    config = transformers.MT5Config.from_pretrained(model_dir)
    model = transformers.MT5ForConditionalGeneration(config)
    state = torch.load(model_dir / WEIGHTS, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The embedding tables of the encoder and decoder are the shared table, which the file holds once.
    if unexpected or set(missing) - {"encoder.embed_tokens.weight", "decoder.embed_tokens.weight"}:
        raise SystemExit(f"weights do not match MT5ForConditionalGeneration: {missing=} {unexpected=}")
    model.eval().to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)
    return model, tokenizer, weights_sha256


def score(model, tokenizer, samples: list[dict], device: str) -> list[float]:
    """One segment at a time (batch size 1, as the MetricX README runs it): no padding."""
    import torch

    scores = []
    with torch.inference_mode():
        for sample in samples:
            encoded = tokenizer(
                qe_input(sample["src"], sample["mt"]), max_length=MAX_INPUT_LENGTH, truncation=True
            )
            ids = torch.tensor([encoded["input_ids"][:-1]], device=device)  # drop </s> (predict.py)
            mask = torch.ones_like(ids)
            decoder = torch.zeros((1, 1), dtype=torch.long, device=device)
            logits = model(input_ids=ids, attention_mask=mask, decoder_input_ids=decoder).logits
            scores.append(float(torch.clamp(logits[0, 0, SCORE_TOKEN], 0, 25)))
    return scores


def run(args: argparse.Namespace) -> None:
    import os

    import torch
    import transformers

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    os.environ["HF_HUB_OFFLINE"] = "1"
    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in lines if line.strip()]
    keys = [sample["key"] for sample in samples]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate keys in the input")
    started = time.perf_counter()
    model, tokenizer, weights_sha256 = load(args.device)
    loaded = time.perf_counter()
    values = score(model, tokenizer, samples, args.device)
    finished = time.perf_counter()
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("a MetricX score is not a finite number; nothing written")
    meta = {
        "model": MODEL,
        "revision": REVISION,
        "weights_sha256": weights_sha256,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "code": CODE,
        "mode": "qe",
        "precision": "fp32",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "max_input_length": MAX_INPUT_LENGTH,
        "segments": len(samples),
        "load_s": round(loaded - started, 1),
        "score_s": round(finished - loaded, 1),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "input": str(args.input),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    result = {"meta": meta, "scores": dict(zip(keys, values, strict=True))}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(samples)} segments in {meta['score_s']} s -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="fetch the pinned model and tokenizer files and check them")
    scoring = commands.add_parser("score", help="score segments offline, fp32")
    scoring.add_argument("--input", required=True)
    scoring.add_argument("--output", required=True)
    scoring.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "download":
        download()
    else:
        run(args)


if __name__ == "__main__":
    main()
