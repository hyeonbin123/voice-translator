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

# [파일 역할] 저장된 번역 결과의 문장마다 COMET-22 점수(모델 기반 번역 품질 지표)를 매기는 독립 스크립트.
# 기록된 번역 결정을 chrF 말고 다른 지표로도 다시 보는 재분석(docs/experiments.md 13절)과 Hy-MT2 시험(15절)에
# 썼다.
# [설계 이유] unbabel-comet 2.2.7은 transformers<5·numpy<2가 필요해 backend 가상환경(eval 그룹은 transformers
# 5·numpy 2)과 함께 설치할 수 없다. 그래서 별도 가상환경에서 돌리고, 파일 맨 위에서는 표준 라이브러리만
# import해 ruff와 backend 테스트가 이 파일을 읽을 수 있게 했다(위 docstring). 결과가 '어떤 모델로 매겼는지'에
# 묶이도록 모델·인코더 리비전과 체크포인트 SHA-256을 고정하고, 점수를 매길 때 실제로 읽은 파일을 해시해
# 메타데이터에 남긴다.
# [흐름] download(고정 리비전 받기·체크포인트 검증) → score(오프라인으로 적재 → 원문 길이순 배치로 채점 →
# 유한하지 않은 점수가 있으면 아무것도 쓰지 않고 멈춤 → 키별 점수와 메타데이터를 JSON으로).
# [연결] 입력은 `python -m eval.reanalysis comet-input`(재분석)이나 `python -m eval.mt_decide
# comet-input`(Hy-MT2 시험)이 만든 JSONL(key, src, mt, ref)이고, 출력 JSON은 eval/reanalysis.py와
# eval/mt_decide.py가 읽어 짝지은 차이와 구간을 계산한다.

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

# 고정한 COMET-22 모델: 저장소, 리비전, 체크포인트 경로와 SHA-256(docs/experiments.md 13절에 같은 값이 적혀
# 있다). 인코더(xlm-roberta-large)의 토크나이저·설정도 리비전을 고정해 받는다(위 영어 주석). BATCH_SIZE: 한
# 번에 채점할 문장 수.
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


# 큰 체크포인트를 1MiB씩 읽어 SHA-256을 계산한다.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Hugging Face 캐시에서 고정 리비전의 모델 폴더와 인코더 파일 폴더를 얻는다. offline=True면 내려받지 않고
# 캐시에 있는 것만 쓴다(채점 때). 인코더는 필요한 네 파일만 받는다(allow_patterns).
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


# 체크포인트가 고정한 파일인지 해시로 확인하고, 다르면 멈춘다.
def check_checkpoint(model_dir: Path) -> str:
    """SHA-256 of the checkpoint file in `model_dir`; stops unless it is the pinned one."""
    actual = sha256(model_dir / CHECKPOINT)
    if actual != CHECKPOINT_SHA256:
        raise SystemExit(f"{CHECKPOINT} SHA-256 {actual} is not the pinned {CHECKPOINT_SHA256}")
    return actual


# 준비 단계: 모델·인코더를 받고 체크포인트 해시와 인코더 파일이 모두 있는지 확인한다.
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
    # torch·yaml·comet은 COMET 전용 환경에만 있으므로 함수 안에서 import한다.
    # 체크포인트를 먼저 해시해 받은 뒤 바뀐 캐시를 거절하고, hparams.yaml의 class_identifier로 모델 클래스를
    # 골라 적재한다. 인코더 가중치는 체크포인트에 들어 있어 따로 받지 않고(load_pretrained_weights=False),
    # 토크나이저·설정은 고정한 인코더 폴더에서 읽는다(pretrained_model). 먼저 CPU로 읽은 뒤 장치로 옮기고,
    # fp16이면 반정밀도로 바꾼다. comet의 sparsemax_patch 우회가 필요한 체크포인트는 이 경로가 다루지 않으므로
    # 멈춘다.
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


# comet의 predict처럼 원문 길이순으로 배치를 만들어(비슷한 길이끼리 묶어 패딩 낭비를 줄임) 채점하고, 점수는
# 입력 순서대로 되돌려 놓는다. inference_mode로 기울기 계산을 끈다. 점수는 float32로 바꿔 CPU로 가져온다.
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


# score 명령. 잘못된 조합(GPU가 없는데 cuda, CPU에서 fp16)은 시작 전에 거절하고, HF_HUB_OFFLINE=1로 채점 중
# 내려받기를 막는다. 입력 키가 겹치면 결과가 덮어써지므로 멈춘다. --limit은 점검용으로 앞 N문장만 채점한다.
# NaN·무한대 점수가 하나라도 있으면 파일을 쓰지 않고 멈춘다. 메타데이터에는 모델·인코더 리비전, 실제로 읽은
# 체크포인트 해시, 장치·GPU 이름·정밀도·배치 크기, 적재·채점 시간, 패키지 버전, 입력 경로를 남긴다. fp16과
# fp32 점수가 같은지는 13절의 관문(문장 점수 차이 최대 0.01 이하 등)으로 따로 확인했다.
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


# 하위 명령 download / score. score는 입력·출력·장치·정밀도를 반드시 받는다.
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
