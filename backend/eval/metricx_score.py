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

# [파일 역할] Hy-MT2 시험의 '오염 거부권' 점검에 쓰는 MetricX-24 참조 없는(QE) 문장 점수를 매기는 독립
# 스크립트(docs/experiments.md 15절).
# [설계 이유] FLEURS는 FLORES 문장이라 Hy-MT2가 학습 때 봤을 수 있다. 그래서 공개일 뒤의 새 글(오염 점검
# 세트)을 Hy-MT2와 opus로 번역하고, 참조 번역이 없으므로 참조 없는 점수로 비교한다. 이 점수는 채택 근거가
# 아니라 거부에만 쓴다(15절: 참조 없는 점수는 매끄러운 LLM 출력을 좋게 볼 수 있다).
# 공식 코드는 다른 transformers 버전을 요구해, COMET 채점과 같은 별도 가상환경에서 같은 계산을 직접 옮긴 이
# 스크립트로 돌린다. 파일 맨 위에서는 표준 라이브러리만 import한다(ruff·backend 테스트가 읽을 수 있게).
# 가중치·토크나이저는 리비전과 SHA-256으로 고정한다.
# [연결] 입력 JSONL은 eval/mt_decide.py(contam 단계)가 만들고, 출력 점수도 mt_decide가 읽어 짝지은 구간을
# 계산한다.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

# 고정한 모델·토크나이저(15절에 같은 값이 적혀 있다). MAX_INPUT_LENGTH: 공식 코드의 최대 입력 토큰 1536.
# SCORE_TOKEN: 점수로 읽는 출력 토큰 번호(<extra_id_10>). CODE: 계산을 옮겨 온 공식 코드의 커밋(메타데이터에
# 남김).
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


# 4.9GB 가중치를 1MiB씩 읽어 SHA-256을 계산한다.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# 참조 없는 입력 형식: "source: <원문> candidate: <번역문>".
def qe_input(source: str, translation: str) -> str:
    """MetricX-24's reference-free input (predict.py with --qe)."""
    return "source: " + source + " candidate: " + translation


# 고정 리비전의 모델 폴더와 토크나이저 파일 폴더. 내려받기 스레드를 하나로 제한하는 이유는 아래 영어
# 주석(Windows 심볼릭 링크 권한 경쟁)이다.
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


# 가중치와 토크나이저 모델(spiece.model)이 고정한 파일인지 해시로 확인하고, 다르면 멈춘다.
def check_files(model_dir: Path, tokenizer_dir: Path) -> str:
    """SHA-256 of the weights; stops unless the weights and the tokenizer model are the pinned files."""
    actual = sha256(model_dir / WEIGHTS)
    if actual != WEIGHTS_SHA256:
        raise SystemExit(f"{WEIGHTS} SHA-256 {actual} is not the pinned {WEIGHTS_SHA256}")
    spiece = sha256(tokenizer_dir / "spiece.model")
    if spiece != SPIECE_SHA256:
        raise SystemExit(f"spiece.model SHA-256 {spiece} is not the pinned {SPIECE_SHA256}")
    return actual


# 준비 단계: 받고 해시를 확인한다.
def download() -> None:
    model, tokenizer = snapshots(offline=False)
    check_files(model, tokenizer)
    print(f"ok: {model} and {tokenizer} (SHA-256 checked)")


# MetricX의 회귀 모델은 transformers의 MT5ForConditionalGeneration과 층이 같으므로 그 클래스에 가중치를
# 넣는다. pickle 파일이라 weights_only=True로 텐서만 읽어 임의 코드 실행을 막는다. 빠진 키가 공유 임베딩 두
# 개(파일에 한 번만 저장된 표) 말고 더 있거나 남는 키가 있으면 다른 모델로 보고 멈춘다. 정밀도는 fp32(mT5는
# fp16에서 넘치고, 이 GPU에는 bf16이 없다: 위 docstring).
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


# 문장마다(배치 1, 패딩 없음): 입력을 토큰화해 1536토큰으로 자르고 끝의 </s>를 뺀 뒤(공식 predict.py와 같게),
# 디코더에 토큰 0을 한 단계 넣어 SCORE_TOKEN 위치의 로짓을 점수로 읽고 0~25로 자른다. 오류 점수라 낮을수록
# 좋다.
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


# score 명령: 오프라인(HF_HUB_OFFLINE=1)으로 적재·채점하고, 키가 겹치거나 유한하지 않은 점수가 있으면 아무것도
# 쓰지 않고 멈춘다. 결과에는 키별 점수와, 모델·가중치 해시·토크나이저·공식 코드
# 커밋·모드(qe)·정밀도·장치·시간·패키지 버전 메타데이터를 남긴다. 기본 장치는 CPU다(15절: 400문장이 CPU로 약
# 7분이라 GPU를 쓰지 않음).
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


# 하위 명령 download / score.
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
