"""Speech recognition evaluation on FLEURS (task T2).

Usage, from backend/ with the gpu and eval groups installed (`uv sync --group gpu --group eval`):
    uv run python -m eval.stt_eval --split validation --models small large-v3-turbo large-v3 --tag t2_dev

The candidates and the selection rule are written in docs/experiments.md before any run.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import jiwer
import pyarrow.parquet as pq

from app.services.interfaces import InvalidAudioError
from app.services.stt import WhisperSpeechToText
from eval.text_norm import normalize

DATA = Path(__file__).resolve().parents[2] / "data" / "fleurs"
REPORTS = Path(__file__).resolve().parent / "reports"
FLEURS_CONFIG = {"ko": "ko_kr", "en": "en_us"}
SAMPLE_RATE = 16_000


def load_split(language: str, split: str, limit: int | None) -> list[dict]:
    path = DATA / FLEURS_CONFIG[language] / f"{split}.parquet"
    rows = pq.read_table(path, columns=["id", "num_samples", "audio", "transcription"]).to_pylist()
    return rows[:limit] if limit else rows


def gpu_memory_mb() -> tuple[str, float]:
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
    return (name.decode() if isinstance(name, bytes) else name), used


def error_rate(language: str, references: list[str], hypotheses: list[str]) -> float:
    # Corpus-level: total edits over total reference length, not a mean of per-sentence rates.
    return jiwer.cer(references, hypotheses) if language == "ko" else jiwer.wer(references, hypotheses)


def evaluate_model(model_size: str, languages: list[str], split: str, limit: int | None, beam: int) -> dict:
    _, before = gpu_memory_mb()
    stt = WhisperSpeechToText(model_size, beam_size=beam)
    _, after = gpu_memory_mb()
    result: dict = {"model": model_size, "vram_mb": round(after - before), "languages": {}}

    for language in languages:
        rows = load_split(language, split, limit)
        stt.transcribe(rows[0]["audio"]["bytes"], language)  # warm-up, not timed
        references, hypotheses, seconds_per_second, items = [], [], [], []
        for row in rows:
            start = time.perf_counter()
            try:
                text = stt.transcribe(row["audio"]["bytes"], language).text
            except InvalidAudioError:
                text = ""
            elapsed = time.perf_counter() - start
            audio_seconds = row["num_samples"] / SAMPLE_RATE
            reference = normalize(row["transcription"], language)
            hypothesis = normalize(text, language)
            references.append(reference)
            hypotheses.append(hypothesis)
            seconds_per_second.append(elapsed / audio_seconds)
            items.append(
                {
                    "id": row["id"],
                    "audio_s": round(audio_seconds, 2),
                    "elapsed_s": round(elapsed, 3),
                    "error": round(error_rate(language, [reference], [hypothesis]), 4),
                    "reference": reference,
                    "hypothesis": hypothesis,
                }
            )
        result["languages"][language] = {
            "metric": "CER" if language == "ko" else "WER",
            "error": error_rate(language, references, hypotheses),
            "speed_p50": statistics.median(seconds_per_second),
            "speed_p95": statistics.quantiles(seconds_per_second, n=20)[18],
            "count": len(rows),
            "items": items,
        }

    del stt
    gc.collect()
    return result


def write_report(results: list[dict], args: argparse.Namespace, gpu_name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = REPORTS / f"stt_{args.tag}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# 음성 인식 평가 ({args.tag})",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}",
        f"- 데이터: FLEURS {args.split}" + (f", 언어별 앞 {args.limit}개" if args.limit else ""),
        f"- GPU: {gpu_name}, float16, beam {args.beam}",
        "- 속도: 음성 1초당 처리 시간(초). 언어별 첫 호출 제외",
        "",
        "| 모델 | VRAM (MB) | 한국어 CER | 영어 WER | 평균 "
        "| 한국어 속도 p50 | 영어 속도 p50 | 속도 p95 (최대) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        by_language = result["languages"]
        ko, en = by_language.get("ko"), by_language.get("en")
        mean = statistics.mean(v["error"] for v in by_language.values())
        p95 = max(v["speed_p95"] for v in by_language.values())
        lines.append(
            f"| {result['model']} | {result['vram_mb']} "
            f"| {ko['error']:.2%} | {en['error']:.2%} | {mean:.2%} "
            f"| {ko['speed_p50']:.3f} | {en['speed_p50']:.3f} | {p95:.3f} |"
            if ko and en
            else f"| {result['model']} | {result['vram_mb']} | - | - | {mean:.2%} | - | - | {p95:.3f} |"
        )
    for result in results:
        for language, values in result["languages"].items():
            lines += ["", f"## {result['model']} / {language}: 오류가 큰 5개", ""]
            for item in sorted(values["items"], key=lambda i: i["error"], reverse=True)[:5]:
                lines.append(
                    f"- id {item['id']} ({item['error']:.0%}): "
                    f"참조 `{item['reference']}` / 결과 `{item['hypothesis']}`"
                )

    report = base.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--models", nargs="+", default=["small", "large-v3-turbo", "large-v3"])
    parser.add_argument("--languages", nargs="+", choices=["ko", "en"], default=["ko", "en"])
    parser.add_argument("--limit", type=int, default=None, help="first N utterances per language")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    gpu_name, _ = gpu_memory_mb()
    results = []
    for model_size in args.models:
        print(f"evaluating {model_size} ...", flush=True)
        results.append(evaluate_model(model_size, args.languages, args.split, args.limit, args.beam))
    print(f"written to {write_report(results, args, gpu_name)}")


if __name__ == "__main__":
    main()
