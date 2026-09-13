"""Speech recognition evaluation on FLEURS (tasks T2 and T14).

Usage, from backend/ with the gpu and eval groups installed (`uv sync --group gpu --group eval`):
    uv run python -m eval.stt_eval --split validation --models small large-v3-turbo large-v3 --tag t2_dev
    uv run python -m eval.stt_eval --split validation --models large-v3-turbo --variants a b c d --tag t14_dev

The candidates and the selection rule are written in docs/experiments.md before any run.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import statistics
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

import jiwer
import numpy as np
import pyarrow.parquet as pq

from app.services.interfaces import InvalidAudioError
from app.services.stt import WhisperSpeechToText
from eval.common import DATA, FLEURS_CONFIG, REPORTS, gpu_memory_mb
from eval.text_norm import normalize

SAMPLE_RATE = 16_000
# Decoding variants compared in T14 (docs/experiments.md 1-1). "a" is faster-whisper's defaults, as in T2.
VARIANTS: dict[str, dict] = {
    "a": {},
    "b": {"vad_filter": True},
    "c": {"retry_without_no_speech": True},
    "d": {"no_speech_threshold": None},
}
NO_SPEECH_INPUTS = ("silence", "noise")


def load_split(language: str, split: str, limit: int | None) -> list[dict]:
    path = DATA / "fleurs" / FLEURS_CONFIG[language] / f"{split}.parquet"
    rows = pq.read_table(path, columns=["id", "num_samples", "audio", "transcription"]).to_pylist()
    return rows[:limit] if limit else rows


def error_rate(language: str, references: list[str], hypotheses: list[str]) -> float:
    # Corpus-level: total edits over total reference length, not a mean of per-sentence rates.
    return jiwer.cer(references, hypotheses) if language == "ko" else jiwer.wer(references, hypotheses)


def no_speech_wav(kind: str, seconds: float = 3.0) -> bytes:
    """Audio without speech: digital silence, or quiet white noise (peak amplitude 0.01, fixed seed)."""
    count = int(SAMPLE_RATE * seconds)
    samples = np.zeros(count) if kind == "silence" else np.random.default_rng(0).uniform(-0.01, 0.01, count)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def recognize(stt: WhisperSpeechToText, audio: bytes, language: str) -> str:
    """The recognized text, or "" when nothing was found (what the server reports as no speech)."""
    try:
        return stt.transcribe(audio, language).text
    except InvalidAudioError:
        return ""


def evaluate_model(
    model_size: str, variant: str, languages: list[str], split: str, limit: int | None, beam: int
) -> dict:
    _, before = gpu_memory_mb()
    stt = WhisperSpeechToText(model_size, beam_size=beam, **VARIANTS[variant])
    _, after = gpu_memory_mb()
    result: dict = {
        "model": model_size,
        "variant": variant,
        "vram_mb": round(after - before),
        "languages": {},
        "no_speech": {},
    }

    for language in languages:
        rows = load_split(language, split, limit)
        recognize(stt, rows[0]["audio"]["bytes"], language)  # warm-up, not timed
        references, hypotheses, seconds_per_second, items = [], [], [], []
        for row in rows:
            start = time.perf_counter()
            text = recognize(stt, row["audio"]["bytes"], language)
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
        pairs = list(zip(references, hypotheses, strict=True))
        result["languages"][language] = {
            "metric": "CER" if language == "ko" else "WER",
            "error": error_rate(language, references, hypotheses),
            "empty": sum(not hypothesis for _, hypothesis in pairs),
            # More than twice the reference's length: likely made-up text (T14's hallucination signal).
            "long": sum(len(hypothesis) > 2 * len(reference) for reference, hypothesis in pairs),
            "speed_p50": statistics.median(seconds_per_second),
            "speed_p95": statistics.quantiles(seconds_per_second, n=20)[18],
            "count": len(rows),
            "items": items,
        }
        # Inputs without speech must come back empty; anything else is text the model made up.
        for kind in NO_SPEECH_INPUTS:
            result["no_speech"][f"{language}/{kind}"] = recognize(stt, no_speech_wav(kind), language)

    del stt
    gc.collect()
    return result


def cell(values: dict | None, key: str, spec: str) -> str:
    return "-" if values is None else format(values[key], spec)


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
        "- 변형: a 기본값, b `vad_filter=True`, c 빈 결과면 무음 판정 없이 한 번 더, "
        "d `no_speech_threshold=None` (docs/experiments.md 1-1)",
        "- 속도: 음성 1초당 처리 시간(초). 언어별 첫 호출 제외",
        "- 빈 결과: 결과가 빈 문장 수. 환각 신호: 참조 길이의 2배를 넘는 결과 수. "
        "무음 확인: 무음·잡음 3초를 언어마다 넣어 빈 결과가 나온 수",
        "",
        "| 모델/변형 | VRAM (MB) | 한국어 CER | 영어 WER | 평균 | 빈 결과 (한/영) | 환각 신호 (한/영) "
        "| 한국어 속도 p50 | 영어 속도 p50 | 속도 p95 (최대) | 무음 확인 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        by_language = result["languages"]
        ko, en = by_language.get("ko"), by_language.get("en")
        mean = statistics.mean(v["error"] for v in by_language.values())
        p95 = max(v["speed_p95"] for v in by_language.values())
        no_speech = result["no_speech"]
        lines.append(
            f"| {result['model']}/{result['variant']} | {result['vram_mb']} "
            f"| {cell(ko, 'error', '.2%')} | {cell(en, 'error', '.2%')} | {mean:.2%} "
            f"| {cell(ko, 'empty', 'd')}/{cell(en, 'empty', 'd')} "
            f"| {cell(ko, 'long', 'd')}/{cell(en, 'long', 'd')} "
            f"| {cell(ko, 'speed_p50', '.3f')} | {cell(en, 'speed_p50', '.3f')} | {p95:.3f} "
            f"| {sum(not text for text in no_speech.values())}/{len(no_speech)} |"
        )
    for result in results:
        label = f"{result['model']}/{result['variant']}"
        made_up = {name: text for name, text in result["no_speech"].items() if text}
        if made_up:
            lines += ["", f"## {label}: 말이 없는 입력에서 나온 글자", ""]
            lines += [f"- {name}: `{text}`" for name, text in made_up.items()]
        for language, values in result["languages"].items():
            lines += ["", f"## {label} / {language}: 오류가 큰 5개", ""]
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
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["a"])
    parser.add_argument("--languages", nargs="+", choices=["ko", "en"], default=["ko", "en"])
    parser.add_argument("--limit", type=int, default=None, help="first N utterances per language")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    gpu_name, _ = gpu_memory_mb()
    results = []
    for model_size in args.models:
        for variant in args.variants:
            print(f"evaluating {model_size}/{variant} ...", flush=True)
            results.append(
                evaluate_model(model_size, variant, args.languages, args.split, args.limit, args.beam)
            )
    print(f"written to {write_report(results, args, gpu_name)}")


if __name__ == "__main__":
    main()
