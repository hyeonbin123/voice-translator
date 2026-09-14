"""T35: which language was spoken, Korean or English (docs/experiments.md 10).

  uv run python -m eval.langid_eval --split validation --tag t35_dev
  uv run python -m eval.langid_eval --split test --tag t35_test --thresholds 0.8

Each FLEURS recording, brought to the same level as in section 7, is cut as the browser cuts it (Silero,
section 8: from 192 ms before speech to 192 ms after it), then three ways: its first second of speech,
its first two, and all of it. Whisper's language detection keeps the Korean and English probabilities
only: the larger one decides, and its share of the two is the confidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime

import numpy as np

from eval.common import REPORTS, gpu_memory_mb
from eval.eos_eval import LANGS, RATE, SEED, Silero, load_clips
from eval.stream_eval import PAD_FRAMES, cut

LENGTHS = {"1s": 1.0, "2s": 2.0, "full": None}
THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95)
PREROLL_S = PAD_FRAMES * 512 / RATE  # the browser's clip starts 192 ms before speech (section 7)
REQUIRED_ACCURACY = 0.99


def load_model():
    from faster_whisper import WhisperModel

    from app.services.cuda import add_cuda_dll_dirs

    add_cuda_dll_dirs()
    return WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def korean_or_english(model, audio: np.ndarray) -> tuple[str, float, float]:
    """(the language, its share of the two probabilities, seconds taken)."""
    began = time.perf_counter()
    _, _, probabilities = model.detect_language(audio, vad_filter=True)
    took = time.perf_counter() - began
    p = dict(probabilities)
    ko, en = p.get("ko", 0.0), p.get("en", 0.0)
    if ko + en == 0:
        return "ko", 0.5, took
    return ("ko" if ko >= en else "en"), max(ko, en) / (ko + en), took


def pieces(clip: np.ndarray) -> dict[str, np.ndarray]:
    """The first second and two of speech, and all of it, from the clip the browser would send."""
    return {
        name: clip[: int((PREROLL_S + seconds) * RATE)] if seconds else clip
        for name, seconds in LENGTHS.items()
    }


def run(args: argparse.Namespace) -> None:
    model = load_model()
    silero = Silero()
    guesses, skipped = [], {}
    for lang in LANGS:
        clips = load_clips(lang, args.split)
        korean_or_english(model, clips[0][1])  # warm-up, not counted
        skipped[lang] = 0
        for index, (key, clip) in enumerate(clips):
            # Cut as the browser cuts it (section 8): the Whisper word times of section 7 put the first word
            # at 0 s in most recordings, so they cannot show where speech starts.
            utterance = cut(silero, clip, seed=SEED[args.split] * 1000 + index)
            if utterance is None:
                skipped[lang] += 1
                continue
            for length, piece in pieces(utterance["audio"]).items():
                guess, confidence, took = korean_or_english(model, piece)
                guesses.append(
                    {
                        "key": key,
                        "lang": lang,
                        "length": length,
                        "guess": guess,
                        "confidence": confidence,
                        "seconds": took,
                    }
                )
        print(
            f"{lang}: {sum(1 for g in guesses if g['lang'] == lang) // len(LENGTHS)} recordings", flush=True
        )
    write_report(guesses, skipped, args)


def summarize(guesses: list[dict], thresholds: list[float]) -> tuple[list[dict], list[dict]]:
    conditions = []
    for lang in LANGS:
        for length in LENGTHS:
            group = [g for g in guesses if g["lang"] == lang and g["length"] == length]
            row = {
                "lang": lang,
                "length": length,
                "count": len(group),
                "accuracy": sum(g["guess"] == lang for g in group) / len(group),
                "seconds_p50": statistics.median(g["seconds"] for g in group),
            }
            for tau in thresholds:
                sure = [g for g in group if g["confidence"] >= tau]
                row[f"unsure_{tau}"] = 1 - len(sure) / len(group)
                row[f"sure_accuracy_{tau}"] = (
                    sum(g["guess"] == lang for g in sure) / len(sure) if sure else None
                )
            conditions.append(row)
    rule = []
    for tau in thresholds:
        worst = min((c[f"sure_accuracy_{tau}"] or 0.0) for c in conditions)
        rule.append(
            {
                "threshold": tau,
                "worst_sure_accuracy": worst,
                "meets": worst >= REQUIRED_ACCURACY,
                "unsure_overall": sum(c[f"unsure_{tau}"] * c["count"] for c in conditions)
                / sum(c["count"] for c in conditions),
            }
        )
    return conditions, rule


def write_report(guesses: list[dict], skipped: dict, args: argparse.Namespace) -> None:
    conditions, rule = summarize(guesses, args.thresholds)
    gpu, _ = gpu_memory_mb()
    stamp = datetime.now(UTC)
    base = REPORTS / f"langid_{args.tag}_{stamp:%Y%m%d_%H%M%S}"
    report = {"gpu": gpu, "skipped": skipped, "conditions": conditions, "rule": rule, "guesses": guesses}
    base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# 언어 판별 ({args.tag})",
        "",
        f"- 날짜: {stamp.isoformat()}, FLEURS {args.split}, GPU {gpu}. 한국어·영어 확률만 남겨 큰 쪽, "
        "확신 = 큰 쪽 ÷ 둘의 합",
        f"- 조각: 브라우저처럼 Silero로 자른 한 마디(말 시작 192ms 앞부터)의 앞 1초·2초·전체. "
        f"한 마디로 잘리지 않아 뺀 녹음: {skipped}",
        "",
        "| 언어 | 길이 | 녹음 | 정확도 | 판별 시간 p50 | "
        + " | ".join(f"τ={t}: 넘김 / 나머지 정확도" for t in args.thresholds)
        + " |",
        "|---|---|---|---|---|" + "---|" * len(args.thresholds),
    ]
    for c in conditions:
        cells = []
        for tau in args.thresholds:
            sure = c[f"sure_accuracy_{tau}"]
            cells.append(f"{c[f'unsure_{tau}']:.1%} / {'-' if sure is None else f'{sure:.1%}'}")
        lines.append(
            f"| {c['lang']} | {c['length']} | {c['count']} | {c['accuracy']:.1%} | "
            f"{c['seconds_p50'] * 1000:.0f}ms | " + " | ".join(cells) + " |"
        )
    lines += ["", "| τ | 조건 중 가장 낮은 나머지 정확도 | 99% 이상 | 전체 넘김 비율 |", "|---|---|---|---|"]
    for r in rule:
        lines.append(
            f"| {r['threshold']} | {r['worst_sure_accuracy']:.2%} | {'예' if r['meets'] else '아니오'} | "
            f"{r['unsure_overall']:.1%} |"
        )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--tag", required=True)
    # The test split gets only the chosen threshold (docs/experiments.md 10).
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(THRESHOLDS))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
