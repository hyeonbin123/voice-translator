"""말한 언어가 한국어인지 영어인지 판별한다(docs/experiments.md 10절).

  uv run python -m eval.langid_eval --split validation --tag t35_dev
  uv run python -m eval.langid_eval --split test --tag t35_test --thresholds 0.6

FLEURS 녹음마다 7절과 같은 음량으로 맞춘 뒤 브라우저가 자르는 대로 자르고(Silero, 8절: 말 앞 192ms부터
말 뒤 192ms까지), 다시 세 가지로 자른다: 말소리의 처음 1초, 처음 2초, 전체. Whisper 언어 판별에서 한국어와
영어 확률만 남긴다: 큰 쪽이 결정하고, 둘 중 그쪽이 차지하는 몫이 확신이다.
"""

# [파일 역할] 두 사람 대화 모드의 언어 판별(한국어인가 영어인가)과 확신 문턱 τ를 고르는 측정
# 도구(docs/experiments.md 10절).
# [설계 이유] 짧은 대답도 흉내 내려고 같은 녹음에서 앞 1초·2초·전체 세 길이를 만든다. 조각은 브라우저가 실제로
# 보내는 대로(Silero로 자른 한 마디, 말 앞 192ms부터) 자른다: 처음에는 Whisper 단어 시각으로 말 시작을
# 잡았는데, 대부분 녹음에서 첫 단어가 0초로 나와 '1초' 조각이 앞의 조용한 부분을 판별하는 문제가 있어
# 바꿨다(10절 1차 측정, 아래 run의 주석).
# [판단 규칙] 확신이 τ 이상인 판별의 정확도가 모든 조건(두 언어 × 1초·2초·전체)에서 99% 이상인 τ 중 가장 작은
# 것. validation에서 0.6을 골랐다(DIALOG_LANGUAGE_THRESHOLD).
# [연결] 판별 계산은 서버(app/services/stt.WhisperSpeechToText.detect_language)와 같고, 녹음 읽기·Silero는
# eval/eos_eval.py, 한 마디 자르기는 eval/stream_eval.py에서 가져온다.

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime

# eos_eval: FLEURS 읽기·음량 맞추기(load_clips), Silero 판정, 표본 추출률·언어·시드 상수. stream_eval:
# 브라우저처럼 한 마디를 자르는 cut과 앞뒤로 남기는 조각 수(PAD_FRAMES).
import numpy as np

from eval.common import REPORTS, gpu_memory_mb
from eval.eos_eval import LANGS, RATE, SEED, Silero, load_clips
from eval.stream_eval import PAD_FRAMES, cut

# LENGTHS: 조각 길이(말 시작부터 초, None은 전체). THRESHOLDS: τ 후보. PREROLL_S: 마디 앞에 붙는 6조각 ×
# 512표본 = 192ms. REQUIRED_ACCURACY: 판단 규칙의 99%.
LENGTHS = {"1s": 1.0, "2s": 2.0, "full": None}
THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95)
PREROLL_S = PAD_FRAMES * 512 / RATE  # 브라우저 조각은 말 192ms 앞에서 시작한다(7절)
REQUIRED_ACCURACY = 0.99


# 서버와 같은 large-v3-turbo를 GPU float16으로 올린다(Windows면 CUDA DLL 경로 먼저 등록).
def load_model():
    from faster_whisper import WhisperModel

    from app.services.cuda import add_cuda_dll_dirs

    add_cuda_dll_dirs()
    return WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def korean_or_english(model, audio: np.ndarray) -> tuple[str, float, float]:
    """(언어, 두 확률 중 그 언어의 몫, 걸린 초).

    서버의 detect_language와 같은 규칙(VAD 켬, 한국어·영어 확률만 남겨 큰 쪽, 확신 = 큰 쪽 ÷ 둘의 합, 둘 다
    0이면 한국어 0.5)에 판별 시간을 더해 돌려준다.
    """
    began = time.perf_counter()
    _, _, probabilities = model.detect_language(audio, vad_filter=True)
    took = time.perf_counter() - began
    p = dict(probabilities)
    ko, en = p.get("ko", 0.0), p.get("en", 0.0)
    if ko + en == 0:
        return "ko", 0.5, took
    return ("ko" if ko >= en else "en"), max(ko, en) / (ko + en), took


def pieces(clip: np.ndarray) -> dict[str, np.ndarray]:
    """브라우저가 보낼 조각에서 말소리 처음 1초와 2초, 그리고 전체를 잘라 낸다.

    앞 192ms + 1초, 192ms + 2초, 전체다(마디가 그보다 짧으면 슬라이스가 전체가 된다).
    """
    return {
        name: clip[: int((PREROLL_S + seconds) * RATE)] if seconds else clip
        for name, seconds in LENGTHS.items()
    }


# 언어마다 녹음을 읽고, 첫 녹음으로 한 번 예열한 뒤(시간에 넣지 않음), 녹음마다 한 마디로 잘라 세 길이를
# 판별한다. 한 마디로 잘리지 않는 녹음(중간에 1초 넘게 쉬는 등)은 빼고 수를 센다. 잡음 시드는 분할 시드와 녹음
# 순번으로 정해 다시 돌려도 같다.
def run(args: argparse.Namespace) -> None:
    model = load_model()
    silero = Silero()
    guesses, skipped = [], {}
    for lang in LANGS:
        clips = load_clips(lang, args.split)
        korean_or_english(model, clips[0][1])  # 예열, 세지 않음
        skipped[lang] = 0
        for index, (key, clip) in enumerate(clips):
            # 브라우저가 자르는 대로 자른다(8절): 7절의 Whisper 단어 시각은 대부분 녹음에서 첫 단어를 0초에
            # 두어 말이 어디서 시작하는지 알려 주지 못한다.
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


# 조건(언어 × 길이)마다 전체 정확도, 판별 시간 중앙값, τ마다 '확신이 τ보다 낮아 번갈아 처리로 넘기는 비율'과
# '나머지(확신 ≥ τ)의 정확도'를 낸다.
# 그다음 τ마다 조건 중 가장 낮은 나머지 정확도가 99% 이상인지(규칙 충족)와 전체 넘김 비율을 낸다. 나머지가
# 하나도 없는 조건(None)은 0으로 보아 충족하지 않는 것으로 친다.
def summarize(guesses: list[dict], thresholds: list[float]) -> tuple[list[dict], list[dict]]:
    conditions = []
    for lang in LANGS:
        for length in LENGTHS:
            group = [g for g in guesses if g["lang"] == lang and g["length"] == length]
            # 녹음이 모두 빠진(한 마디로 잘리지 않은) 조건은 판별이 없어 비율·중앙값을 None으로 둔다. 0으로
            # 나누거나 빈 목록의 중앙값을 구하다 멈추면 그때까지 GPU로 낸 판별도 보고서에 남지 않는다.
            row = {
                "lang": lang,
                "length": length,
                "count": len(group),
                "accuracy": sum(g["guess"] == lang for g in group) / len(group) if group else None,
                "seconds_p50": statistics.median(g["seconds"] for g in group) if group else None,
            }
            for tau in thresholds:
                sure = [g for g in group if g["confidence"] >= tau]
                row[f"unsure_{tau}"] = 1 - len(sure) / len(group) if group else None
                row[f"sure_accuracy_{tau}"] = (
                    sum(g["guess"] == lang for g in sure) / len(sure) if sure else None
                )
            conditions.append(row)
    rule = []
    total = sum(c["count"] for c in conditions)
    for tau in thresholds:
        worst = min((c[f"sure_accuracy_{tau}"] or 0.0) for c in conditions)
        rule.append(
            {
                "threshold": tau,
                "worst_sure_accuracy": worst,
                "meets": worst >= REQUIRED_ACCURACY,
                "unsure_overall": sum(c[f"unsure_{tau}"] * c["count"] for c in conditions if c["count"])
                / total
                if total
                else None,
            }
        )
    return conditions, rule


# 비율을 백분율 글자로 바꾼다. 판별이 없는 조건(None)은 '-'.
def _percent(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}%}"


# 판별 하나하나까지 담은 JSON과, 조건별 표·τ별 규칙 표를 담은 마크다운을 eval/reports/langid_<태그>_<시각>에
# 남긴다.
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
            cells.append(f"{_percent(c[f'unsure_{tau}'], 1)} / {_percent(sure, 1)}")
        seconds = "-" if c["seconds_p50"] is None else f"{c['seconds_p50'] * 1000:.0f}ms"
        lines.append(
            f"| {c['lang']} | {c['length']} | {c['count']} | {_percent(c['accuracy'], 1)} | "
            f"{seconds} | " + " | ".join(cells) + " |"
        )
    lines += ["", "| τ | 조건 중 가장 낮은 나머지 정확도 | 99% 이상 | 전체 넘김 비율 |", "|---|---|---|---|"]
    for r in rule:
        lines.append(
            f"| {r['threshold']} | {r['worst_sure_accuracy']:.2%} | {'예' if r['meets'] else '아니오'} | "
            f"{_percent(r['unsure_overall'], 1)} |"
        )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")


# --split으로 validation/test, --thresholds로 볼 τ를 고른다.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--tag", required=True)
    # test 분할에는 고른 τ 하나만 적용한다(docs/experiments.md 10절).
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(THRESHOLDS))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
