"""T34: offline check of live subtitles while speaking (docs/experiments.md 8).

  uv run python -m eval.stream_eval run --split validation --tag t34_dev

Each FLEURS clip is cut as conversation mode cuts it in the browser (Silero VAD, 1000 ms of quiet, T33),
then streamed through a simulated clock. While the person speaks, the server recognizes all of the
utterance's audio received so far and translates the result, one update at a time on its single model
thread. The model calls are real and timed on this machine's GPU; their measured times advance the clock.
Network time is taken as zero. Once the browser has seen 192 ms of quiet, the clip conversation mode would
send is complete: the server recognizes it with the final settings, and that result replaces the live text.

Texts are compared by their letters only (app/services/live.py): Korean spacing changes between results.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import UTC, datetime
from difflib import SequenceMatcher

import numpy as np

from app.services.live import common_start, letters, stable_length
from eval.common import MODELS, REPORTS, gpu_memory_mb
from eval.eos_eval import (
    LANGS,
    PREROLL_MS,
    QUIET_FLOOR_DBFS,
    RATE,
    S_CHUNK,
    S_START_FRAMES,
    SEED,
    Silero,
    group,
    load_clips,
    pink_noise,
)

SILENCE_MS = 1000  # D9
PAD_FRAMES = round(PREROLL_MS / (S_CHUNK * 1000 / RATE))  # 6 frames kept before and after speech (silero.ts)
MAX_CLIP_S = 28.9  # longer clips are cut in two by the browser's 29 s limit
LEAD_S, TAIL_S = 1.0, 1.5
PER_LANGUAGE = 40
INTERVALS_MS = (250, 500, 1000)
# Every setting keeps the server's VAD filter (config.stt_vad_filter) and no-speech threshold.
BASE = {"vad_filter": True, "no_speech_threshold": 0.6}
FINAL = {**BASE, "beam_size": 5}  # conversation mode's recognition (app/services/models.py)
PARTIAL = {"P1": {**BASE, "beam_size": 1, "temperature": 0.0}, "P5": FINAL}


def quantize(audio: np.ndarray) -> np.ndarray:
    """The 16-bit PCM the browser sends."""
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768


def long_pauses(flags: np.ndarray) -> int:
    """Quiet runs of at least PAD_FRAMES inside an utterance: each starts a final that speech then drops."""
    count = run = 0
    for flag in flags:
        if flag:
            count += run >= PAD_FRAMES
            run = 0
        else:
            run += 1
    return count


def cut(silero: Silero, clip: np.ndarray, seed: int) -> dict | None:
    """The clip conversation mode would send for this recording, or None unless it is exactly one."""
    track = np.concatenate(
        [np.zeros(int(LEAD_S * RATE), np.float32), clip, np.zeros(int(TAIL_S * RATE), np.float32)]
    )
    noise = pink_noise(len(track), np.random.default_rng(seed)) * 10 ** (QUIET_FLOOR_DBFS / 20)
    track = (track + noise).astype(np.float32)  # the VAD takes float32 only
    flags = silero.flags(track)
    utterances = group(flags, S_CHUNK, SILENCE_MS, S_START_FRAMES)
    if len(utterances) != 1:
        return None
    first, end = utterances[0]["start"] // S_CHUNK, utterances[0]["end"] // S_CHUNK
    begin = first - min(PAD_FRAMES, first)
    if (end + PAD_FRAMES - begin) * S_CHUNK / RATE >= MAX_CLIP_S:
        return None
    return {
        "audio": quantize(track[begin * S_CHUNK : (end + PAD_FRAMES) * S_CHUNK]),
        "detected_s": (first + S_START_FRAMES - begin) * S_CHUNK / RATE,  # when the browser starts streaming
        "pauses": long_pauses(flags[first:end]),
    }


def recognize(model, audio: np.ndarray, lang: str, options: dict) -> tuple[str, float]:
    began = time.perf_counter()
    segments, _ = model.transcribe(audio, language=lang, **options)
    text = " ".join(segment.text.strip() for segment in segments).strip()  # the model runs while this reads
    return text, time.perf_counter() - began


def translate(translator, text: str, lang: str, target: str) -> tuple[str, float]:
    began = time.perf_counter()
    translation = translator.translate(text, lang, target)
    return translation, time.perf_counter() - began


def final_result(model, translator, audio: np.ndarray, lang: str, target: str) -> tuple[dict | None, str]:
    """Conversation mode's result for the clip, with each word's letters and end time, or why there is
    none."""
    text, stt_s = recognize(model, audio, lang, FINAL)
    if not letters(text):
        return None, "no_speech"
    # A second, untimed pass for word times; it must have the timed pass's letters.
    segments, _ = model.transcribe(audio, language=lang, word_timestamps=True, **FINAL)
    timed = [word for segment in segments for word in (segment.words or [])]
    if letters("".join(word.word for word in timed)) != letters(text):
        return None, "word_times_differ"
    spans, cursor = [], 0
    for word in timed:
        size = len(letters(word.word))
        if size:  # a word of punctuation alone has no letters to show
            spans.append((cursor, cursor + size, word.end))
        cursor += size
    translation, mt_s = translate(translator, text, lang, target)
    return {"text": text, "spans": spans, "stt_s": stt_s, "translation": translation, "mt_s": mt_s}, ""


def covered(shown: str, final: str) -> np.ndarray:
    """Which letters of the final text the shown text has, by aligning the two."""
    mask = np.zeros(len(final), bool)
    for _, start, size in SequenceMatcher(None, shown, final, autojunk=False).get_matching_blocks():
        mask[start : start + size] = True
    return mask


def appearance_lags(events: list[tuple[float, str]], spans: list[tuple[int, int, float]]) -> list[float]:
    """Per final word: the first time its letters were all on screen and stayed, minus the word's end.
    Events are (time, letters) with the final result last."""
    final = events[-1][1]
    masks = [covered(shown, final) for _, shown in events]
    lags = []
    for start, end, word_end in spans:
        appeared = events[-1][0]
        for (shown_at, _), mask in zip(reversed(events), reversed(masks), strict=True):
            if not mask[start:end].all():
                break
            appeared = shown_at
        lags.append(appeared - word_end)
    return lags


def erased(sequence: list[str]) -> int:
    """Letters taken off the end of the screen from one update to the next (normalized erasure's
    numerator)."""
    total, before = 0, ""
    for shown in sequence:
        total += len(before) - common_start(before, shown)
        before = shown
    return total


def dark_changes(shown: list[tuple[str, int]], final: str) -> tuple[int, int]:
    """(dark letters later taken away, letters that turned dark) over (letters, dark letters) on screen.
    The final result takes dark letters away too; its own letters are not counted as turning dark."""
    dropped = became = 0
    for (before, dark_before), (after, dark_after) in zip(shown, shown[1:], strict=False):
        kept = min(dark_before, common_start(before, after))
        dropped += dark_before - kept
        became += max(0, dark_after - kept)
    if shown:
        before, dark_before = shown[-1]
        dropped += dark_before - min(dark_before, common_start(before, final))
    return dropped, became


def simulate(
    model, translator, utterance: dict, final: dict, lang: str, target: str, interval_s: float, options: dict
) -> dict:
    audio = utterance["audio"]
    duration = len(audio) / RATE
    free_at, last_start = 0.0, None  # when the model thread is free, when the last update started
    previous = previous_translation = None
    sources: list[tuple[float, str, int]] = []  # (shown at, text, dark characters)
    translations: list[tuple[float, str, int]] = []
    model_s, updates = 0.0, 0
    while True:
        earliest = interval_s if last_start is None else last_start + interval_s
        start = max(utterance["detected_s"], free_at, earliest)
        if start >= duration:  # the whole clip is in: the final takes over
            break
        text, took = recognize(model, audio[: int(start * RATE)], lang, options)
        model_s += took
        updates += 1
        last_start, free_at = start, start + took
        dark = stable_length(previous, text)
        previous = text
        if sources and (text, dark) == sources[-1][1:]:
            continue  # nothing new to show or translate
        text_changed = not sources or text != sources[-1][1]
        sources.append((free_at, text, dark))
        if text_changed and letters(text):
            translation, took = translate(translator, text, lang, target)
            model_s += took
            free_at += took
            translations.append((free_at, translation, stable_length(previous_translation, translation)))
            previous_translation = translation
    final_start = max(duration, free_at)  # an update still running holds the one model thread
    source_final_at = final_start + final["stt_s"]
    translation_final_at = source_final_at + final["mt_s"]
    final_letters, translation_letters = letters(final["text"]), letters(final["translation"])
    source_dark = [(letters(text), len(letters(text[:dark]))) for _, text, dark in sources]
    translation_dark = [(letters(text), len(letters(text[:dark]))) for _, text, dark in translations]
    events = [(at, shown) for (at, _, _), (shown, _) in zip(sources, source_dark, strict=True)]
    source_dropped, source_became = dark_changes(source_dark, final_letters)
    translation_dropped, translation_became = dark_changes(translation_dark, translation_letters)
    return {
        "lags": appearance_lags(events + [(source_final_at, final_letters)], final["spans"]),
        "letters": len(final_letters),
        "source_erased": erased([shown for shown, _ in source_dark] + [final_letters]),
        "source_dark_dropped": source_dropped,
        "source_dark_became": source_became,
        "translation_letters": len(translation_letters),
        "translation_erased": erased([shown for shown, _ in translation_dark] + [translation_letters]),
        "translation_dark_dropped": translation_dropped,
        "translation_dark_became": translation_became,
        "model_s": model_s,
        "duration_s": duration,
        "updates": updates,
        "final_wait_s": final_start - duration,
        "final_text_s": translation_final_at - final["spans"][-1][2],
    }


def load_models():
    from faster_whisper import WhisperModel

    from app.services.cuda import add_cuda_dll_dirs
    from app.services.translation import DirectionalTranslator, MarianTranslator

    add_cuda_dll_dirs()
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    ct2 = MODELS / "ct2"
    translator = DirectionalTranslator(
        {
            ("ko", "en"): MarianTranslator(ct2 / "opus-mt-tc-big-ko-en", "ko", "en"),
            ("en", "ko"): MarianTranslator(ct2 / "opus-mt-tc-big-en-ko", "en", "ko", by_sentence=True),
        }
    )
    return model, translator


def utterances(silero: Silero, model, translator, lang: str, split: str, count: int) -> tuple[list, dict]:
    """The first `count` usable clips in a seeded order, with the number skipped for each reason."""
    target = "ko" if lang == "en" else "en"
    clips = load_clips(lang, split)
    random.Random(f"stream-{SEED[split]}-{lang}").shuffle(clips)
    chosen, skipped = [], {"not_one_utterance": 0, "no_speech": 0, "word_times_differ": 0}
    for index, (key, clip) in enumerate(clips):
        utterance = cut(silero, clip, seed=SEED[split] * 1000 + index)
        if utterance is None:
            skipped["not_one_utterance"] += 1
            continue
        final, reason = final_result(model, translator, utterance["audio"], lang, target)
        if final is None:
            skipped[reason] += 1
            continue
        chosen.append((key, utterance, final))
        if len(chosen) == count:
            break
    return chosen, skipped


def summarize(results: list[dict]) -> dict:
    def ratio(part: str, whole: str) -> float:
        total = sum(r[whole] for r in results)
        return sum(r[part] for r in results) / total if total else 0.0

    lags = [lag for r in results for lag in r["lags"]]
    waits = [r["final_wait_s"] for r in results]
    return {
        "utterances": len(results),
        "lag_p50_ms": statistics.median(lags) * 1000,
        "lag_p90_ms": float(np.percentile(lags, 90)) * 1000,
        "source_erasure": ratio("source_erased", "letters"),
        "source_dark_dropped": ratio("source_dark_dropped", "source_dark_became"),
        "translation_erasure": ratio("translation_erased", "translation_letters"),
        "translation_dark_dropped": ratio("translation_dark_dropped", "translation_dark_became"),
        "model_use": ratio("model_s", "duration_s"),
        "updates_per_s": ratio("updates", "duration_s"),
        "final_wait_p50_ms": statistics.median(waits) * 1000,
        "final_wait_p90_ms": float(np.percentile(waits, 90)) * 1000,
        "final_text_p50_ms": statistics.median(r["final_text_s"] for r in results) * 1000,
    }


def run(args: argparse.Namespace) -> None:
    silero = Silero()
    model, translator = load_models()
    rows, skipped, pauses, details = [], {}, {}, {}
    for lang in args.langs:
        target = "ko" if lang == "en" else "en"
        began = time.perf_counter()
        chosen, skipped[lang] = utterances(silero, model, translator, lang, args.split, args.count)
        speech_min = sum(len(u["audio"]) for _, u, _ in chosen) / RATE / 60
        pauses[lang] = sum(u["pauses"] for _, u, _ in chosen) / speech_min
        # Warm up both settings on this language before any call is timed.
        for options in PARTIAL.values():
            recognize(model, chosen[0][1]["audio"], lang, options)
        translate(translator, chosen[0][2]["text"], lang, target)
        took = time.perf_counter() - began
        print(f"{lang}: {len(chosen)} clips, skipped {skipped[lang]}, {took:.0f} s", flush=True)
        for partial in args.partials:
            for interval in args.intervals:
                began = time.perf_counter()
                results = [
                    simulate(model, translator, u, final, lang, target, interval / 1000, PARTIAL[partial])
                    for _, u, final in chosen
                ]
                rows.append({"interval_ms": interval, "partial": partial, "lang": lang, **summarize(results)})
                details[f"{partial}_{interval}_{lang}"] = [
                    {"key": key, **r} for (key, _, _), r in zip(chosen, results, strict=True)
                ]
                took = time.perf_counter() - began
                print(f"  {partial} {interval}ms: {took:.0f} s", flush=True)
    write_report(rows, skipped, pauses, details, args)


def write_report(
    rows: list[dict], skipped: dict, pauses: dict, details: dict, args: argparse.Namespace
) -> None:
    gpu, _ = gpu_memory_mb()
    stamp = datetime.now(UTC)
    base = REPORTS / f"stream_{args.tag}_{stamp:%Y%m%d_%H%M%S}"
    report = {"gpu": gpu, "skipped": skipped, "pauses_per_min": pauses, "rows": rows, "details": details}
    base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# 실시간 자막 오프라인 측정 ({args.tag})",
        "",
        f"- 날짜: {stamp.isoformat()}, FLEURS {args.split}, 언어별 {args.count}문장, GPU {gpu}",
        f"- 뺀 문장: {skipped}",
        "- 한 마디 안에서 헛 미리 처리를 부르는 쉼(192~992ms), 소리 1분당: "
        + ", ".join(f"{lang} {value:.1f}" for lang, value in pauses.items()),
        "- 모두 대소문자·띄어쓰기·문장부호를 뺀 글자로 센다. 표시 지연: 최종 단어의 글자가 화면에 모두 "
        "나타나 계속 남게 된 첫 시각 − 소리에서 그 단어가 끝난 시각. 흔들림: 지운 글자 ÷ 최종 글자. "
        "확정 뒤 바뀜: 진하게 보인 뒤 지워진 글자 ÷ 진하게 된 글자. 모델 사용: 호출 시간 ÷ 소리 길이",
        "",
        "| 갱신 간격 | 인식 | 언어 | 문장 | 표시 지연 p50 | p90 | 원문 흔들림 | 원문 확정 뒤 바뀜 | "
        "번역 흔들림 | 번역 확정 뒤 바뀜 | 모델 사용 | 초당 갱신 | 최종 대기 p50/p90 | 최종 글자까지 p50 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['interval_ms']}ms | {r['partial']} | {r['lang']} | {r['utterances']} | "
            f"{r['lag_p50_ms']:.0f}ms | {r['lag_p90_ms']:.0f}ms | {r['source_erasure']:.2f} | "
            f"{r['source_dark_dropped']:.1%} | {r['translation_erasure']:.2f} | "
            f"{r['translation_dark_dropped']:.1%} | {r['model_use']:.2f} | {r['updates_per_s']:.1f} | "
            f"{r['final_wait_p50_ms']:.0f}/{r['final_wait_p90_ms']:.0f}ms | {r['final_text_p50_ms']:.0f}ms |"
        )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run")
    command.add_argument("--split", choices=("validation", "test"), default="validation")
    command.add_argument("--langs", nargs="+", choices=tuple(LANGS), default=list(LANGS))
    command.add_argument("--tag", required=True)
    command.add_argument("--count", type=int, default=PER_LANGUAGE)
    # The test split gets only the chosen combination (docs/experiments.md 8).
    command.add_argument("--intervals", nargs="+", type=int, default=list(INTERVALS_MS))
    command.add_argument("--partials", nargs="+", choices=tuple(PARTIAL), default=list(PARTIAL))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
