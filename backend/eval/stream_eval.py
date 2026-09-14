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
import asyncio
import json
import random
import secrets
import statistics
import time
import uuid
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
        "flags": flags[begin : end + PAD_FRAMES],  # the browser's speech flags, one per frame of the clip
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
    """(dark letters later changed, letters that turned dark) over (letters, dark letters) on screen.

    A dark letter has changed when the next text no longer has it, by aligning the two texts (the rule
    fixed after the first validation, docs/experiments.md 8: a word changed early in the dark part no
    longer counts every dark letter after it). The final result can change dark letters too; its own
    letters are not counted as turning dark."""

    def lost(before: str, dark: int, after: str) -> int:
        return int((~covered(after, before)[:dark]).sum())

    dropped = became = 0
    for (before, dark_before), (after, dark_after) in zip(shown, shown[1:], strict=False):
        changed = lost(before, dark_before, after)
        dropped += changed
        became += max(0, dark_after - (dark_before - changed))
    if shown:
        dropped += lost(*shown[-1], final)
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
        "확정 뒤 바뀜: 진하게 보인 뒤 다음 글에서 (글자 정렬로) 찾을 수 없게 된 글자 ÷ 진하게 된 글자. "
        "모델 사용: 호출 시간 ÷ 소리 길이",
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


FRAME_S = S_CHUNK / RATE
QUIET_END_FRAMES = round(SILENCE_MS / (FRAME_S * 1000))  # 31: the browser decides the end here


async def stream_one(
    ws, http, auth: dict, number: int, utterance: dict, final: dict, pause_frames: int = PAD_FRAMES
) -> dict:
    """Send one clip as the browser would, a 32 ms frame at a time in real time, with its pause, resume and
    end messages, and time what comes back from the clip's start."""
    loop = asyncio.get_running_loop()
    pcm = np.round(utterance["audio"] * 32768).astype("<i2")
    flags = utterance["flags"]
    detected = round(utterance["detected_s"] / FRAME_S)
    messages: list[tuple[float, dict]] = []
    t0 = loop.time()

    async def receive() -> dict:
        while True:
            message = json.loads(await ws.recv())
            messages.append((loop.time() - t0, message))
            if message.get("id") == number and message["type"] in ("final", "error"):
                return message

    async def at_frame(count: int) -> None:
        await asyncio.sleep(max(0.0, t0 + count * FRAME_S - loop.time()))

    def frame(index: int) -> bytes:
        return pcm[index * S_CHUNK : (index + 1) * S_CHUNK].tobytes()

    # As the browser does (frontend/src/conversation/silero.ts): quiet frames past the six after speech are
    # held back and sent only if speech resumes, so at a pause the server has exactly the clip.
    receiver = asyncio.create_task(receive())
    quiet, pauses = 0, 0
    for i, flag in enumerate(flags):
        await at_frame(i + 1)
        if i + 1 < detected:
            continue
        if i + 1 == detected:  # the second speech frame starts the utterance, with the frames before it
            await ws.send(json.dumps({"type": "utterance", "id": number}))
            await ws.send(pcm[: detected * S_CHUNK].tobytes())
            continue
        if flag:
            if quiet >= pause_frames:
                await ws.send(json.dumps({"type": "resume", "id": number}))
            for held in range(i - max(0, quiet - PAD_FRAMES), i):
                await ws.send(frame(held))
            await ws.send(frame(i))
            quiet = 0
        else:
            quiet += 1
            if quiet <= PAD_FRAMES:
                await ws.send(frame(i))
            if quiet == pause_frames:  # the final starts here (T61 candidate Q); the clip ends at six
                await ws.send(json.dumps({"type": "pause", "id": number}))
                pauses += 1
    for j in range(QUIET_END_FRAMES - PAD_FRAMES):  # the quiet up to the end decision stays in the browser
        await at_frame(len(flags) + j + 1)
    await ws.send(json.dumps({"type": "end", "id": number}))
    outcome = await asyncio.wait_for(receiver, 120)
    final_at, audio_at = messages[-1][0], None
    if outcome["type"] == "final" and outcome["result"]["audio_id"]:
        (await http.get(f"/api/audio/{outcome['result']['audio_id']}", headers=auth)).raise_for_status()
        audio_at = loop.time() - t0
    speech_end = final["spans"][-1][2]
    lags = None
    if outcome["type"] == "final" and letters(outcome["result"]["source_text"]) == letters(final["text"]):
        events = [
            (at, letters(m["text"])) for at, m in messages if m["type"] == "source" and m["id"] == number
        ]
        lags = appearance_lags(events + [(final_at, letters(final["text"]))], final["spans"])
    return {
        "outcome": outcome["type"],
        "lags": lags,
        "final_s": final_at - speech_end,
        "audio_s": None if audio_at is None else audio_at - speech_end,
        "pauses": pauses,
        "updates": sum(1 for _, m in messages if m["type"] == "source"),
    }


async def stream_all(args: argparse.Namespace, chosen: dict) -> dict:
    import httpx
    from websockets.asyncio.client import connect

    base = args.base_url.rstrip("/")
    results = {}
    async with httpx.AsyncClient(base_url=base, timeout=60) as http:
        email, password = f"live-{uuid.uuid4().hex[:8]}@example.com", secrets.token_urlsafe(16)
        (
            await http.post("/api/auth/register", json={"email": email, "password": password})
        ).raise_for_status()
        login = await http.post("/api/auth/login", data={"username": email, "password": password})
        token = login.json()["access_token"]
        auth = {"Authorization": f"Bearer {token}"}
        for lang, clips in chosen.items():
            target = "ko" if lang == "en" else "en"
            async with connect("ws" + base.removeprefix("http") + "/api/translate/live", max_size=None) as ws:
                start = {"type": "start", "token": token, "source_lang": lang, "target_lang": target}
                await ws.send(json.dumps(start))
                assert json.loads(await ws.recv())["type"] == "ready"
                results[lang] = []
                for number, (key, utterance, final) in enumerate(clips, 1):
                    result = await stream_one(ws, http, auth, number, utterance, final, args.pause_frames)
                    results[lang].append({"key": key, **result})
                    await asyncio.sleep(0.5)  # a moment between utterances, like playback would take
            print(f"{lang}: {len(results[lang])} clips streamed", flush=True)
    return results


def service(args: argparse.Namespace) -> None:
    """docs/experiments.md 8, real service check: the composed service through nginx, in real time.
    The clips and their word times come from this machine's models first; they are idle while streaming."""
    silero = Silero()
    model, translator = load_models()
    chosen, skipped = {}, {}
    for lang in args.langs:
        chosen[lang], skipped[lang] = utterances(silero, model, translator, lang, args.split, args.count)
    results = asyncio.run(stream_all(args, chosen))
    rows = []
    for lang, items in results.items():
        lags = [lag for item in items if item["lags"] for lag in item["lags"]]
        finals = [item["final_s"] for item in items if item["outcome"] == "final"]
        audio = [item["audio_s"] for item in items if item["audio_s"] is not None]
        rows.append(
            {
                "lang": lang,
                "utterances": len(items),
                "errors": sum(1 for item in items if item["outcome"] != "final"),
                "text_differs": sum(
                    1 for item in items if item["outcome"] == "final" and item["lags"] is None
                ),
                "lag_p50_ms": statistics.median(lags) * 1000,
                "lag_p90_ms": float(np.percentile(lags, 90)) * 1000,
                "final_p50_ms": statistics.median(finals) * 1000,
                "audio_p50_ms": statistics.median(audio) * 1000,
                "audio_p90_ms": float(np.percentile(audio, 90)) * 1000,
                "pauses": sum(item["pauses"] for item in items),
            }
        )
    stamp = datetime.now(UTC)
    base = REPORTS / f"stream_{args.tag}_{stamp:%Y%m%d_%H%M%S}"
    report = {"skipped": skipped, "rows": rows, "details": results}
    base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# 동시통역 실제 서비스 측정 ({args.tag})",
        "",
        f"- 날짜: {stamp.isoformat()}, FLEURS {args.split}, 언어별 {args.count}문장, {args.base_url}",
        f"- 미리 처리를 부르는 쉼: {args.pause_frames}조각({args.pause_frames * 32}ms). "
        "서버 설정은 따로 적는다",
        f"- 뺀 문장: {skipped}",
        "- 표시 지연: 오프라인과 같은 정의(서버 최종 원문이 이 PC의 최종 인식과 글자가 다른 마디는 뺌). "
        "말 끝 → 최종·음성: 마지막 단어 끝부터 final 메시지, 번역 음성 받기 완료까지",
        "",
        "| 언어 | 문장 | 오류 | 원문 다름 | 표시 지연 p50 | p90 | "
        "말 끝→final p50 | 말 끝→음성 p50 | p90 | pause 수 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['lang']} | {r['utterances']} | {r['errors']} | {r['text_differs']} | "
            f"{r['lag_p50_ms']:.0f}ms | {r['lag_p90_ms']:.0f}ms | {r['final_p50_ms']:.0f}ms | "
            f"{r['audio_p50_ms']:.0f}ms | {r['audio_p90_ms']:.0f}ms | {r['pauses']} |"
        )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "service"):
        command = sub.add_parser(name)
        command.add_argument("--split", choices=("validation", "test"), default="validation")
        command.add_argument("--langs", nargs="+", choices=tuple(LANGS), default=list(LANGS))
        command.add_argument("--tag", required=True)
        command.add_argument("--count", type=int, default=PER_LANGUAGE)
    # The test split gets only the chosen combination (docs/experiments.md 8).
    sub.choices["run"].add_argument("--intervals", nargs="+", type=int, default=list(INTERVALS_MS))
    sub.choices["run"].add_argument("--partials", nargs="+", choices=tuple(PARTIAL), default=list(PARTIAL))
    sub.choices["service"].add_argument("--base-url", default="http://localhost:8080")
    # T61 candidate Q: quiet frames before the browser asks for the final (6 = 192 ms, the clip's end).
    sub.choices["service"].add_argument("--pause-frames", type=int, default=PAD_FRAMES)
    args = parser.parse_args()
    {"run": run, "service": service}[args.command](args)


if __name__ == "__main__":
    main()
