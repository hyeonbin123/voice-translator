"""T34: offline check of live subtitles while speaking (docs/experiments.md 8).

  uv run python -m eval.stream_eval run --split validation --tag t34_dev
  uv run python -m eval.stream_eval modes --split validation --tag t77_dev   (T77, docs/experiments.md 11)

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
import contextlib
import gc
import hashlib
import itertools
import json
import random
import secrets
import statistics
import time
import uuid
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from app.services.live import common_start, letters, stable_length, wav_from_pcm
from eval.common import DATA, MODELS, PROJECT, REPORTS, gpu_memory_mb
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


def browser_plan(flags, detected: int, pause_frames: int) -> list[tuple[int, str, tuple[int, int] | None]]:
    """What the browser sends for a clip, in order: (frames heard by then, kind, frames of audio).

    As frontend/src/conversation/silero.ts does: the second speech frame starts the utterance with the
    frames before it; quiet frames past the six after speech are held back and sent only if speech
    resumes, so at any pause the server has exactly the clip. The final is asked for after `pause_frames`
    of quiet (T61 candidate Q, at least six), counting on past the clip's end, and the end comes at 31.
    """
    plan: list[tuple[int, str, tuple[int, int] | None]] = []
    quiet = 0
    for i, flag in enumerate(flags):
        heard = i + 1
        if heard < detected:
            continue
        if heard == detected:
            plan += [(heard, "utterance", None), (heard, "audio", (0, heard))]
            continue
        if flag:
            if quiet >= pause_frames:
                plan.append((heard, "resume", None))
            plan.append((heard, "audio", (i - max(0, quiet - PAD_FRAMES), heard)))  # held frames too
            quiet = 0
            continue
        quiet += 1
        if quiet <= PAD_FRAMES:
            plan.append((heard, "audio", (i, heard)))
        if quiet == pause_frames:
            plan.append((heard, "pause", None))
    for heard in range(len(flags) + 1, len(flags) - quiet + QUIET_END_FRAMES + 1):  # quiet past the clip
        quiet += 1
        if quiet == pause_frames:
            plan.append((heard, "pause", None))
    plan.append((len(flags) - PAD_FRAMES + QUIET_END_FRAMES, "end", None))
    return plan


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

    receiver = asyncio.create_task(receive())
    pauses = resumes = 0
    for at, kind, frames in browser_plan(flags, detected, pause_frames):
        await at_frame(at)
        if kind == "audio":
            await ws.send(pcm[frames[0] * S_CHUNK : frames[1] * S_CHUNK].tobytes())
        else:
            await ws.send(json.dumps({"type": kind, "id": number}))
            pauses += kind == "pause"
            resumes += kind == "resume"  # each drops the final prepared at the pause before (T77)
    outcome = await asyncio.wait_for(receiver, 120)
    final_at, audio_at, audio_sha256 = messages[-1][0], None, None
    if outcome["type"] == "final" and outcome["result"]["audio_id"]:
        audio = await http.get(f"/api/audio/{outcome['result']['audio_id']}", headers=auth)
        audio.raise_for_status()
        audio_at = loop.time() - t0
        audio_sha256 = hashlib.sha256(audio.content).hexdigest()
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
        "resumes": resumes,
        "result": outcome.get("result"),
        "detail": outcome.get("detail"),
        "audio_sha256": audio_sha256,
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


class GpuLog:
    """GPU use while streaming, sampled twice a second (T63). The server's own work shows in it too: what
    tells of another program (a game on the same GPU) is use that stays high between utterances."""

    def __enter__(self) -> GpuLog:
        import threading

        import psutil
        import pynvml

        self.nvml = pynvml
        self.psutil = psutil
        self.cpu: list[float] = []
        psutil.cpu_percent(None)  # the first call only starts the count
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples: list[tuple[float, float]] = []
        self.at_start = set(self.processes())
        self.programs: list[str] = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self) -> None:
        while not self.stopped.wait(0.5):
            use = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.samples.append((use.gpu, memory.used / 2**20))
            self.cpu.append(self.psutil.cpu_percent(None))  # the whole machine, since the last sample

    def processes(self) -> list[str]:
        import psutil

        names = []
        for query in ("nvmlDeviceGetGraphicsRunningProcesses", "nvmlDeviceGetComputeRunningProcesses"):
            for process in getattr(self.nvml, query)(self.handle):
                try:
                    names.append(psutil.Process(process.pid).name())
                except (psutil.Error, OSError):
                    names.append(f"pid {process.pid}")
        return names

    def __exit__(self, *exc) -> None:
        self.stopped.set()
        self.thread.join()
        self.programs = sorted(self.at_start | set(self.processes()))

    def summary(self) -> dict:
        use = [u for u, _ in self.samples] or [0.0]
        return {
            "samples": len(self.samples),
            "mean_util": float(np.mean(use)),
            "p90_util": float(np.percentile(use, 90)),
            "max_util": float(max(use)),
            "max_memory_mb": max((m for _, m in self.samples), default=0.0),
            "mean_cpu": float(np.mean(self.cpu)) if self.cpu else 0.0,
        }


def service(args: argparse.Namespace) -> None:
    """docs/experiments.md 8, real service check: the composed service through nginx, in real time.
    The clips and their word times come from this machine's models first; they are idle while streaming."""
    silero = Silero()
    model, translator = load_models()
    chosen, skipped = {}, {}
    for lang in args.langs:
        chosen[lang], skipped[lang] = utterances(silero, model, translator, lang, args.split, args.count)
    with GpuLog() as gpu:
        results = asyncio.run(stream_all(args, chosen))
    # Other programs' names stay out of the published report: work/ is not committed.
    (PROJECT / "work" / f"gpu_programs_{args.tag}.txt").write_text("\n".join(gpu.programs), encoding="utf-8")
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
    report = {"skipped": skipped, "gpu_use": gpu.summary(), "rows": rows, "details": results}
    base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# 동시통역 실제 서비스 측정 ({args.tag})",
        "",
        f"- 날짜: {stamp.isoformat()}, FLEURS {args.split}, 언어별 {args.count}문장, {args.base_url}",
        f"- 미리 처리를 부르는 쉼: {args.pause_frames}조각({args.pause_frames * 32}ms). "
        "서버 설정은 따로 적는다",
        "- GPU (측정 중 0.5초마다, 이 서버의 사용 포함): 사용률 평균 {mean_util:.0f}%, p90 {p90_util:.0f}%, "
        "최대 {max_util:.0f}%, 메모리 최대 {max_memory_mb:.0f}MB".format(**gpu.summary()),
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


# T77 (docs/experiments.md 11): conversation and dialog modes, each clip once over the old path (the browser
# uploads the WAV over HTTP after the 1000 ms end) and once over the new one (streamed over the live
# WebSocket without subtitles, prepared at the 192 ms pause, committed at the end), back to back.

MODES = ("conversation", "dialog")
OTHER = {"ko": "en", "en": "ko"}
SAVING_MS = 300  # the rule: both directions' p50 at least this much sooner
UNDECIDABLE_LIMIT = 2  # clips per direction whose HTTP result itself changes from call to call
SECOND_USER_EVERY_S = 5.0
# What a clip's result must share over both paths. The synthesized audio itself cannot be compared:
# MeloTTS samples noise (and durations) at every call, so the same text gives different WAV bytes.
SAME_FIELDS = (
    "source_lang",
    "target_lang",
    "source_text",
    "translated_text",
    "stt_model",
    "mt_model",
    "tts_model",
    "tts_error",
    "language_guessed",
)


def clip_wav(audio: np.ndarray) -> bytes:
    """The WAV the browser uploads for a clip (frontend pcm.ts encodeWav): the same file the server builds
    from the streamed PCM (app/services/live.py)."""
    return wav_from_pcm(np.round(audio * 32768).astype("<i2").tobytes())


def end_frame(utterance: dict) -> int:
    """Frames heard when the browser decides the end (1000 ms of quiet), as browser_plan sends it."""
    detected = round(utterance["detected_s"] / FRAME_S)
    return browser_plan(utterance["flags"], detected, PAD_FRAMES)[-1][0]


def comparable(outcome: dict) -> dict:
    """What must be the same over both paths: the error, or the result's texts, languages and models."""
    if outcome["outcome"] != "final":
        return {"outcome": outcome["outcome"], "detail": outcome.get("detail")}
    result = outcome["result"]
    return {
        "outcome": "final",
        "has_audio": result.get("audio_id") is not None,
        **{key: result.get(key) for key in SAME_FIELDS},
    }


def same(a: dict, b: dict) -> bool:
    return comparable(a) == comparable(b)


def classify(upload: dict, prepared: dict, rechecks: list[dict]) -> str:
    """'same', or for a clip whose paths differ: 'different' when two more uploads of its WAV give the
    first upload's result again (the prepared path changed it), else 'undecidable' (the HTTP path itself
    gives other results for the same WAV: model nondeterminism, not the prepare)."""
    if same(upload, prepared):
        return "same"
    if len(rechecks) == 2 and all(same(upload, again) for again in rechecks):
        return "different"
    return "undecidable"


def paired_ci(differences: list[float], seed: int = 77, rounds: int = 10_000) -> tuple[float, float, float]:
    """The median of per-clip differences and its 95% percentile bootstrap interval."""
    values = np.asarray(differences, dtype=float)
    draws = np.random.default_rng(seed).choice(values, size=(rounds, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.median(values)), float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def quantile_ms(values: list[float], q: float) -> float:
    """A percentile in ms, or NaN when a path produced no audio at all (it then fails the rule)."""
    return float(np.percentile(values, q)) * 1000 if values else float("nan")


def summarize_modes(rows: list[dict]) -> list[dict]:
    """One row per mode and source language."""
    out = []
    for mode in MODES:
        for lang in LANGS:
            items = [r for r in rows if r["mode"] == mode and r["lang"] == lang]
            if not items:
                continue
            upload = [r["upload"]["audio_s"] for r in items if r["upload"]["audio_s"] is not None]
            prepared = [r["prepared"]["audio_s"] for r in items if r["prepared"]["audio_s"] is not None]
            pairs = [
                r["upload"]["audio_s"] - r["prepared"]["audio_s"]
                for r in items
                if r["upload"]["audio_s"] is not None and r["prepared"]["audio_s"] is not None
            ]
            median, low, high = paired_ci(pairs) if pairs else (float("nan"),) * 3
            classes = [r["identity"] for r in items]
            out.append(
                {
                    "mode": mode,
                    "lang": lang,
                    "clips": len(items),
                    "upload_p50_ms": quantile_ms(upload, 50),
                    "upload_p90_ms": quantile_ms(upload, 90),
                    "prepared_p50_ms": quantile_ms(prepared, 50),
                    "prepared_p90_ms": quantile_ms(prepared, 90),
                    "saving_ms": quantile_ms(upload, 50) - quantile_ms(prepared, 50),
                    "paired_median_ms": median * 1000,
                    "paired_ci_ms": (low * 1000, high * 1000),
                    "upload_final_p50_ms": quantile_ms([r["upload"]["final_s"] for r in items], 50),
                    "prepared_final_p50_ms": quantile_ms([r["prepared"]["final_s"] for r in items], 50),
                    "same": classes.count("same"),
                    "different": classes.count("different"),
                    "undecidable": classes.count("undecidable"),
                    "audio_bytes_same": sum(
                        1
                        for r in items
                        if r["upload"].get("audio_sha256")
                        and r["upload"].get("audio_sha256") == r["prepared"].get("audio_sha256")
                    ),
                    "errors_upload": sum(1 for r in items if r["upload"]["outcome"] != "final"),
                    "errors_prepared": sum(1 for r in items if r["prepared"]["outcome"] != "final"),
                    "prepares": sum(r["prepared"]["pauses"] for r in items),
                    "dropped": sum(r["prepared"]["resumes"] for r in items),
                }
            )
    return out


def decide(rows: list[dict]) -> dict[str, dict]:
    """The rule written before measuring (docs/experiments.md 11), for the parts this tool measures: per
    mode, adopt when in both directions the upload path's end-to-audio p50 is at least 300 ms later than
    the prepared path's, no clip is 'different' and at most two are 'undecidable'. The e2e_eval regression
    check is the rule's third part and is judged from its own report."""
    verdict = {}
    for mode in MODES:
        mine = [r for r in rows if r["mode"] == mode]
        if not mine:
            continue
        reasons = []
        if {r["lang"] for r in mine} != set(LANGS):
            reasons.append("not both directions")
        for r in mine:
            if not r["saving_ms"] >= SAVING_MS:
                reasons.append(f"{r['lang']}: p50 {r['saving_ms']:.0f}ms sooner, under {SAVING_MS}ms")
            if r["different"]:
                reasons.append(f"{r['lang']}: {r['different']} clips differ")
            if r["undecidable"] > UNDECIDABLE_LIMIT:
                reasons.append(f"{r['lang']}: {r['undecidable']} undecidable clips, over {UNDECIDABLE_LIMIT}")
        verdict[mode] = {"adopt": not reasons, "reasons": reasons}
    return verdict


async def upload_one(http, auth: dict, mode: str, lang: str, utterance: dict, final: dict, previous) -> dict:
    """The old path for one clip: nothing is sent until the browser decides the end, then the WAV goes to
    the HTTP API and the translated speech is fetched. Times from the clip's start, as stream_one's."""
    loop = asyncio.get_running_loop()
    wav = clip_wav(utterance["audio"])
    t0 = loop.time()
    await asyncio.sleep(max(0.0, t0 + end_frame(utterance) * FRAME_S - loop.time()))
    sent_at = loop.time() - t0
    files = {"audio": ("conversation.wav", wav, "audio/wav")}
    if mode == "conversation":
        path, data = "/api/translate/speech", {"source_lang": lang, "target_lang": OTHER[lang]}
    else:
        path, data = "/api/translate/dialog", ({"previous_lang": previous} if previous else {})
    reply = await http.post(path, headers=auth, files=files, data=data)
    final_at = loop.time() - t0
    speech_end = final["spans"][-1][2]
    out: dict = {"sent_s": sent_at - speech_end, "final_s": final_at - speech_end, "audio_s": None}
    if reply.status_code != 201:
        try:
            detail = reply.json().get("detail")
        except ValueError:
            detail = None
        return out | {"outcome": "error", "status": reply.status_code, "detail": detail}
    result = reply.json()
    out |= {"outcome": "final", "result": result}
    if result["audio_id"]:
        audio = await http.get(f"/api/audio/{result['audio_id']}", headers=auth)
        audio.raise_for_status()
        out["audio_s"] = loop.time() - t0 - speech_end
        out["audio_sha256"] = hashlib.sha256(audio.content).hexdigest()
    return out


async def reupload(http, auth: dict, mode: str, lang: str, utterance: dict, previous) -> dict:
    """The same clip over HTTP once more, at once, for the identity check of a clip whose paths differ."""
    files = {"audio": ("conversation.wav", clip_wav(utterance["audio"]), "audio/wav")}
    if mode == "conversation":
        path, data = "/api/translate/speech", {"source_lang": lang, "target_lang": OTHER[lang]}
    else:
        path, data = "/api/translate/dialog", ({"previous_lang": previous} if previous else {})
    reply = await http.post(path, headers=auth, files=files, data=data)
    if reply.status_code != 201:
        return {"outcome": "error", "detail": reply.json().get("detail")}
    return {"outcome": "final", "result": reply.json()}


async def account(http, prefix: str) -> tuple[str, dict]:
    email, password = f"{prefix}-{uuid.uuid4().hex[:8]}@example.com", secrets.token_urlsafe(16)
    (await http.post("/api/auth/register", json={"email": email, "password": password})).raise_for_status()
    login = await http.post("/api/auth/login", data={"username": email, "password": password})
    token = login.raise_for_status().json()["access_token"]
    return token, {"Authorization": f"Bearer {token}"}


async def watch_health(http, phase: list[str], out: list[tuple[str, float]], done: asyncio.Event) -> None:
    """GET /api/health every 0.2 s (T23's check that model work does not hold other requests)."""
    while not done.is_set():
        label, began = phase[0], time.perf_counter()
        try:
            reply = await http.get("/api/health")
            took = time.perf_counter() - began if reply.status_code == 200 else float("inf")
        except Exception:  # noqa: BLE001 - a failed check counts as an endless one
            took = float("inf")
        out.append((label, took))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(done.wait(), 0.2)


async def second_user(http, auth: dict, split: str, phase: list[str], out: list[dict], done: asyncio.Event):
    """Another person on the same server: an 8-12 s recording to the HTTP speech API every 5 s."""
    folder = DATA / "e2e" / split
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    loop = asyncio.get_running_loop()
    number = 0
    while not done.is_set():
        item = manifest[number % len(manifest)]
        number += 1
        label, began = phase[0], loop.time()
        files = {"audio": (item["file"], (folder / item["file"]).read_bytes(), "audio/wav")}
        data = {"source_lang": item["language"], "target_lang": OTHER[item["language"]]}
        reply = await http.post("/api/translate/speech", headers=auth, files=files, data=data)
        took = loop.time() - began
        body = reply.json() if reply.status_code == 201 else {}
        out.append({"phase": label, "elapsed_s": took, "status": reply.status_code, "id": body.get("id")})
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(done.wait(), max(0.0, SECOND_USER_EVERY_S - took))


def turns(mode: str, chosen: dict) -> list[tuple[str, list]]:
    """Connections and their clips: conversation mode one per direction, dialog one with the two languages
    taking turns, as two people would."""
    if mode == "conversation":
        return [(lang, [(key, lang, u, f) for key, u, f in chosen[lang]]) for lang in chosen]
    lists = [[(key, lang, u, f) for key, u, f in chosen[lang]] for lang in chosen]
    mixed = [item for group in itertools.zip_longest(*lists) for item in group if item is not None]
    return [("dialog", mixed)]


async def run_modes(args: argparse.Namespace, chosen: dict) -> tuple[list[dict], dict]:
    import httpx
    from websockets.asyncio.client import connect

    base = args.base_url.rstrip("/")
    phase = ["idle"]
    health: list[tuple[str, float]] = []
    others: list[dict] = []
    rows: list[dict] = []
    async with httpx.AsyncClient(base_url=base, timeout=120) as http:
        token, auth = await account(http, "modes")
        other = auth  # the second user's account, when there is one
        done = asyncio.Event()
        watchers = [asyncio.create_task(watch_health(http, phase, health, done))]
        if args.second_user:
            _, other = await account(http, "second")
            watchers.append(asyncio.create_task(second_user(http, other, args.split, phase, others, done)))
        for mode in args.modes:
            for name, items in turns(mode, chosen):
                start = {"type": "start", "token": token, "mode": mode}
                if mode == "conversation":
                    start |= {"source_lang": name, "target_lang": OTHER[name]}
                async with connect(
                    "ws" + base.removeprefix("http") + "/api/translate/live", max_size=None
                ) as ws:
                    await ws.send(json.dumps(start))
                    assert json.loads(await ws.recv())["type"] == "ready"
                    previous = None  # dialog over HTTP: the browser's last answered language
                    for number, (key, lang, utterance, final) in enumerate(items, 1):
                        row = {"mode": mode, "lang": lang, "key": key, "number": number, "previous": previous}
                        # Alternate which path goes first, so a slow stretch of time hits both alike.
                        for path in ("upload", "prepared") if number % 2 else ("prepared", "upload"):
                            phase[0] = f"{mode}:{path}"
                            if path == "upload":
                                row[path] = await upload_one(
                                    http, auth, mode, lang, utterance, final, previous
                                )
                            else:
                                row[path] = await stream_one(ws, http, auth, number, utterance, final)
                            phase[0] = "idle"
                            await asyncio.sleep(0.5)  # a moment between utterances, like playback would take
                        if mode == "dialog" and row["upload"]["outcome"] == "final":
                            previous = row["upload"]["result"]["source_lang"]
                        rows.append(row)
                        print(f"  {mode} {lang} {number}/{len(items)}", flush=True)  # a hang shows here
                print(f"{mode} {name}: {len(items)} clips", flush=True)
        done.set()
        await asyncio.gather(*watchers)
        # Clips whose paths differ: the same WAV over HTTP twice more (the identity check's rule).
        clips = {(lang, key): utterance for lang in chosen for key, utterance, _ in chosen[lang]}
        for row in rows:
            row["rechecks"] = []
            if not same(row["upload"], row["prepared"]):
                for _ in range(2):
                    clip = clips[(row["lang"], row["key"])]
                    row["rechecks"].append(
                        await reupload(http, auth, row["mode"], row["lang"], clip, row["previous"])
                    )
            row["identity"] = classify(row["upload"], row["prepared"], row["rechecks"])
        # The two throwaway accounts' records go; the accounts stay in the database (as T58's do).
        ids = [
            o["result"]["id"]
            for r in rows
            for o in (r["upload"], r["prepared"], *r["rechecks"])
            if o.get("result")
        ]
        for record in ids:
            await http.delete(f"/api/history/{record}", headers=auth)
        for item in others:
            if item["id"]:
                await http.delete(f"/api/history/{item['id']}", headers=other)
    return rows, {"health": health, "second_user": others}


def phase_stats(samples: list[tuple[str, float]]) -> dict[str, dict]:
    stats = {}
    for label in sorted({label for label, _ in samples} - {"idle"}):
        values = [value for name, value in samples if name == label]
        stats[label] = {
            "count": len(values),
            "p95_s": float(np.percentile(values, 95)),
            "max_s": float(max(values)),
        }
    return stats


def modes(args: argparse.Namespace) -> None:
    """docs/experiments.md 11: the composed service through nginx, in real time, both paths per clip."""
    silero = Silero()
    model, translator = load_models()
    chosen, skipped = {}, {}
    for lang in args.langs:
        chosen[lang], skipped[lang] = utterances(silero, model, translator, lang, args.split, args.count)
    # The clips and their word times are ready: free this process's GPU memory before the service runs.
    del model, translator
    gc.collect()
    with GpuLog() as gpu:
        rows, watched = asyncio.run(run_modes(args, chosen))
    (PROJECT / "work" / f"gpu_programs_{args.tag}.txt").write_text("\n".join(gpu.programs), encoding="utf-8")
    write_modes_report(args, skipped, gpu.summary(), rows, watched, REPORTS)


def write_modes_report(
    args: argparse.Namespace, skipped: dict, gpu_use: dict, rows: list[dict], watched: dict, folder: Path
) -> Path:
    table = summarize_modes(rows)
    verdict = decide(table)
    health = phase_stats(watched["health"])
    others = {}
    for label in sorted({o["phase"] for o in watched["second_user"]} - {"idle"}):
        values = [
            o["elapsed_s"] for o in watched["second_user"] if o["phase"] == label and o["status"] == 201
        ]
        if values:
            others[label] = {
                "count": len(values),
                "p50_s": statistics.median(values),
                "p90_s": float(np.percentile(values, 90)),
            }
    stamp = datetime.now(UTC)
    base = folder / f"stream_{args.tag}_{stamp:%Y%m%d_%H%M%S}"
    report = {
        "split": args.split,
        "skipped": skipped,
        "gpu_use": gpu_use,
        "rows": table,
        "verdict": verdict,
        "health": health,
        "second_user": others,
        "details": rows,
        "watched": watched,
    }
    base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# 대화 모드·두 사람 대화 미리 처리 측정 ({args.tag})",
        "",
        f"- 날짜: {stamp.isoformat()}, FLEURS {args.split}, 언어별 {args.count}문장, {args.base_url}, "
        f"두 번째 사용자 {'있음' if args.second_user else '없음'}",
        "- 경로: 업로드 = 1초 판정 뒤 HTTP로 WAV(지금까지의 화면), "
        "미리 처리 = 웹소켓으로 흘리고 192ms 쉼에서 미리 처리, 1초에서 확정. 문장마다 두 경로를 번갈아 먼저",
        "- GPU (0.5초마다, 서버 포함): 사용률 평균 {mean_util:.0f}%, p90 {p90_util:.0f}%, "
        "최대 {max_util:.0f}%, 메모리 최대 {max_memory_mb:.0f}MB, CPU 평균 {mean_cpu:.0f}%".format(**gpu_use),
        f"- 뺀 문장: {skipped}",
        "- 말 끝 → 음성: 마지막 단어 끝부터 번역 음성 받기 완료까지. "
        "차이 중앙값은 문장별 (업로드 − 미리 처리)의 중앙값과 95% 붓스트랩 구간",
        "- 같음: 원문·번역문·언어·모델·합성 오류·추정 여부·음성 유무가 같음. "
        "다르면 같은 WAV를 HTTP로 두 번 더 보내 업로드 결과가 되풀이되면 '다름', "
        "아니면 '판단 불가'(모델 비결정성)",
        "",
        "| 모드 | 원문 | 문장 | 업로드 p50 | p90 | 미리 처리 p50 | p90 | p50 차이 | "
        "차이 중앙값 (95% 구간) | 같음/다름/판단 불가 | 음성 바이트 같음 | 오류 (업/미) | "
        "미리 처리 / 버림 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def ms(value: float) -> str:
        return f"{value:.0f}ms"

    for r in table:
        low, high = r["paired_ci_ms"]
        cells = [
            r["mode"],
            r["lang"],
            r["clips"],
            ms(r["upload_p50_ms"]),
            ms(r["upload_p90_ms"]),
            ms(r["prepared_p50_ms"]),
            ms(r["prepared_p90_ms"]),
            ms(r["saving_ms"]),
            f"{ms(r['paired_median_ms'])} ({low:.0f}~{high:.0f})",
            f"{r['same']}/{r['different']}/{r['undecidable']}",
            r["audio_bytes_same"],
            f"{r['errors_upload']}/{r['errors_prepared']}",
            f"{r['prepares']} / {r['dropped']}",
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines += [
        "",
        "| 구간 | health 수 | p95 | 최대 | 두 번째 사용자 p50 | p90 | 수 |",
        "|---|---|---|---|---|---|---|",
    ]
    nan = float("nan")
    for label in sorted(set(health) | set(others)):
        h, o = health.get(label, {}), others.get(label, {})
        cells = [
            label,
            h.get("count", 0),
            f"{h.get('p95_s', nan):.3f}s",
            f"{h.get('max_s', nan):.3f}s",
            f"{o.get('p50_s', nan):.2f}s",
            f"{o.get('p90_s', nan):.2f}s",
            o.get("count", 0),
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines += ["", "규칙의 시간·동일성 부분 (e2e 회귀는 e2e_eval 보고서로 따로 본다):"]
    for mode, v in verdict.items():
        lines.append(
            f"- {mode}: " + ("채택 조건 충족" if v["adopt"] else "불충족: " + "; ".join(v["reasons"]))
        )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")
    return base.with_suffix(".md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "service", "modes"):
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
    sub.choices["modes"].add_argument("--base-url", default="http://localhost:8080")
    sub.choices["modes"].add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    # T23's interference check: another person's HTTP requests during both paths (report only).
    sub.choices["modes"].add_argument("--second-user", action="store_true")
    args = parser.parse_args()
    {"run": run, "service": service, "modes": modes}[args.command](args)


if __name__ == "__main__":
    main()
