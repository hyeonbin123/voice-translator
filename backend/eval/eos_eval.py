"""T33: offline check of the browser's end-of-speech detection (docs/experiments.md 7).

  uv run python -m eval.eos_eval run --split validation --tag t33_dev
  uv run python -m eval.eos_eval server --split validation --tag t33_dev_server   (needs docker compose up)

Sessions: FLEURS utterances of one language, each brought to the same level (standing in for the browser's
automatic gain control), joined like one speaker talking with 1.5-3.0 s pauses. The reference speech bounds
come from Whisper's word times, independent of both detectors, and are cached under data/eos/.
Per-frame speech flags do not depend on the silence threshold, so each detector runs once per session and
condition, and only the grouping into utterances runs once per threshold.

The energy detector E computes what frontend/src/conversation/endpointer.ts computes; change both together.
The Silero detector S runs the ONNX file faster-whisper ships, one chunk at a time as a browser would.
"""

import argparse
import io
import json
import random
import secrets
import statistics
import time
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from eval.common import DATA, REPORTS

RATE = 16000
LANGS = {"en": "en_us", "ko": "ko_kr"}
SILENCES_MS = (500, 700, 1000)
CONDITIONS = {"clean": None, "snr20": 20.0, "snr10": 10.0}
QUIET_FLOOR_DBFS = -60.0  # the clean condition still has a quiet room floor, never digital zeros
PEAK_DBFS = -20.0  # every clip's loudest 20 ms frame, in place of the browser's automatic gain control
SESSION_UTTERANCES = 10
PAUSE_S = (1.5, 3.0)
LEAD_S = 1.0  # silence before the first utterance, for the energy detector's noise estimate
SEED = {"validation": 17, "test": 31}
BOUNDS = DATA / "eos"

# E: energy detector (endpointer.ts)
E_FRAME = 320  # 20 ms
E_CALIBRATION_FRAMES = 25  # only calibrate; their median starts the noise level
E_MARGIN_DB = 10.0
E_NOISE_RATE = 0.05
E_START_FRAMES = 3  # 60 ms of speech starts an utterance

# S: Silero VAD v6
S_CHUNK = 512  # 32 ms
S_CONTEXT = 64
S_THRESHOLD = 0.5
S_START_FRAMES = 2  # 64 ms

PREROLL_MS = 200  # kept before the first speech frame
MIN_SPEECH_MS = 250
MAX_CLIP_MS = 29_000  # including the pre-roll, as endpointer.ts counts it


def frame_db(audio: np.ndarray, frame: int) -> np.ndarray:
    count = len(audio) // frame
    frames = audio[: count * frame].reshape(count, frame).astype(np.float64)
    return 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-10)


def pink_noise(samples: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(samples))
    freqs = np.fft.rfftfreq(samples)
    freqs[0] = freqs[1]
    noise = np.fft.irfft(spectrum / np.sqrt(freqs), n=samples)
    return (noise / np.sqrt((noise**2).mean())).astype(np.float32)


def load_clips(lang: str, split: str) -> list[tuple[str, np.ndarray]]:
    """(key, clip) with the key "<parquet row>-<FLEURS id>": the same sentence is read by several people."""
    from faster_whisper.audio import decode_audio

    table = pq.read_table(DATA / "fleurs" / LANGS[lang] / f"{split}.parquet", columns=["id", "audio"])
    clips = []
    for row_number, row in enumerate(table.to_pylist()):
        audio = decode_audio(io.BytesIO(row["audio"]["bytes"]), sampling_rate=RATE).astype(np.float32)
        gain = 10 ** ((PEAK_DBFS - frame_db(audio, E_FRAME).max()) / 20)
        clips.append((f"{row_number}-{row['id']}", (audio * gain).astype(np.float32)))
    return clips


def word_bounds(clips: list[tuple[str, np.ndarray]], lang: str, split: str) -> dict[str, list[int] | None]:
    """Reference speech bounds in samples: Whisper's first word start to last word end, or None without words.

    Whisper runs without VAD, at temperature 0 only, so neither detector nor retries shape the reference.
    """
    path = BOUNDS / f"{split}_{lang}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from faster_whisper import WhisperModel

    from app.services.cuda import add_cuda_dll_dirs

    add_cuda_dll_dirs()  # Windows: CTranslate2 finds cuBLAS/cuDNN only in the nvidia wheels' folders
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    bounds: dict[str, list[int] | None] = {}
    for key, clip in clips:
        segments, _ = model.transcribe(
            clip,
            language=lang,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=True,
        )
        words = [word for segment in segments for word in (segment.words or [])]
        bounds[key] = [int(words[0].start * RATE), int(words[-1].end * RATE)] if words else None
    BOUNDS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bounds), encoding="utf-8")
    return bounds


def build_sessions(clips, bounds: dict, split: str, lang: str) -> list[dict]:
    """Each session: one float32 track and the reference speech bounds of its utterances (samples)."""
    usable = [(key, clip) for key, clip in clips if bounds.get(key)]
    rng = random.Random(f"{SEED[split]}-{lang}")
    rng.shuffle(usable)
    sessions = []
    for first in range(0, len(usable) - SESSION_UTTERANCES + 1, SESSION_UTTERANCES):
        cursor = int(LEAD_S * RATE)
        parts, refs, keys = [np.zeros(cursor, np.float32)], [], []
        for key, clip in usable[first : first + SESSION_UTTERANCES]:
            start, end = bounds[key]
            refs.append((cursor + start, cursor + end))
            keys.append(key)
            pause = np.zeros(int(rng.uniform(*PAUSE_S) * RATE), np.float32)
            parts += [clip, pause]
            cursor += len(clip) + len(pause)
        sessions.append({"track": np.concatenate(parts), "refs": refs, "keys": keys})
    return sessions


def with_noise(session: dict, snr_db: float | None, seed: int) -> np.ndarray:
    track = session["track"]
    noise = pink_noise(len(track), np.random.default_rng(seed))
    if snr_db is None:
        level = 10 ** (QUIET_FLOOR_DBFS / 20)
    else:
        speech = np.concatenate([track[start:end] for start, end in session["refs"]]).astype(np.float64)
        level = np.sqrt((speech**2).mean()) / 10 ** (snr_db / 20)
    return (track + noise * level).astype(np.float32)


def energy_flags(audio: np.ndarray) -> np.ndarray:
    """E: speech frames as endpointer.ts decides them. Calibration frames are never speech."""
    db = frame_db(audio, E_FRAME)
    flags = np.zeros(len(db), bool)
    noise = float(np.median(db[:E_CALIBRATION_FRAMES]))  # the 13th of 25 sorted values
    for i in range(E_CALIBRATION_FRAMES, len(db)):
        if db[i] > noise + E_MARGIN_DB:
            flags[i] = True
        else:
            noise = (1 - E_NOISE_RATE) * noise + E_NOISE_RATE * db[i]
    return flags


class Silero:
    def __init__(self) -> None:
        import faster_whisper
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        path = Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"
        self.session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def flags(self, audio: np.ndarray) -> np.ndarray:
        """S: one 512-sample chunk at a time, with the previous 64 samples and the carried state."""
        h = np.zeros((1, 1, 128), np.float32)
        c = np.zeros((1, 1, 128), np.float32)
        context = np.zeros(S_CONTEXT, np.float32)
        count = len(audio) // S_CHUNK
        probs = np.empty(count, np.float32)
        for i in range(count):
            chunk = audio[i * S_CHUNK : (i + 1) * S_CHUNK]
            window = np.concatenate([context, chunk])[None, :]
            out, h, c = self.session.run(None, {"input": window, "h": h, "c": c})
            probs[i] = out.reshape(-1)[0]
            context = chunk[-S_CONTEXT:]
        return probs >= S_THRESHOLD


def group(flags: np.ndarray, frame: int, silence_ms: int, start_frames: int) -> list[dict]:
    """Utterances as the browser cuts them: speech span (first speech frame to first quiet frame) and the
    sample at which the end was decided, in samples."""
    frame_ms = frame * 1000 / RATE
    silence_frames = round(silence_ms / frame_ms)
    preroll_frames = round(PREROLL_MS / frame_ms)
    max_frames = round(MAX_CLIP_MS / frame_ms)
    utterances: list[dict] = []
    run, start, speech, quiet, preroll = 0, None, 0, 0, 0

    for i, flag in enumerate(flags):
        if start is None:
            run = run + 1 if flag else 0
            if run == start_frames:
                start, speech, quiet, run = i - start_frames + 1, start_frames, 0, 0
                preroll = min(preroll_frames, start)
            continue
        if flag:
            speech, quiet = speech + 1, 0
        else:
            quiet += 1
        forced = i + 1 - start + preroll >= max_frames
        if quiet < silence_frames and not forced:
            continue
        if speech * frame_ms >= MIN_SPEECH_MS:
            end = i + 1 if forced else i - quiet + 1
            utterances.append({"start": start * frame, "end": end * frame, "decided": (i + 1) * frame})
        start = None
    return utterances


def score(utterances: list[dict], refs: list[tuple[int, int]]) -> dict:
    found = [[u for u in utterances if u["start"] < end and u["end"] > start] for start, end in refs]
    covers = [sum(1 for start, end in refs if u["start"] < end and u["end"] > start) for u in utterances]
    latencies = []
    for (_, end), parts in zip(refs, found, strict=True):
        if len(parts) == 1 and covers[utterances.index(parts[0])] == 1:
            latencies.append((parts[0]["decided"] - end) * 1000 / RATE)
    return {
        "utterances": len(refs),
        "split": sum(1 for parts in found if len(parts) >= 2),
        "missed": sum(1 for parts in found if not parts),
        "merged_pairs": sum(max(0, count - 1) for count in covers),
        "pairs": len(refs) - 1,
        "false": sum(1 for count in covers if count == 0),
        "latencies_ms": latencies,
    }


def run(args: argparse.Namespace) -> None:
    silero = Silero()
    detectors = {"E": (energy_flags, E_FRAME, E_START_FRAMES), "S": (silero.flags, S_CHUNK, S_START_FRAMES)}
    # The test split gets only the chosen combination (docs/experiments.md 7).
    detectors = {name: detector for name, detector in detectors.items() if name in args.detectors}
    totals: dict = {}
    dropped: dict = {}
    for lang in args.langs:
        started = time.perf_counter()
        clips = load_clips(lang, args.split)
        bounds = word_bounds(clips, lang, args.split)
        dropped[lang] = sum(1 for key, _ in clips if not bounds.get(key))
        sessions = build_sessions(clips, bounds, args.split, lang)
        speech_s = sum(end - start for s in sessions for start, end in s["refs"]) / RATE
        pause_s = sum(len(s["track"]) for s in sessions) / RATE - speech_s
        for condition, snr in CONDITIONS.items():
            for index, session in enumerate(sessions):
                audio = with_noise(session, snr, seed=SEED[args.split] * 1000 + index)
                for name, (detect, frame, start_frames) in detectors.items():
                    flags = detect(audio)
                    for silence in args.silences:
                        result = score(group(flags, frame, silence, start_frames), session["refs"])
                        total = totals.setdefault((name, silence, condition, lang), {"pause_s": pause_s})
                        for field, value in result.items():
                            total[field] = total.get(field, [] if field == "latencies_ms" else 0) + value
        took = time.perf_counter() - started
        print(
            f"{lang}: {len(sessions)} sessions, {dropped[lang]} clips without words, {took:.0f} s", flush=True
        )
    write_report(totals, dropped, args)


def write_report(totals: dict, dropped: dict, args: argparse.Namespace) -> None:
    rows = []
    for (name, silence, condition, lang), t in sorted(totals.items()):
        rows.append(
            {
                "detector": name,
                "silence_ms": silence,
                "condition": condition,
                "lang": lang,
                "utterances": t["utterances"],
                "split_rate": t["split"] / t["utterances"],
                "merge_rate": t["merged_pairs"] / t["pairs"],
                "missed": t["missed"],
                "false_per_min": t["false"] / (t["pause_s"] / 60),
                "latency_p50_ms": statistics.median(t["latencies_ms"]) if t["latencies_ms"] else None,
            }
        )
    stamp = datetime.now(UTC)
    base = REPORTS / f"eos_{args.tag}_{stamp:%Y%m%d_%H%M%S}"
    base.with_suffix(".json").write_text(
        json.dumps({"dropped": dropped, "rows": rows}, indent=1), encoding="utf-8"
    )
    lines = [
        f"# 말 끝 판정 오프라인 측정 ({args.tag})",
        "",
        f"- 날짜: {stamp.isoformat()}, FLEURS {args.split}, 세션당 {SESSION_UTTERANCES}문장, "
        f"문장마다 가장 큰 프레임 {PEAK_DBFS:.0f}dBFS, 기준 경계는 Whisper 단어 시각",
        f"- 단어가 없어 뺀 문장: {dropped}",
        "- 잘림·붙음: 비율, 놓침: 개수, 헛판정: 마디 사이 쉼 1분당, 판정 지연: 잘리지도 붙지도 않은 문장",
        "",
        "| 방법 | 쉼 기준 | 조건 | 언어 | 문장 | 잘림 | 붙음 | 놓침 | 헛판정/분 | 판정 지연 p50 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        latency = f"{r['latency_p50_ms']:.0f}ms" if r["latency_p50_ms"] is not None else "-"
        lines.append(
            f"| {r['detector']} | {r['silence_ms']}ms | {r['condition']} | {r['lang']} | {r['utterances']} | "
            f"{r['split_rate']:.1%} | {r['merge_rate']:.1%} | {r['missed']} | "
            f"{r['false_per_min']:.2f} | {latency} |"
        )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")


def wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def server(args: argparse.Namespace) -> None:
    """Server response time for conversation-length speech: the first 3 s and 5 s of an utterance's speech,
    with the 200 ms before it, through the composed service. Creates a throwaway account (not stored)."""
    import httpx

    client = httpx.Client(base_url=args.base_url, timeout=120)
    email, password = f"eos-{uuid.uuid4().hex[:8]}@example.com", secrets.token_urlsafe(16)
    client.post("/api/auth/register", json={"email": email, "password": password}).raise_for_status()
    login = client.post("/api/auth/login", data={"username": email, "password": password})
    auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
    results = {}
    for lang in args.langs:
        target = "ko" if lang == "en" else "en"
        clips = load_clips(lang, args.split)
        bounds = word_bounds(clips, lang, args.split)
        usable = [(key, clip) for key, clip in clips if bounds.get(key)]
        random.Random(f"server-{SEED[args.split]}-{lang}").shuffle(usable)
        for seconds in (3, 5):
            times, failures = [], 0
            for key, clip in usable[: args.count]:
                start = bounds[key][0]
                piece = clip[max(0, start - PREROLL_MS * RATE // 1000) : start + seconds * RATE]
                files = {"audio": ("utterance.wav", wav_bytes(piece), "audio/wav")}
                form = {"source_lang": lang, "target_lang": target}
                began = time.perf_counter()
                response = client.post("/api/translate/speech", files=files, data=form, headers=auth)
                if response.status_code == 201:
                    times.append(time.perf_counter() - began)
                else:
                    failures += 1
            key = f"{lang}_{seconds}s"
            results[key] = {"count": len(times), "failures": failures, "p50_s": statistics.median(times)}
            print(key, results[key], flush=True)
    stamp = datetime.now(UTC)
    path = REPORTS / f"eos_{args.tag}_{stamp:%Y%m%d_%H%M%S}.json"
    path.write_text(
        json.dumps({"account": email.split("@")[0], "results": results}, indent=1), encoding="utf-8"
    )
    print(f"written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "server"):
        command = sub.add_parser(name)
        command.add_argument("--split", choices=("validation", "test"), default="validation")
        command.add_argument("--langs", nargs="+", choices=tuple(LANGS), default=list(LANGS))
        command.add_argument("--tag", required=True)
    sub.choices["run"].add_argument("--detectors", nargs="+", choices=("E", "S"), default=["E", "S"])
    sub.choices["run"].add_argument("--silences", nargs="+", type=int, default=list(SILENCES_MS))
    sub.choices["server"].add_argument("--base-url", default="http://localhost:8080")
    sub.choices["server"].add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    {"run": run, "server": server}[args.command](args)


if __name__ == "__main__":
    main()
