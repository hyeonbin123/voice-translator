"""브라우저의 말 끝 판정을 오프라인으로 확인한다(docs/experiments.md 7절).

  uv run python -m eval.eos_eval run --split validation --tag t33_dev
  uv run python -m eval.eos_eval server --split validation --tag t33_dev_server   (docker compose up 필요)

세션: 한 언어의 FLEURS 발화를 각각 같은 음량으로 맞춘 뒤(브라우저 자동 음량 조절 대신), 한 사람이
1.5~3.0초씩 쉬며 이어 말하듯 이어 붙인다. 기준 말 경계는 두 판정 방법과 무관한 Whisper 단어 시각에서 얻고
data/eos/ 아래에 캐시한다. 프레임별 말소리 표시는 쉼 기준과 상관없으므로 판정 방법마다 세션·조건당 한
번만 돌리고, 마디로 묶는 일만 쉼 기준마다 한 번씩 한다.

음량 기준 판정 E는 frontend/src/conversation/endpointer.ts와 같은 계산을 한다. 둘을 함께 바꿔야 한다.
Silero 판정 S는 faster-whisper에 들어 있는 ONNX 파일을 브라우저처럼 한 조각씩 돌린다.
"""

# [파일 역할] 대화 모드의 '말 끝 판정'(브라우저가 한 마디가 끝났다고 보는 규칙)을 오프라인으로 재는 측정
# 도구(docs/experiments.md 7절). run은 판정 방법(E 음량 기준, S Silero VAD) × 쉼 기준(500·700·1000ms) × 잡음
# 조건마다 잘림·붙음·놓침·헛판정·판정 지연을 재고, server는 실제 서비스에 3초·5초 음성을 보내 응답 시간을
# 잰다.
# [설계 이유] 실제 대화 녹음 대신 FLEURS 문장을 한 사람이 이어 말하듯 이어 붙여 세션을 만들고, 기준 말 경계는
# 두 판정 방법과 무관한 Whisper 단어 시각에서 얻는다. 판정 계산은 브라우저
# 코드(frontend/src/conversation/endpointer.ts, Silero ONNX 파일)와 같게 옮겨, 오프라인 결과가 브라우저 동작을
# 대신하게 한다. 측정 결과 Silero VAD 1초 쉼을 골랐다(README: 잘림 2.2~3.1%, 붙음·놓침 0).
# [흐름] FLEURS 읽기·음량 맞추기(load_clips) → Whisper 단어 시각으로 기준 경계(word_bounds, data/eos/에 캐시)
# → 세션 만들기(build_sessions) → 잡음 섞기(with_noise) → 프레임별 말소리 표시(energy_flags / Silero.flags) →
# 쉼 기준별 마디 묶기(group) → 기준과 맞춰 채점(score) → 리포트(write_report).
# [연결] eval/langid_eval.py와 eval/stream_eval.py가 이 파일의 load_clips, Silero, 상수(RATE, LANGS, SEED)를
# 가져다 쓴다.

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

# numpy: 파형·프레임 계산. pyarrow·faster-whisper·onnxruntime은 측정 때만 필요해 쓰는 함수 안에서
# import한다(마디 묶기·채점 함수는 그런 의존성 없이 테스트된다).
import numpy as np

from eval.common import DATA, REPORTS

# 세션·잡음 설정. 쉼 기준 후보 500·700·1000ms, 잡음 조건은 잡음 없음(-60dBFS 바닥 잡음만)·SNR 20dB·10dB 분홍
# 잡음. 세션은 10문장, 문장 사이 쉼은 1.5~3.0초 무작위, 맨 앞 1초 쉼(E의 잡음 수준 측정용). SEED는 분할마다
# 고정해 다시 돌려도 같은 세션이 나온다(validation 17, test 31).
RATE = 16000
LANGS = {"en": "en_us", "ko": "ko_kr"}
SILENCES_MS = (500, 700, 1000)
CONDITIONS = {"clean": None, "snr20": 20.0, "snr10": 10.0}
QUIET_FLOOR_DBFS = -60.0  # 잡음 없음 조건도 조용한 방의 바닥 잡음을 둔다(완전한 디지털 0은 없음)
PEAK_DBFS = -20.0  # 녹음마다 가장 큰 20ms 프레임의 크기(브라우저 자동 음량 조절 대신)
SESSION_UTTERANCES = 10
PAUSE_S = (1.5, 3.0)
LEAD_S = 1.0  # 첫 마디 앞의 쉼(E의 잡음 수준 측정용)
SEED = {"validation": 17, "test": 31}
BOUNDS = DATA / "eos"

# E(음량 기준) 매개변수: 20ms 프레임, 처음 25프레임(0.5초)으로 잡음 수준을 잼, 잡음보다 10dB 넘게 크면 말소리,
# 조용한 프레임으로 잡음 수준을 5%씩 갱신, 말소리 3프레임(60ms) 연속이면 시작.
# S(Silero VAD v6) 매개변수: 512표본(32ms) 조각, 앞 64표본 문맥, 확률 0.5 이상이면 말소리, 2조각(64ms)
# 연속이면 시작.
# 공통: 마디 앞 200ms를 함께 보내고, 말소리 합이 250ms 미만인 덩어리(기침·짧은 잡음)는 버리고, 앞 200ms를
# 포함해 29초가 되면 강제로 끊는다(서버의 30초 제한 안).
# E: 음량 기준 판정(endpointer.ts)
E_FRAME = 320  # 20 ms
E_CALIBRATION_FRAMES = 25  # 잡음 수준 측정에만 쓰고, 그 중앙값이 잡음 수준의 시작값이다
E_MARGIN_DB = 10.0
E_NOISE_RATE = 0.05
E_START_FRAMES = 3  # 말소리 60ms가 이어지면 마디 시작

# S: Silero VAD v6
S_CHUNK = 512  # 32 ms
S_CONTEXT = 64
S_THRESHOLD = 0.5
S_START_FRAMES = 2  # 64 ms

PREROLL_MS = 200  # 첫 말소리 프레임 앞에 함께 남기는 길이
MIN_SPEECH_MS = 250
MAX_CLIP_MS = 29_000  # endpointer.ts처럼 앞 200ms(pre-roll)를 포함해 센다


# 프레임마다 RMS를 dB로(20·log10(RMS + 1e-10)). 1e-10은 완전한 무음에서 log(0)을 피한다. 끝의 남는 표본은
# 버린다.
def frame_db(audio: np.ndarray, frame: int) -> np.ndarray:
    count = len(audio) // frame
    frames = audio[: count * frame].reshape(count, frame).astype(np.float64)
    return 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-10)


# 분홍 잡음(1/f): 흰 잡음의 스펙트럼을 주파수의 제곱근으로 나눠(전력이 1/f) 만든 뒤 RMS가 1이 되게 맞춘다.
# 0Hz에서 0으로 나누지 않도록 첫 주파수를 다음 값으로 바꾼다.
def pink_noise(samples: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(samples))
    freqs = np.fft.rfftfreq(samples)
    freqs[0] = freqs[1]
    noise = np.fft.irfft(spectrum / np.sqrt(freqs), n=samples)
    return (noise / np.sqrt((noise**2).mean())).astype(np.float32)


def load_clips(lang: str, split: str) -> list[tuple[str, np.ndarray]]:
    """(키, 녹음) 목록. 키는 "<parquet 행 번호>-<FLEURS id>"다: 같은 문장을 여러 사람이 읽기 때문이다.

    FLEURS 녹음을 16kHz로 풀고, 가장 큰 20ms 프레임이 -20dBFS가 되도록 음량을 맞춘다(브라우저 자동 음량
    조절 대신).
    """
    # 여기서 import한다: 이것은 eval 그룹(pyarrow)이 필요하지만, 마디 묶기와 지표 계산은 필요 없다.
    import pyarrow.parquet as pq
    from faster_whisper.audio import decode_audio

    table = pq.read_table(DATA / "fleurs" / LANGS[lang] / f"{split}.parquet", columns=["id", "audio"])
    clips = []
    for row_number, row in enumerate(table.to_pylist()):
        audio = decode_audio(io.BytesIO(row["audio"]["bytes"]), sampling_rate=RATE).astype(np.float32)
        gain = 10 ** ((PEAK_DBFS - frame_db(audio, E_FRAME).max()) / 20)
        clips.append((f"{row_number}-{row['id']}", (audio * gain).astype(np.float32)))
    return clips


def word_bounds(clips: list[tuple[str, np.ndarray]], lang: str, split: str) -> dict[str, list[int] | None]:
    """표본 단위의 기준 말 경계: Whisper의 첫 단어 시작 ~ 마지막 단어 끝. 단어가 없으면 None.

    Whisper는 VAD 없이 temperature 0으로만 돌려, 어느 판정 방법도 재시도도 기준에 영향을 주지 않게 한다.
    한 번 계산하면 data/eos/<분할>_<언어>.json에 저장해 다시 쓴다(GPU 시간 절약). 단어가 없는 녹음은
    세션에서 뺀다.
    """
    path = BOUNDS / f"{split}_{lang}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from faster_whisper import WhisperModel

    from app.services.cuda import add_cuda_dll_dirs
    from app.services.stt import TURBO_REVISION

    add_cuda_dll_dirs()  # Windows: nvidia wheel 폴더에 있는 cuBLAS/cuDNN을 CTranslate2가 찾게 한다
    # 서버와 같은 고정 리비전의 모델 파일을 쓴다(app/services/stt.py의 TURBO_REVISION).
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16", revision=TURBO_REVISION)
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
    """세션마다 float32 트랙 하나와 그 안 발화들의 기준 말 경계(표본 단위).

    단어가 있는 녹음을 고정 시드로 섞어 10개씩 묶고, 맨 앞 1초 쉼 + (녹음 + 무작위 쉼)을 이어 붙인 트랙을
    만든다. 각 녹음의 기준 경계는 트랙 안의 위치(cursor)로 옮겨 적는다. 10개가 안 되는 나머지는 버린다.
    """
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


# 트랙에 분홍 잡음을 더한다. 잡음 없음 조건도 -60dBFS 바닥 잡음을 깐다(실제 마이크는 완전한 0을 내지 않음).
# SNR 조건은 기준 말소리 구간의 RMS를 기준으로 잡음 크기를 정한다.
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
    """E: endpointer.ts가 정하는 것과 같은 말소리 프레임. 잡음 측정 프레임은 말소리로 보지 않는다.

    처음 25프레임의 중앙값으로 잡음 수준을 시작하고, 말소리가 아닌 프레임에서만 지수 이동 평균으로 잡음
    수준을 갱신한다(말소리가 잡음 수준을 끌어올리지 않게).
    """
    db = frame_db(audio, E_FRAME)
    flags = np.zeros(len(db), bool)
    noise = float(np.median(db[:E_CALIBRATION_FRAMES]))  # 정렬한 25개 값 중 13번째
    for i in range(E_CALIBRATION_FRAMES, len(db)):
        if db[i] > noise + E_MARGIN_DB:
            flags[i] = True
        else:
            noise = (1 - E_NOISE_RATE) * noise + E_NOISE_RATE * db[i]
    return flags


# faster-whisper에 들어 있는 Silero VAD v6 ONNX 파일을 CPU에서 스레드 하나로 돌린다. 대화 모드는 같은 파일을
# 브라우저에서 onnxruntime-web으로 돌린다(README).
class Silero:
    def __init__(self) -> None:
        import faster_whisper
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        path = Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"
        self.session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def flags(self, audio: np.ndarray) -> np.ndarray:
        """S: 앞 조각의 64표본과 이어받은 상태를 붙여 512표본 조각을 하나씩 처리한다.

        브라우저 스트리밍처럼 앞 조각의 끝 64표본을 문맥으로 붙이고, LSTM 상태(h, c)를 다음 조각으로
        넘긴다. 조각마다 말소리 확률을 구해 0.5 이상이면 말소리로 표시한다.
        """
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
    """브라우저가 자르는 것과 같이 프레임별 표시를 마디로 묶는다.

    마디마다 표본 단위의 말소리 구간(첫 말소리 프레임 ~ 첫 조용한 프레임)과 끝을 판정한 표본 위치를
    돌려준다. 쉼 기준만 바꿔 여러 번 부를 수 있게 프레임별 표시와 분리했다. 시작: 말소리가 start_frames만큼
    연속이면 그 첫 프레임에서 시작(앞 200ms는 있는 만큼). 진행: 말소리면 quiet를 0으로, 아니면 늘린다. 끝:
    조용함이 쉼 기준에 닿거나 앞 200ms 포함 29초가 되면 끝. 말소리 합이 250ms 이상일 때만 마디로 남긴다.
    결과의 end는 말소리 구간의 끝(쉼이 시작된 프레임, 강제로 끊었으면 그 프레임), decided는 판정이 난
    시각이다.
    """
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


# 판정 마디와 기준 경계가 겹치면 대응한다. found: 기준마다 겹친 판정들, covers: 판정마다 겹친 기준 수.
# 잘림 = 판정 둘 이상과 겹친 기준, 놓침 = 겹친 판정이 없는 기준, 붙음 = 한 판정이 기준 여러 개를 덮은 이웃 쌍
# 수, 헛판정 = 어떤 기준과도 겹치지 않는 판정. 판정 지연(ms)은 잘리지도 붙지도 않은(1:1로 대응한) 문장에서만
# 판정 시각 − 기준 끝으로 잰다.
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


# 오프라인 측정 본체. 말소리 표시는 쉼 기준과 상관없으므로 세션·조건·판정 방법마다 한 번만 계산하고, 쉼
# 기준마다 group·score만 다시 한다(위 docstring). 결과는 (방법, 쉼, 조건, 언어)마다 합산한다(지연은 목록으로
# 이어 붙임).
def run(args: argparse.Namespace) -> None:
    silero = Silero()
    detectors = {"E": (energy_flags, E_FRAME, E_START_FRAMES), "S": (silero.flags, S_CHUNK, S_START_FRAMES)}
    # test 분할에는 고른 조합만 적용한다(docs/experiments.md 7절).
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


# 합산 결과를 비율로 바꿔 JSON과 마크다운 표(eval/reports/eos_<태그>_<시각>)로 남긴다. 잘림 비율 = 잘린 문장 ÷
# 문장, 붙음 비율 = 붙은 쌍 ÷ 이웃 쌍, 헛판정은 마디 사이 쉼 1분당.
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


# server 측정에서 보낼 16비트 모노 WAV(범위를 넘는 표본은 잘라 냄).
def wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def server(args: argparse.Namespace) -> None:
    """대화 길이 음성에 대한 서버 응답 시간을 잰다.

    한 발화의 말소리 처음 3초·5초를 그 앞 200ms와 함께 compose로 띄운 서비스(기본 8080)에 보낸다. 한 번
    쓰고 버리는 계정을 만들어 로그인하고(저장하지 않음), 언어마다 무작위 30문장을 음성 번역 API에 보내
    201까지 걸린 벽시계 시간의 중앙값을 남긴다. 비밀번호는 임의로 만들고 어디에도 저장하지 않으며,
    리포트에는 계정 이름의 앞부분만 남긴다.
    """
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
            # 한 조건의 요청이 모두 실패하면 중앙값이 없다(None). 빈 목록의 중앙값을 구하다 멈추면 앞서 잰
            # 조건의 결과도 보고서에 남지 않는다.
            p50 = statistics.median(times) if times else None
            results[key] = {"count": len(times), "failures": failures, "p50_s": p50}
            print(key, results[key], flush=True)
    stamp = datetime.now(UTC)
    path = REPORTS / f"eos_{args.tag}_{stamp:%Y%m%d_%H%M%S}.json"
    path.write_text(
        json.dumps({"account": email.split("@")[0], "results": results}, indent=1), encoding="utf-8"
    )
    print(f"written to {path}")


# 하위 명령 run / server. 둘 다 분할·언어·태그를 받고, run은 판정 방법과 쉼 기준을 고를 수 있다(test에는 고른
# 조합만 적용, docs/experiments.md 7절).
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
