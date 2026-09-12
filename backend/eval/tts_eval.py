"""Speech synthesis evaluation on FLEURS sentences (task T4).

Two phases, so that candidates with conflicting dependencies (MeloTTS pins transformers 4.27) run in their
own environments but are scored by the same speech recognition model. From backend/:

1. Synthesize, with the candidate's environment:
       <that environment's python> -m eval.tts_eval synth --candidate mms --tag t4_dev
   writes data/tts_audio/<tag>/<candidate>/<language>/<id>.wav (git-ignored) and
   eval/reports/tts_<tag>_<candidate>_synth.json (timing and VRAM per sentence).
2. Score, with the backend environment (`uv sync --group gpu --group eval`):
       uv run python -m eval.tts_eval score --tag t4_dev
   re-recognizes every saved wav with large-v3-turbo and writes eval/reports/tts_<tag>_<stamp>.md/.json.

The candidates and the selection rule are written in docs/experiments.md before any run.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
import wave
from datetime import UTC, datetime

import pyarrow.parquet as pq

from eval.common import DATA, FLEURS_CONFIG, REPORTS
from eval.common import gpu_memory_mb as device_memory

AUDIO = DATA / "tts_audio"  # git-ignored with the rest of data/


def sentences(language: str, split: str, limit: int | None) -> list[dict]:
    """One row per sentence ID, in ID order: the text to read and the transcription to score against."""
    path = DATA / "fleurs" / FLEURS_CONFIG[language] / f"{split}.parquet"
    rows = pq.read_table(path, columns=["id", "raw_transcription", "transcription"]).to_pylist()
    unique = sorted({row["id"]: row for row in rows}.values(), key=lambda row: row["id"])
    return unique[:limit] if limit else unique


def gpu_memory_mb() -> float | None:
    """GPU memory in use on the device, or None when nvidia-ml-py isn't in this environment."""
    try:
        return device_memory()[1]
    except ImportError:
        return None


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as clip:
        return clip.getnframes() / clip.getframerate()


def synth(args: argparse.Namespace) -> None:
    from eval.tts_candidates import CANDIDATES

    load, languages = CANDIDATES[args.candidate]
    record: dict = {"candidate": args.candidate, "split": args.split, "languages": {}}
    for language in languages:
        before = gpu_memory_mb()
        model = load(language)
        after = gpu_memory_mb()
        rows = sentences(language, args.split, args.limit)
        model.synthesize(rows[0]["raw_transcription"], language)  # warm-up, not timed
        out = AUDIO / args.tag / args.candidate / language
        out.mkdir(parents=True, exist_ok=True)
        items, failures = [], 0
        for row in rows:
            start = time.perf_counter()
            try:
                audio = model.synthesize(row["raw_transcription"], language)
            except Exception as exc:  # noqa: BLE001 - count every failure and keep going
                failures += 1
                items.append({"id": row["id"], "failure": f"{type(exc).__name__}: {exc}"})
                continue
            elapsed = time.perf_counter() - start
            (out / f"{row['id']}.wav").write_bytes(audio.wav)
            seconds = wav_seconds(audio.wav)
            items.append(
                {
                    "id": row["id"],
                    "synth_s": round(elapsed, 3),
                    "audio_s": round(seconds, 2),
                    "per_audio_second": round(elapsed / seconds, 4) if seconds else None,
                }
            )
        record["languages"][language] = {
            "model_name": model.model_name,
            "vram_mb": round(after - before) if before is not None and after is not None else None,
            "failures": failures,
            "items": items,
        }
        del model
    path = REPORTS / f"tts_{args.tag}_{args.candidate}_synth.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written to {path}")


def score(args: argparse.Namespace) -> None:
    from app.services.interfaces import InvalidAudioError
    from app.services.stt import WhisperSpeechToText
    from eval.stt_eval import error_rate
    from eval.text_norm import normalize

    stt = WhisperSpeechToText("large-v3-turbo")
    results = []
    for synth_file in sorted(REPORTS.glob(f"tts_{args.tag}_*_synth.json")):
        record = json.loads(synth_file.read_text(encoding="utf-8"))
        result: dict = {"candidate": record["candidate"], "languages": {}}
        for language, values in record["languages"].items():
            rows = {row["id"]: row for row in sentences(language, record["split"], None)}
            audio_dir = AUDIO / args.tag / record["candidate"] / language
            references, hypotheses = [], []
            for item in values["items"]:
                reference = normalize(rows[item["id"]]["transcription"], language)
                text = ""  # a failed synthesis or unrecognizable audio counts as fully wrong
                if "failure" not in item:
                    try:
                        text = stt.transcribe((audio_dir / f"{item['id']}.wav").read_bytes(), language).text
                    except InvalidAudioError:
                        pass
                hypothesis = normalize(text, language)
                item.update(reference=reference, hypothesis=hypothesis)
                item["error_rate"] = round(error_rate(language, [reference], [hypothesis]), 4)
                references.append(reference)
                hypotheses.append(hypothesis)
            speeds = [item["per_audio_second"] for item in values["items"] if item.get("per_audio_second")]
            synth_times = [item["synth_s"] for item in values["items"] if "synth_s" in item]
            result["languages"][language] = {
                "model_name": values["model_name"],
                "metric": "CER" if language == "ko" else "WER",
                "error": error_rate(language, references, hypotheses),
                "speed_p50": statistics.median(speeds),
                "speed_p95": statistics.quantiles(speeds, n=20)[18],
                "synth_s_p50": statistics.median(synth_times),
                "vram_mb": values["vram_mb"],
                "failures": values["failures"],
                "count": len(values["items"]),
                "items": values["items"],
            }
        results.append(result)
    write_report(results, args)


def write_report(results: list[dict], args: argparse.Namespace) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = REPORTS / f"tts_{args.tag}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [
        f"# 음성 합성 평가 ({args.tag})",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}",
        "- 재인식: faster-whisper large-v3-turbo (T2에서 고른 모델), 정규화는 eval/text_norm.py",
        "- 속도: 생성 음성 1초당 합성 시간(초). 언어별 첫 호출 제외",
    ]
    for language, title in (("ko", "한국어 (재인식 CER)"), ("en", "영어 (재인식 WER)")):
        lines += ["", f"## {title}", ""]
        lines.append("| 후보 | 오류율 | 속도 p50 | 속도 p95 | 문장당 합성 p50 | VRAM (MB) | 실패 |")
        lines.append("|---|---|---|---|---|---|---|")
        for result in results:
            if values := result["languages"].get(language):
                lines.append(
                    f"| {result['candidate']} | {values['error']:.2%} | {values['speed_p50']:.3f} "
                    f"| {values['speed_p95']:.3f} | {values['synth_s_p50']:.3f}초 | {values['vram_mb']} "
                    f"| {values['failures']} |"
                )
    for result in results:
        for language, values in result["languages"].items():
            lines += ["", f"## {result['candidate']} / {language}: 오류가 큰 5개", ""]
            for item in sorted(values["items"], key=lambda i: i["error_rate"], reverse=True)[:5]:
                lines.append(
                    f"- id {item['id']} ({item['error_rate']:.0%}): "
                    f"참조 `{item['reference']}` / 재인식 `{item['hypothesis']}`"
                )
    report = base.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    synth_parser = phases.add_parser("synth", help="synthesize with one candidate")
    synth_parser.add_argument("--candidate", required=True)
    synth_parser.add_argument("--split", choices=["validation", "test"], default="validation")
    synth_parser.add_argument("--limit", type=int, default=None, help="first N sentences per language")
    synth_parser.add_argument("--tag", default="run")
    score_parser = phases.add_parser("score", help="re-recognize every candidate's audio for a tag")
    score_parser.add_argument("--tag", default="run")
    args = parser.parse_args()
    if args.phase == "synth":
        synth(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
