"""Speech synthesis evaluation on FLEURS sentences (tasks T4 and T78).

Two phases, so that candidates with conflicting dependencies (MeloTTS pins transformers 4.27) run in their
own environments but are scored by the same speech recognition model. From backend/:

1. Synthesize, with the candidate's environment:
       <that environment's python> -m eval.tts_eval synth --candidate mms --tag t4_dev
   writes data/tts_audio/<tag>/<candidate>/<language>/<id>.wav (git-ignored) and
   eval/reports/tts_<tag>_<candidate>_synth.json (timing and VRAM per sentence).
2. Score, with the backend environment (`uv sync --group gpu --group eval`):
       uv run python -m eval.tts_eval score --tag t4_dev
   re-recognizes every saved wav with large-v3-turbo and writes eval/reports/tts_<tag>_<stamp>.md/.json.

For the T78 trial the synthesis runs inside the API's Linux container (docs/experiments.md 12), which has no
pyarrow: `export` writes the sentences to JSON on the host first and `synth --sentences-dir` reads them.
`synth --stt-load <folder>` keeps a recognition model busy on another thread while it times the sentences,
and `timing` summarizes the speed of a tag without recognizing anything.

The candidates and the selection rule are written in docs/experiments.md before any run.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import subprocess
import threading
import time
import wave
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import cycle
from pathlib import Path

from eval.common import DATA, FLEURS_CONFIG, REPORTS
from eval.common import gpu_memory_mb as device_memory

AUDIO = DATA / "tts_audio"  # git-ignored with the rest of data/
SENTENCES = AUDIO / "sentences"
_DIGIT = re.compile(r"[0-9]")


def has_digits(text: str) -> bool:
    """The pre-registered sub-metric's sentences (T78): those written with Arabic digits."""
    return bool(_DIGIT.search(text))


def _read_parquet(language: str, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    path = DATA / "fleurs" / FLEURS_CONFIG[language] / f"{split}.parquet"
    return pq.read_table(path, columns=["id", "raw_transcription", "transcription"]).to_pylist()


def sentences(language: str, split: str, limit: int | None, sentences_dir: Path | None = None) -> list[dict]:
    """One row per sentence ID, in ID order: the text to read and the transcription to score against."""
    if sentences_dir is not None:
        rows = json.loads((Path(sentences_dir) / f"{split}_{language}.json").read_text(encoding="utf-8"))
    else:
        rows = _read_parquet(language, split)
    unique = sorted({row["id"]: row for row in rows}.values(), key=lambda row: row["id"])
    return unique[:limit] if limit else unique


def export(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for language in args.languages:
        path = out / f"{args.split}_{language}.json"
        path.write_text(
            json.dumps(sentences(language, args.split, None), ensure_ascii=False, indent=0), encoding="utf-8"
        )
        print(f"written to {path}")


def nvidia_smi_memory_mb(run: Callable = subprocess.run) -> float | None:
    """GPU memory in use on the whole device, from nvidia-smi (the container has no nvidia-ml-py)."""
    try:
        result = run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return float(result.stdout.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_memory_mb() -> float | None:
    """GPU memory in use on the device, or None when neither nvidia-ml-py nor nvidia-smi is there."""
    try:
        return device_memory()[1]
    except ImportError:
        return nvidia_smi_memory_mb()


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as clip:
        return clip.getnframes() / clip.getframerate()


def load_clips(folder: Path) -> list[tuple[bytes, str]]:
    """Recognition load input: the e2e clips, named <language>_<id>.wav (eval/e2e_eval.py)."""
    return [(path.read_bytes(), path.name.split("_", 1)[0]) for path in sorted(Path(folder).glob("*_*.wav"))]


class SttLoad:
    """Keeps a speech recognition model busy on a thread of its own while sentences are synthesized: the
    server's model thread recognizing back to back while Supertonic runs beside it (T78)."""

    def __init__(self, stt, clips: list[tuple[bytes, str]]) -> None:
        if not clips:
            raise ValueError("the recognition load needs clips")
        self._stt, self._clips = stt, clips
        self._stop = threading.Event()
        self._first = threading.Event()
        self._times: list[float] = []
        self._errors: list[str] = []
        self._thread = threading.Thread(target=self._loop, name="stt-load", daemon=True)

    def _loop(self) -> None:
        for audio, language in cycle(self._clips):
            if self._stop.is_set():
                return
            start = time.perf_counter()
            try:
                self._stt.transcribe(audio, language)
                self._times.append(time.perf_counter() - start)
            except Exception as exc:  # noqa: BLE001 - counted, and the first one stops the measurement
                self._errors.append(f"{type(exc).__name__}: {exc}")
            self._first.set()

    def start(self, timeout: float = 600) -> None:
        """Start, and return once one recognition has finished (so the model is warm)."""
        self._thread.start()
        if not self._first.wait(timeout) or not self._times:
            self.stop()
            raise RuntimeError(f"the recognition load did not run: {self._errors[:1]}")

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        return {
            "calls": len(self._times) + len(self._errors),
            "errors": len(self._errors),
            "call_s_p50": round(statistics.median(self._times), 3) if self._times else None,
        }


def load_recognizer(device: str):
    """The server's recognition settings (app/services/models.py) for the load."""
    from app.services.stt import WhisperSpeechToText

    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperSpeechToText("large-v3-turbo", device=device, compute_type=compute_type, vad_filter=True)


def synth(args: argparse.Namespace) -> None:
    from eval.tts_candidates import CANDIDATES

    load, languages = CANDIDATES[args.candidate]
    if args.languages:  # e.g. the test run covers only the language a candidate was chosen for
        languages = [language for language in languages if language in args.languages]
    record: dict = {
        "tag": args.tag,
        "candidate": args.candidate,
        "split": args.split,
        "cpu_count": os.cpu_count(),
        "stt_load": bool(args.stt_load),
        "languages": {},
    }
    recognizer = load_recognizer(args.stt_load_device) if args.stt_load else None
    for language in languages:
        before = gpu_memory_mb()
        model = load(language)
        after = gpu_memory_mb()
        rows = sentences(language, args.split, args.limit, args.sentences_dir)
        model.synthesize(rows[0]["raw_transcription"], language)  # warm-up, not timed
        stt_load = None
        if recognizer is not None:
            stt_load = SttLoad(recognizer, load_clips(args.stt_load))
            stt_load.start()
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
        if stt_load is not None:
            record.setdefault("load", {})[language] = stt_load.stop()
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


def synth_records(tag: str) -> list[dict]:
    """The synthesis records of exactly this tag: tts_t78_dev_* also matches tag t78_dev_load's files."""
    records = []
    for path in sorted(REPORTS.glob(f"tts_{tag}_*_synth.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if path.name == f"tts_{tag}_{record['candidate']}_synth.json":
            records.append(record)
    return records


def speed_stats(items: list[dict]) -> dict:
    speeds = [item["per_audio_second"] for item in items if item.get("per_audio_second")]
    synth_times = [item["synth_s"] for item in items if "synth_s" in item]
    return {
        "speed_p50": statistics.median(speeds),
        "speed_p95": statistics.quantiles(speeds, n=20)[18],
        "synth_s_p50": statistics.median(synth_times),
    }


def score_language(
    language: str,
    values: dict,
    rows: dict[int, dict],
    recognize: Callable[[dict], str],
    error_rate: Callable[[str, list[str], list[str]], float] | None = None,
) -> dict:
    """Error rate over every sentence (a failed synthesis counts as fully wrong) and over digit sentences."""
    from eval.text_norm import normalize

    if error_rate is None:
        from eval.stt_eval import error_rate
    references, hypotheses, digit_refs, digit_hyps = [], [], [], []
    for item in values["items"]:
        row = rows[item["id"]]
        reference = normalize(row["transcription"], language)
        hypothesis = normalize("" if "failure" in item else recognize(item), language)
        item.update(
            reference=reference, hypothesis=hypothesis, has_digits=has_digits(row["raw_transcription"])
        )
        item["error_rate"] = round(error_rate(language, [reference], [hypothesis]), 4)
        references.append(reference)
        hypotheses.append(hypothesis)
        if item["has_digits"]:
            digit_refs.append(reference)
            digit_hyps.append(hypothesis)
    return {
        "model_name": values["model_name"],
        "metric": "CER" if language == "ko" else "WER",
        "error": error_rate(language, references, hypotheses),
        "error_digits": error_rate(language, digit_refs, digit_hyps) if digit_refs else None,
        "count_digits": len(digit_refs),
        **speed_stats(values["items"]),
        "vram_mb": values["vram_mb"],
        "failures": values["failures"],
        "count": len(values["items"]),
        "items": values["items"],
    }


def score(args: argparse.Namespace) -> None:
    from app.services.interfaces import InvalidAudioError
    from app.services.stt import WhisperSpeechToText

    stt = WhisperSpeechToText("large-v3-turbo")
    results = []
    for record in synth_records(args.tag):
        result: dict = {"candidate": record["candidate"], "languages": {}}
        for language, values in record["languages"].items():
            rows = {row["id"]: row for row in sentences(language, record["split"], None)}
            audio_dir = AUDIO / args.tag / record["candidate"] / language

            def recognize(item: dict, audio_dir: Path = audio_dir, language: str = language) -> str:
                try:  # unrecognizable audio counts as fully wrong
                    return stt.transcribe((audio_dir / f"{item['id']}.wav").read_bytes(), language).text
                except InvalidAudioError:
                    return ""

            result["languages"][language] = score_language(language, values, rows, recognize)
        if "load" in record:
            result["load"] = record["load"]
        results.append(result)
    write_report(results, args)


def timing(args: argparse.Namespace) -> None:
    """Speed only, for runs that are not scored (the recognition-load runs, T78)."""
    results = []
    for record in synth_records(args.tag):
        result: dict = {"candidate": record["candidate"], "load": record.get("load"), "languages": {}}
        for language, values in record["languages"].items():
            result["languages"][language] = {
                "model_name": values["model_name"],
                **speed_stats(values["items"]),
                "failures": values["failures"],
                "count": len(values["items"]),
            }
        results.append(result)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = REPORTS / f"tts_{args.tag}_timing_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [
        f"# 음성 합성 속도 ({args.tag}, 재인식 없음)",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}",
        "- 속도: 생성 음성 1초당 합성 시간(초). 언어별 첫 호출 제외",
        "",
        "| 후보 | 언어 | 속도 p50 | 속도 p95 | 문장당 합성 p50 | 실패 | 동시 인식 호출 수 | 인식 호출 p50 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        for language, values in result["languages"].items():
            load = (result["load"] or {}).get(language, {})
            lines.append(
                f"| {result['candidate']} | {language} | {values['speed_p50']:.3f} "
                f"| {values['speed_p95']:.3f} "
                f"| {values['synth_s_p50']:.3f}초 | {values['failures']} | {load.get('calls', '-')} "
                f"| {load.get('call_s_p50', '-')} |"
            )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")


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
        "- 숫자 문장: 원문에 아라비아 숫자가 있는 문장만 모은 오류율 (T78의 사전 등록 부지표)",
    ]
    for language, title in (("ko", "한국어 (재인식 CER)"), ("en", "영어 (재인식 WER)")):
        lines += ["", f"## {title}", ""]
        lines.append(
            "| 후보 | 오류율 | 숫자 문장 오류율 | 속도 p50 | 속도 p95 | 문장당 합성 p50 | VRAM (MB) | 실패 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for result in results:
            if values := result["languages"].get(language):
                digits = values.get("error_digits")
                digit_text = f"{digits:.2%} ({values.get('count_digits')}문장)" if digits is not None else "-"
                lines.append(
                    f"| {result['candidate']} | {values['error']:.2%} | {digit_text} "
                    f"| {values['speed_p50']:.3f} "
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
    export_parser = phases.add_parser(
        "export", help="write the sentences to JSON for an environment without pyarrow"
    )
    export_parser.add_argument("--split", choices=["validation", "test"], default="validation")
    export_parser.add_argument("--languages", nargs="+", choices=["ko", "en"], default=["ko"])
    export_parser.add_argument("--out", type=Path, default=SENTENCES)
    synth_parser = phases.add_parser("synth", help="synthesize with one candidate")
    synth_parser.add_argument("--candidate", required=True)
    synth_parser.add_argument("--split", choices=["validation", "test"], default="validation")
    synth_parser.add_argument("--limit", type=int, default=None, help="first N sentences per language")
    synth_parser.add_argument("--languages", nargs="+", choices=["ko", "en"], help="default: all it speaks")
    synth_parser.add_argument("--tag", default="run")
    synth_parser.add_argument(
        "--sentences-dir", type=Path, help="read the sentences from `export` JSON files"
    )
    synth_parser.add_argument(
        "--stt-load", type=Path, help="folder of <language>_<id>.wav clips to recognize meanwhile"
    )
    synth_parser.add_argument("--stt-load-device", choices=["cuda", "cpu"], default="cuda")
    score_parser = phases.add_parser("score", help="re-recognize every candidate's audio for a tag")
    score_parser.add_argument("--tag", default="run")
    timing_parser = phases.add_parser("timing", help="summarize the speed of a tag without recognition")
    timing_parser.add_argument("--tag", default="run")
    args = parser.parse_args()
    {"export": export, "synth": synth, "score": score, "timing": timing}[args.phase](args)


if __name__ == "__main__":
    main()
