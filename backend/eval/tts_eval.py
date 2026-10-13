"""Speech synthesis evaluation on FLEURS sentences (tasks T4, T78 and T82).

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

# [파일 역할] 음성 합성 후보를 재는 측정 스크립트다(T4 3절, T78 12절, T82 14절). 합성한 음성을 음성 인식
# 모델에 다시 넣어 얼마나 알아듣기 쉬운지(재인식 CER·WER)와 합성 속도·GPU 메모리를 잰다.
# [왜 두 단계인가] 후보들은 의존성이 충돌해(MeloTTS는 transformers 4.27 고정) 각자 환경에서 합성해야 하지만,
# 비교는 같은 인식 모델로 해야 공정하다. 그래서 synth(후보 환경에서 합성해 WAV와 시간 기록을 남김)와
# score(백엔드 환경에서 large-v3-turbo로 한꺼번에 재인식)를 나눴다(3절 방법).
# [하위 명령] export: pyarrow가 없는 API 컨테이너용으로 문장을 JSON으로 내보냄(T78). synth: 후보 하나로
# 합성(--stt-load면 다른 스레드에서 인식을 계속 돌리는 부하 아래서). score: 재인식·채점·보고서. timing: 재인식
# 없이 속도만 요약(부하 실행용).
# [설계 이유] 자연스러움은 자동으로 믿을 만하게 잴 방법이 없어 규칙에 넣지 않았고, 원문을 읽혀 참조 글자를
# 정확히 해 오류가 합성 탓인지 알 수 있게 했다(3절 무엇을 재나·데이터). 재인식 CER은 자연스러움을 대신하지
# 못한다는 한계가 있다.

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

# 무거운 의존성(pyarrow, 합성기, 인식기)은 쓰는 함수 안에서 가져온다. 이 파일이 서로 다른 환경(후보 환경,
# 컨테이너)에서 불려도 import 단계에서 실패하지 않게 하려는 것이다(tests/test_tts_eval.py가 eval 그룹 없이
# import되는지 확인).
from eval.common import DATA, FLEURS_CONFIG, REPORTS
from eval.common import gpu_memory_mb as device_memory

# AUDIO: 합성 음성을 data/tts_audio/<tag>/<후보>/<언어>/<id>.wav로 남기는 폴더(사람이 들어 볼 표본, 커밋하지
# 않음). SENTENCES: export가 문장 JSON을 쓰는 기본 폴더. _DIGIT: 아라비아 숫자.
AUDIO = DATA / "tts_audio"  # git-ignored with the rest of data/
SENTENCES = AUDIO / "sentences"
_DIGIT = re.compile(r"[0-9]")


# 원문에 아라비아 숫자가 있는 문장인지. T78에서 사전 등록한 부지표 '숫자 문장 CER'(숫자·단위 읽기)을 따로 셀
# 문장을 고른다.
def has_digits(text: str) -> bool:
    """The pre-registered sub-metric's sentences (T78): those written with Arabic digits."""
    return bool(_DIGIT.search(text))


# FLEURS parquet에서 id, 읽힐 원문(raw_transcription), 채점용 정규화 전사(transcription)를 읽는다. pyarrow는
# 이 함수에서만 가져온다.
def _read_parquet(language: str, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    path = DATA / "fleurs" / FLEURS_CONFIG[language] / f"{split}.parquet"
    return pq.read_table(path, columns=["id", "raw_transcription", "transcription"]).to_pylist()


# 문장 ID마다 한 행(같은 문장의 여러 녹음 중 하나)만 ID 순서로 돌려준다. 합성은 문장마다 한 번이면 되므로
# 한국어 validation 129문장, 영어 150문장이 된다(3절 데이터). sentences_dir를 주면 export JSON을 읽는다.
def sentences(language: str, split: str, limit: int | None, sentences_dir: Path | None = None) -> list[dict]:
    """One row per sentence ID, in ID order: the text to read and the transcription to score against."""
    if sentences_dir is not None:
        rows = json.loads((Path(sentences_dir) / f"{split}_{language}.json").read_text(encoding="utf-8"))
    else:
        rows = _read_parquet(language, split)
    unique = sorted({row["id"]: row for row in rows}.values(), key=lambda row: row["id"])
    return unique[:limit] if limit else unique


# export 명령: 언어별 문장 목록을 <split>_<lang>.json으로 쓴다. 호스트에서 만들어 컨테이너의 synth
# --sentences-dir로 넘긴다.
def export(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for language in args.languages:
        path = out / f"{args.split}_{language}.json"
        path.write_text(
            json.dumps(sentences(language, args.split, None), ensure_ascii=False, indent=0), encoding="utf-8"
        )
        print(f"written to {path}")


# nvidia-smi로 장치 전체 GPU 사용 메모리(MB)를 읽는다. 컨테이너에는 nvidia-ml-py가 없어 대신 쓴다. 명령이
# 없거나 출력이 이상하면 None. run을 인자로 받는 것은 테스트가 가짜 실행 함수를 넣기 위해서다.
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


# NVML(pynvml)로 먼저 읽고, 패키지가 없으면 nvidia-smi로, 둘 다 없으면 None. GPU 없이 CPU 제공자만 쓰는
# 실행(T78 0단계)도 돌 수 있게 한다.
def gpu_memory_mb() -> float | None:
    """GPU memory in use on the device, or None when neither nvidia-ml-py nor nvidia-smi is there."""
    try:
        return device_memory()[1]
    except ImportError:
        return nvidia_smi_memory_mb()


# WAV 바이트의 길이(초). 음성 1초당 합성 시간의 분모다.
def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as clip:
        return clip.getnframes() / clip.getframerate()


# 인식 부하에 쓸 e2e 클립(<언어>_<id>.wav)을 이름순으로 읽어 (음성, 언어) 목록으로 돌려준다.
def load_clips(folder: Path) -> list[tuple[bytes, str]]:
    """Recognition load input: the e2e clips, named <language>_<id>.wav (eval/e2e_eval.py)."""
    return [(path.read_bytes(), path.name.split("_", 1)[0]) for path in sorted(Path(folder).glob("*_*.wav"))]


# 합성 시간을 재는 동안 다른 스레드에서 인식 모델을 쉬지 않고 돌리는 부하 장치(T78의 --stt-load). Supertonic은
# 서버에서 모델 스레드 밖의 합성 스레드에서 돌기 때문에, 서버의 모델 스레드가 인식을 계속 돌릴 때 합성이
# 얼마나 느려지는지 재려는 것이다(12절 방법). 호출 수·오류 수·인식 호출 시간 중앙값을 기록한다.
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

    # 클립을 돌아가며 멈춤 신호가 올 때까지 인식한다. 오류는 세어 두고, 첫 호출이 끝나면(성공이든 실패든)
    # _first를 켜 start가 기다림을 끝내게 한다.
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

    # 스레드를 띄우고 첫 인식이 끝날 때까지 기다린다(모델 준비를 마친 상태에서 합성 시간을 재기 위해). 제한
    # 시간 안에 한 번도 성공하지 못하면 부하가 없는 것이므로 멈추고 예외를 낸다.
    def start(self, timeout: float = 600) -> None:
        """Start, and return once one recognition has finished (so the model is warm)."""
        self._thread.start()
        if not self._first.wait(timeout) or not self._times:
            self.stop()
            raise RuntimeError(f"the recognition load did not run: {self._errors[:1]}")

    # 부하를 멈추고 요약을 돌려준다. 돌던 인식 한 번이 끝나야 스레드가 끝난다.
    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        return {
            "calls": len(self._times) + len(self._errors),
            "errors": len(self._errors),
            "call_s_p50": round(statistics.median(self._times), 3) if self._times else None,
        }


# 부하에 쓸 인식기. 서버와 같은 설정(large-v3-turbo, VAD 필터, GPU면 float16·CPU면 int8)으로 만든다.
def load_recognizer(device: str):
    """The server's recognition settings (app/services/models.py) for the load."""
    from app.services.stt import WhisperSpeechToText

    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperSpeechToText("large-v3-turbo", device=device, compute_type=compute_type, vad_filter=True)


# synth 명령: 후보 하나로 언어마다 모든 문장을 합성해 WAV를 저장하고, 시간·메모리 기록을
# reports/tts_<tag>_<후보>_synth.json에 쓴다.
# 기록하는 것: 모델 적재 시간, 첫 호출(준비, 통계에서 뺌) 시간, 문장별 합성 시간·음성 길이·음성 1초당 합성
# 시간, 실패, 세션 제공자(Supertonic), 장치 메모리의 적재 전 대비 차이(적재 뒤·마지막 문장 뒤). 적재·첫
# 호출·실행 뒤 메모리는 T82에서 더했다(14절 방법).
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
    # 인식 부하용 모델은 언어 반복 밖에서 한 번만 올린다.
    recognizer = load_recognizer(args.stt_load_device) if args.stt_load else None
    for language in languages:
        before = gpu_memory_mb()
        start = time.perf_counter()
        model = load(language)
        load_s = time.perf_counter() - start
        after = gpu_memory_mb()
        rows = sentences(language, args.split, args.limit, args.sentences_dir)
        # Warm-up: not in the sentence statistics, but timed on its own (T82: first-call cost on the GPU).
        start = time.perf_counter()
        model.synthesize(rows[0]["raw_transcription"], language)
        first_call_s = time.perf_counter() - start
        stt_load = None
        if recognizer is not None:
            stt_load = SttLoad(recognizer, load_clips(args.stt_load))
            stt_load.start()
        out = AUDIO / args.tag / args.candidate / language
        out.mkdir(parents=True, exist_ok=True)
        items, failures = [], 0
        # 문장마다 합성 시간을 잰다. 예외가 나도 멈추지 않고 실패로 세어 다음 문장으로 간다(실패 수가 판정
        # 조건이라 끝까지 세야 한다). 실패한 문장은 score에서 빈 결과(전부 틀림)로 센다.
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
        # Device memory after the last sentence: ONNX Runtime's arena keeps its high-water mark (T82).
        after_run = gpu_memory_mb()
        record["languages"][language] = {
            "model_name": model.model_name,
            "providers": getattr(model, "providers", None),
            "load_s": round(load_s, 3),
            "first_call_s": round(first_call_s, 3),
            "vram_mb": round(after - before) if before is not None and after is not None else None,
            "vram_after_run_mb": (
                round(after_run - before) if before is not None and after_run is not None else None
            ),
            "failures": failures,
            "items": items,
        }
        # 다음 언어 모델을 올리기 전에 이 모델의 참조를 놓는다.
        del model
    path = REPORTS / f"tts_{args.tag}_{args.candidate}_synth.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written to {path}")


# 정확히 이 태그의 synth 기록만 모은다. glob 'tts_t78_dev_*'는 't78_dev_load' 태그의 파일도 잡으므로, 파일
# 이름이 '태그_후보_synth.json'과 정확히 같은 것만 남긴다.
def synth_records(tag: str) -> list[dict]:
    """The synthesis records of exactly this tag: tts_t78_dev_* also matches tag t78_dev_load's files."""
    records = []
    for path in sorted(REPORTS.glob(f"tts_{tag}_*_synth.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if path.name == f"tts_{tag}_{record['candidate']}_synth.json":
            records.append(record)
    return records


# 성공한 문장들의 속도 요약: 음성 1초당 합성 시간 p50·p95, 문장당 합성 시간 p50. 모두 실패하면 None, 성공이
# 하나면 그 값이 p50이자 p95다(quantiles는 값이 둘 이상 필요하므로).
def speed_stats(items: list[dict]) -> dict:
    """Speed over the sentences that were synthesized. A candidate whose synthesis failed for all of them
    has no speed (None), and with one success that one is its p50 and p95 (T79): its report still shows
    the failures."""
    speeds = [item["per_audio_second"] for item in items if item.get("per_audio_second")]
    synth_times = [item["synth_s"] for item in items if "synth_s" in item]
    return {
        "speed_p50": statistics.median(speeds) if speeds else None,
        "speed_p95": statistics.quantiles(speeds, n=20)[18] if len(speeds) > 1 else next(iter(speeds), None),
        "synth_s_p50": statistics.median(synth_times) if synth_times else None,
    }


# timing 보고서에 함께 싣는 실행 정보 필드(T82부터 기록, 이전 기록에는 없어 '-'로 보임).
# Recorded per language by synth since T82; older records lack them and show "-".
RUN_FIELDS = ("providers", "load_s", "first_call_s", "vram_mb", "vram_after_run_mb")


# 값이 없으면 '-', 있으면 소수 셋째 자리와 단위를 붙인 표 칸.
def _number(value: float | None, unit: str = "") -> str:
    return "-" if value is None else f"{value:.3f}{unit}"


# 값이 없으면 '-', 있으면 문자열 그대로.
def _value(value) -> str:
    return "-" if value is None else str(value)


# 세션별 제공자 목록에서 첫 제공자(실제로 쓰는 것)만 이름 하나씩 모은다. ONNX Runtime은 CUDA 제공자를 못
# 올려도 CPU로 조용히 돌기 때문에, T82 0단계 조건 ④(네 세션 모두 CUDA)를 보고서에서 확인하려고 쓴다.
def first_providers(providers: dict[str, list[str]] | None) -> str:
    """The provider each session runs on first, each name once ("-" when not recorded)."""
    names = sorted({found[0] for found in (providers or {}).values() if found})
    return ", ".join(names) if names else "-"


# 한 후보·한 언어의 채점. 입력: synth 기록의 언어 값, 문장 ID → FLEURS 행, 문장 하나를 재인식하는 함수,
# (테스트용) 오류율 함수. 출력: 전체 오류율, 숫자 문장 오류율과 문장 수, 속도 요약, VRAM, 실패 수, 문장별
# 항목.
# 정답(transcription)과 재인식 결과를 같은 규칙(eval/text_norm.py)으로 정규화해 전체 합산으로 센다. 합성에
# 실패한 문장은 빈 결과로 두어 전부 틀린 것으로 센다(3절 지표). recognize·error_rate를 인자로 받아 테스트가
# 모델 없이 검증할 수 있다.
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


# score 명령: 이 태그의 모든 synth 기록을 읽어, 저장된 WAV를 large-v3-turbo(T2에서 고른 모델)로 다시 인식해
# 채점하고 보고서를 쓴다. 모든 후보를 같은 인식 모델로 채점하는 것이 두 단계로 나눈 이유다.
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

            # 반복 안에서 만드는 함수라 audio_dir·language를 기본값으로 묶어 둔다(늦은 바인딩으로 마지막 반복
            # 값을 쓰는 실수를 막음). 인식할 수 없는 음성은 빈 결과(전부 틀림).
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


# timing 명령: 재인식 없이 synth 기록의 속도·실패·실행 정보만 요약한다. 인식 부하 아래서 합성한 음성은
# 채점하지 않고 시간만 보기 때문이다(12·14절 방법). 동시 인식 호출 수와 인식 호출 p50도 함께 적는다.
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
                **{key: values.get(key) for key in RUN_FIELDS},
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
        "- 적재·첫 호출: 모델 적재와 첫 합성(통계에서 뺀 준비 문장)에 걸린 시간. VRAM: 장치 전체 메모리의"
        " 적재 전 대비 차이(적재 뒤, 마지막 문장 뒤). 제공자: 세션마다 ONNX Runtime이 알린 첫 제공자",
        "",
        "| 후보 | 언어 | 속도 p50 | 속도 p95 | 문장당 합성 p50 | 실패 | 동시 인식 호출 수 | 인식 호출 p50 "
        "| 적재 | 첫 호출 | VRAM 적재 뒤 (MB) | VRAM 실행 뒤 (MB) | 제공자 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        for language, values in result["languages"].items():
            load = (result["load"] or {}).get(language, {})
            lines.append(
                f"| {result['candidate']} | {language} | {_number(values['speed_p50'])} "
                f"| {_number(values['speed_p95'])} "
                f"| {_number(values['synth_s_p50'], '초')} | {values['failures']} | {load.get('calls', '-')} "
                f"| {load.get('call_s_p50', '-')} | {_number(values['load_s'], '초')} "
                f"| {_number(values['first_call_s'], '초')} | {_value(values['vram_mb'])} "
                f"| {_value(values['vram_after_run_mb'])} | {first_providers(values['providers'])} |"
            )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")


# score 결과를 reports/tts_<tag>_<UTC 시각>.json·.md로 쓴다. 언어별 표(오류율, 숫자 문장 오류율, 속도, VRAM,
# 실패)와 후보·언어별 오류가 큰 5문장(참조와 재인식 결과)을 적어 사람이 실패 유형을 읽게 한다.
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
                    f"| {_number(values['speed_p50'])} "
                    f"| {_number(values['speed_p95'])} | {_number(values['synth_s_p50'], '초')} "
                    f"| {values['vram_mb']} "
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


# 명령줄 진입점. 하위 명령 export·synth·score·timing을 고른다. synth의 --languages는 고른 언어만 확인 측정할
# 때(예: test에서 한국어만), --stt-load·--stt-load-device는 인식 부하 실행에 쓴다.
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
