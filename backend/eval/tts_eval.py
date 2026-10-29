"""FLEURS 문장으로 음성 합성을 평가한다(작업 T4, T78, T82).

두 단계로 나눠, 의존성이 충돌하는 후보(MeloTTS는 transformers 4.27 고정)는 각자 환경에서 돌리되 채점은 같은
음성 인식 모델로 한다. backend/에서:

1. 합성, 그 후보의 환경으로:
       <그 후보 환경의 python> -m eval.tts_eval synth --candidate mms --tag t4_dev
   data/tts_audio/<tag>/<candidate>/<language>/<id>.wav(git이 무시함)와
   eval/reports/tts_<tag>_<candidate>_synth.json(문장별 시간과 VRAM)을 쓴다.
2. 채점, backend 환경으로(`uv sync --group gpu --group eval`):
       uv run python -m eval.tts_eval score --tag t4_dev
   저장된 wav를 모두 large-v3-turbo로 다시 인식하고 eval/reports/tts_<tag>_<stamp>.md/.json을 쓴다.

T78 시험에서는 합성을 API의 Linux 컨테이너 안에서 돌리는데(docs/experiments.md 12절), 그 안에는 pyarrow가
없다. 그래서 `export`가 호스트에서 먼저 문장을 JSON으로 쓰고 `synth --sentences-dir`가 그것을 읽는다.
`synth --stt-load <folder>`는 문장 합성 시간을 재는 동안 다른 스레드에서 인식 모델을 쉬지 않고 돌리고,
`timing`은 인식 없이 한 태그의 속도만 요약한다.

후보와 선택 규칙은 실행 전에 docs/experiments.md에 먼저 적었다.
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
from typing import Any

# 무거운 의존성(pyarrow, 합성기, 인식기)은 쓰는 함수 안에서 가져온다. 이 파일이 서로 다른 환경(후보 환경,
# 컨테이너)에서 불려도 import 단계에서 실패하지 않게 하려는 것이다(tests/test_tts_eval.py가 eval 그룹 없이
# import되는지 확인).
from eval.common import DATA, FLEURS_CONFIG, REPORTS
from eval.common import gpu_memory_mb as device_memory

# AUDIO: 합성 음성을 data/tts_audio/<tag>/<후보>/<언어>/<id>.wav로 남기는 폴더(사람이 들어 볼 표본, 커밋하지
# 않음). SENTENCES: export가 문장 JSON을 쓰는 기본 폴더. _DIGIT: 아라비아 숫자.
AUDIO = DATA / "tts_audio"  # data/의 나머지와 함께 git이 무시한다
SENTENCES = AUDIO / "sentences"
_DIGIT = re.compile(r"[0-9]")


# 원문에 아라비아 숫자가 있는 문장인지. T78에서 사전 등록한 부지표 '숫자 문장 CER'(숫자·단위 읽기)을 따로 셀
# 문장을 고른다.
def has_digits(text: str) -> bool:
    """사전 등록한 부지표(T78)의 대상 문장: 아라비아 숫자로 쓴 문장."""
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
    """문장 ID마다 한 행, ID 순서: 읽을 원문과 채점에 쓸 전사."""
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
    """nvidia-smi로 읽은 장치 전체의 사용 중 GPU 메모리(컨테이너에는 nvidia-ml-py가 없다)."""
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
    """장치의 사용 중 GPU 메모리. nvidia-ml-py도 nvidia-smi도 없으면 None."""
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
    """인식 부하의 입력: <language>_<id>.wav 이름의 e2e 클립(eval/e2e_eval.py)."""
    return [(path.read_bytes(), path.name.split("_", 1)[0]) for path in sorted(Path(folder).glob("*_*.wav"))]


# 합성 시간을 재는 동안 다른 스레드에서 인식 모델을 쉬지 않고 돌리는 부하 장치(T78의 --stt-load). Supertonic은
# 서버에서 모델 스레드 밖의 합성 스레드에서 돌기 때문에, 서버의 모델 스레드가 인식을 계속 돌릴 때 합성이
# 얼마나 느려지는지 재려는 것이다(12절 방법). 호출 수·오류 수·인식 호출 시간 중앙값을 기록한다.
class SttLoad:
    """문장을 합성하는 동안 별도 스레드에서 음성 인식 모델을 쉬지 않고 돌린다: Supertonic이 옆에서 도는
    동안 서버의 모델 스레드가 인식을 연달아 하는 상황(T78)."""

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
            except Exception as exc:  # noqa: BLE001 - 세어 두고, 첫 오류는 측정을 멈춘다
                self._errors.append(f"{type(exc).__name__}: {exc}")
            self._first.set()

    # 스레드를 띄우고 첫 인식이 끝날 때까지 기다린다(모델 준비를 마친 상태에서 합성 시간을 재기 위해). 제한
    # 시간 안에 한 번도 성공하지 못하면 부하가 없는 것이므로 멈추고 예외를 낸다.
    def start(self, timeout: float = 600) -> None:
        """시작하고, 인식이 한 번 끝나면(모델이 준비되면) 돌아온다."""
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
    """부하용 인식기를 서버의 인식 설정(app/services/models.py)으로 만든다."""
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
    if args.languages:  # 예: test 실행은 후보가 뽑힌 언어만 다룬다
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
        # 준비 호출: 문장 통계에는 넣지 않지만 따로 시간을 잰다(T82: GPU에서 첫 호출 비용).
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
            except Exception as exc:  # noqa: BLE001 - 실패를 모두 세고 계속한다
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
        # 마지막 문장 뒤 장치 메모리: ONNX Runtime의 arena는 가장 많이 쓴 양을 그대로 잡고 있다(T82).
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
    """정확히 이 태그의 합성 기록. tts_t78_dev_*는 t78_dev_load 태그의 파일도 잡기 때문이다."""
    records = []
    for path in sorted(REPORTS.glob(f"tts_{tag}_*_synth.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if path.name == f"tts_{tag}_{record['candidate']}_synth.json":
            records.append(record)
    return records


# 성공한 문장들의 속도 요약: 음성 1초당 합성 시간 p50·p95, 문장당 합성 시간 p50. 모두 실패하면 None, 성공이
# 하나면 그 값이 p50이자 p95다(quantiles는 값이 둘 이상 필요하므로).
def speed_stats(items: list[dict]) -> dict:
    """합성된 문장들의 속도. 모든 문장의 합성에 실패한 후보는 속도가 없고(None), 성공이 하나면 그 값이
    p50이자 p95다. 어느 경우든 보고서에는 실패가 그대로 나온다."""
    speeds = [item["per_audio_second"] for item in items if item.get("per_audio_second")]
    synth_times = [item["synth_s"] for item in items if "synth_s" in item]
    return {
        "speed_p50": statistics.median(speeds) if speeds else None,
        "speed_p95": statistics.quantiles(speeds, n=20)[18] if len(speeds) > 1 else next(iter(speeds), None),
        "synth_s_p50": statistics.median(synth_times) if synth_times else None,
    }


# timing 보고서에 함께 싣는 실행 정보 필드. T82부터 synth가 언어별로 기록하며, 이전 기록에는 없어 '-'로
# 보인다.
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
    """세션마다 첫 번째로 쓰는 제공자 이름을 하나씩("-"는 기록 없음)."""
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
    """모든 문장(합성 실패는 전부 틀린 것으로 셈)과 숫자 문장 각각의 오류율."""
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
                try:  # 인식할 수 없는 음성은 전부 틀린 것으로 센다
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
    """채점하지 않는 실행(인식 부하 실행, T78)의 속도만 요약한다."""
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


# 여기부터 melo-stages 명령(docs/experiments.md 17절, 측정 전 규칙과 함께 커밋). 서비스의 한국어 합성(MeloTTS)
# 한 번이 어느 단계에 시간을 쓰는지 나눠 잰다. 고정 리비전의 melo/api.py `tts_to_file`은 ① 글을 문장 조각으로
# 나누고(split_sentence) ② 조각마다 글 전처리와 BERT 특징(utils.get_text_for_tts_infer) ③ 음성
# 생성(model.infer) 을 한 뒤 ④ 호출 끝에 GPU 캐시를 비우고(torch.cuda.empty_cache) ⑤ 조각 사이에 50ms 무음을
# 넣어 잇는다 (audio_numpy_concat). 이 다섯을 감싸 시간을 모으고, 나머지(텐서 옮기기·WAV 만들기)는 '그 밖'으로
# 남긴다. 서비스 코드(app/services/tts.py)는 바꾸지 않는다: 이 명령이 측정하는 동안에만 melo 모듈의 함수를
# 감쌌다가 되돌린다. GPU 작업은 비동기로 돌아 함수가 돌아온 뒤에도 계산이 남을 수 있으므로, ②③을 감쌀 때는
# 끝에 torch.cuda.synchronize()로 기다려 그 단계의 시간에 넣는다(원래 코드도 ③ 바로 뒤의 .cpu()에서 기다린다).
MELO_STAGES = ("split", "text", "infer", "empty_cache", "concat")


class MeloStageClock:
    """MeloTTS 합성 한 번 안의 단계별 시간(초)과 조각 수를 모은다. reset()으로 문장마다 새로 센다."""

    def __init__(self, synchronize: Callable[[], None] | None = None) -> None:
        self.synchronize = synchronize
        self.reset()

    def reset(self) -> None:
        self.seconds = dict.fromkeys(MELO_STAGES, 0.0)
        self.pieces = 0

    def wrap(self, name: str, fn: Callable, sync: bool = False) -> Callable:
        """fn을 감싸 걸린 시간을 name 단계에 더한다. sync면 끝에 GPU를 기다린다. split은 조각 수도 센다."""

        def timed(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            if sync and self.synchronize is not None:
                self.synchronize()
            self.seconds[name] += time.perf_counter() - start
            if name == "split":
                self.pieces = len(result)
            return result

        return timed


def instrument_melo(
    api_module, utils_module, model, torch_module
) -> tuple[MeloStageClock, Callable[[], None]]:
    """MeloTTS의 다섯 단계를 감싸고 (시계, 되돌리기 함수)를 돌려준다.

    api_module은 melo.api(split_sentence 이름과 TTS 클래스가 있음), utils_module은 melo.utils, model은
    melo.api.TTS 객체(그 안의 model.infer를 감쌈), torch_module은 torch다. 인자로 받는 것은 테스트가 가짜
    모듈로 같은 감싸기를 확인하기 위해서다. 되돌리기 함수를 부르면 원래 함수가 제자리로 돌아간다.
    """
    cuda = torch_module.cuda
    clock = MeloStageClock(cuda.synchronize if cuda.is_available() else None)
    originals = [
        (api_module, "split_sentence", api_module.split_sentence),
        (utils_module, "get_text_for_tts_infer", utils_module.get_text_for_tts_infer),
        (cuda, "empty_cache", cuda.empty_cache),
        (api_module.TTS, "audio_numpy_concat", api_module.TTS.__dict__["audio_numpy_concat"]),
    ]
    api_module.split_sentence = clock.wrap("split", api_module.split_sentence)
    utils_module.get_text_for_tts_infer = clock.wrap("text", utils_module.get_text_for_tts_infer, sync=True)
    model.model.infer = clock.wrap("infer", model.model.infer, sync=True)
    cuda.empty_cache = clock.wrap("empty_cache", cuda.empty_cache)
    concat = api_module.TTS.__dict__["audio_numpy_concat"].__func__
    api_module.TTS.audio_numpy_concat = staticmethod(clock.wrap("concat", concat))

    def undo() -> None:
        for owner, name, original in originals:
            setattr(owner, name, original)
        # 인스턴스에 덧붙인 infer를 지워 클래스의 원래 메서드가 다시 보이게 한다.
        vars(model.model).pop("infer", None)

    return clock, undo


def stage_texts(stream_reports: list[Path]) -> list[dict]:
    """17절 서비스 측정 보고서(stream_eval service)에서 한국어로 읽힌 번역문(영어 원문 마디의 최종 결과)을
    모은다.

    같은 문장이 두 모드(동시통역·대화 모드)에서 같은 번역문으로 나오므로 글이 같으면 한 번만 쓴다. 실패한
    마디는 뺀다. 돌려주는 것은 [{"id": "<key>", "text": 번역문}] 목록(처음 나온 순서).
    """
    seen: dict[str, str] = {}
    for path in stream_reports:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in report["details"].get("en", []):
            result = item.get("result") or {}
            text = result.get("translated_text")
            if item.get("outcome") == "final" and result.get("target_lang") == "ko" and text:
                seen.setdefault(text, item["key"])
    return [{"id": key, "text": text} for text, key in seen.items()]


def summarize_stages(rows: list[dict]) -> dict:
    """문장별 단계 시간에서 17절 규칙이 보는 값들: 합성 전체 p50, 단계별 p50과 전체에서의 몫(합 기준),
    조각 2개 이상인 문장의 비율, 계측 없이 잰 합성 p50(계측이 더한 시간을 보려고)."""
    totals = [row["total_ms"] for row in rows]
    plain = [row["plain_ms"] for row in rows]
    stages = {
        name: {
            "p50_ms": statistics.median(row["stages_ms"][name] for row in rows),
            "share": sum(row["stages_ms"][name] for row in rows) / sum(totals),
        }
        for name in MELO_STAGES
    }
    other = [row["total_ms"] - sum(row["stages_ms"].values()) for row in rows]
    return {
        "sentences": len(rows),
        "total_p50_ms": statistics.median(totals),
        "plain_p50_ms": statistics.median(plain),
        "stages": stages,
        "other_p50_ms": statistics.median(other),
        "other_share": sum(other) / sum(totals),
        "multi_piece_ratio": sum(1 for row in rows if row["pieces"] >= 2) / len(rows),
        "pieces": {
            str(n): sum(1 for row in rows if row["pieces"] == n) for n in sorted({r["pieces"] for r in rows})
        },
    }


# [g2pkk 정규식 캐시 조건, 18절] melo-stages --regex-cache: off(기본)는 17절과 같다(캐시를 건드리지 않음).
# on은 서비스의 MELO_REGEX_CACHE=true처럼 생성자가 캐시를 켠 채로 잰다. alternate는 문장마다 같은 글을 끔·켬
# 두 조건으로 재되, 짝수 번째 문장(0부터)은 끔→켬, 홀수 번째는 켬→끔으로 먼저 재는 조건을 바꾼다. 같은
# 문장끼리 견주므로 문장 길이 차이가 빠지고, 번갈아 재므로 GPU 클럭·캐시 상태처럼 순서에 따라 생기는 차이가 한
# 조건에 몰리지 않는다. 켜고 끄는 것은 app/services/tts.py의 _cache_g2pkk_regex(켜기)와 그 되돌리기 함수다.
# 되돌려도 컴파일해 둔 패턴은 비우지 않으므로 켬 조건은 서비스가 오래 돈 뒤처럼 늘 데워진 상태다(데우기에서 한
# 번 채움).
REGEX_CACHE_MODES = ("off", "on", "alternate")


def regex_cache_orders(mode: str, count: int) -> list[tuple[str, ...]]:
    """문장마다 잴 조건의 순서. alternate는 짝수 번째 문장(0부터) ("off", "on"), 홀수 번째 ("on", "off")이고,
    off·on은 문장마다 그 조건 하나다. 입력: mode(REGEX_CACHE_MODES의 하나), count(문장 수)."""
    if mode == "alternate":
        return [("off", "on") if index % 2 == 0 else ("on", "off") for index in range(count)]
    return [(mode,)] * count


def measure_melo_stages(
    model,
    texts: list[dict],
    mode: str,
    api_module,
    utils_module,
    torch_module,
    switch: Callable[[], tuple[tuple[str, ...], Callable[[], None]]],
    after_warm_up: Callable[[], None] | None = None,
) -> list[dict]:
    """첫 문장으로 데운 뒤, 문장마다 정한 조건에서 ① 계측 없이 한 번(plain, 전체 시간만) ② 계측을 켜고 한 번
    (단계별) 합성해 행을 만든다.

    model은 MeloTextToSpeech(synthesize와 안의 melo TTS인 _model이 있음),
    api_module·utils_module·torch_module은 instrument_melo에 넘기는 melo.api·melo.utils·torch, switch는
    app.services.tts._cache_g2pkk_regex다(인자로 받는 것은 테스트가 가짜 모듈로 같은 흐름을 확인하기
    위해서다). 행은 17절과 같은 칸에 "regex_cache"(그 행의 조건)를 더한 dict이고, alternate는 문장마다 두 행이
    생긴다.
    - off·on은 측정 중에 캐시를 바꾸지 않는다(on은 생성자가 이미 켰다). alternate는 켬 조건의 두 합성 동안만
      switch로 켰다가 되돌린다(예외가 나도 되돌림). 켤 때 바꾼 모듈이 하나도 없으면 캐시가 이미 켜져 있던
      것이라 끔 조건이 끔이 아니게 되므로 RuntimeError로 멈춘다.
    - 데우기: 첫 문장을 첫 순서의 조건마다 한 번(alternate는 끔·켬 한 번씩) 합성하고 통계에서 뺀다. 켬 조건의
      첫 합성은 패턴 약 570개를 처음 컴파일하므로, 데우지 않으면 첫 켬 행에만 그 시간이 들어간다.
      after_warm_up은 데우기 직후 한 번 불린다(측정 도구가 캐시 적중 수를 그때 적음). MeloTTS는 호출마다
      잡음을 새로 뽑아 두 합성의 소리 길이가 조금 다를 수 있다(audio_ms는 계측 합성의 것).
    """

    def under(condition: str, run: Callable[[], Any]) -> Any:
        if mode != "alternate" or condition == "off":
            return run()
        modules, restore = switch()
        try:
            if not modules:
                raise RuntimeError(
                    "the g2pkk regex cache is already on, so the off condition would not be off"
                )
            return run()
        finally:
            restore()

    def measure_one(item: dict, condition: str) -> dict:
        start = time.perf_counter()
        model.synthesize(item["text"], "ko")
        plain_ms = (time.perf_counter() - start) * 1000
        clock, undo = instrument_melo(api_module, utils_module, model._model, torch_module)
        try:
            start = time.perf_counter()
            audio = model.synthesize(item["text"], "ko")
            total_ms = (time.perf_counter() - start) * 1000
        finally:
            undo()
        return {
            "id": item["id"],
            "regex_cache": condition,
            "chars": len(item["text"]),
            "pieces": clock.pieces,
            "audio_ms": audio.duration_ms,
            "plain_ms": round(plain_ms, 1),
            "total_ms": round(total_ms, 1),
            "stages_ms": {name: round(seconds * 1000, 1) for name, seconds in clock.seconds.items()},
        }

    orders = regex_cache_orders(mode, len(texts))
    for condition in orders[0]:
        under(condition, lambda: model.synthesize(texts[0]["text"], "ko"))  # 데우기(통계에서 뺌)
    if after_warm_up is not None:
        after_warm_up()
    rows = []
    for item, conditions in zip(texts, orders, strict=True):
        for condition in conditions:
            rows.append(under(condition, lambda item=item, condition=condition: measure_one(item, condition)))
    return rows


def summarize_regex_cache(rows: list[dict]) -> dict:
    """alternate 행의 요약: 조건마다 17절과 같은 요약(summarize_stages)과, 두 조건으로 잰 같은 문장의 끔 − 켬
    차이 중앙값(전처리·BERT 단계 text_saving_p50_ms, 합성 전체 total_saving_p50_ms, 짝지은 문장 수)."""
    by = {condition: [row for row in rows if row["regex_cache"] == condition] for condition in ("off", "on")}
    off = {row["id"]: row for row in by["off"]}
    on = {row["id"]: row for row in by["on"]}
    both = [key for key in off if key in on]
    return {
        "off": summarize_stages(by["off"]),
        "on": summarize_stages(by["on"]),
        "paired_sentences": len(both),
        "text_saving_p50_ms": statistics.median(
            off[key]["stages_ms"]["text"] - on[key]["stages_ms"]["text"] for key in both
        ),
        "total_saving_p50_ms": statistics.median(off[key]["total_ms"] - on[key]["total_ms"] for key in both),
    }


# 단계 표 한 벌(17절 보고서의 표와 같음). summary는 summarize_stages의 결과.
STAGE_NAMES = {
    "split": "조각 나누기",
    "text": "전처리·BERT",
    "infer": "음성 생성",
    "empty_cache": "캐시 비우기",
    "concat": "잇기",
}


def _stage_lines(summary: dict) -> list[str]:
    lines = [
        f"- 합성 전체 p50(계측 켬) {summary['total_p50_ms']:.0f}ms, "
        f"계측 없이 {summary['plain_p50_ms']:.0f}ms",
        f"- 조각 2개 이상인 번역문 {summary['multi_piece_ratio']:.0%}, 조각 수별 문장 수 {summary['pieces']}",
        "",
        "| 단계 | p50 | 전체에서의 몫 |",
        "|---|---|---|",
    ]
    for name in MELO_STAGES:
        stage = summary["stages"][name]
        lines.append(f"| {STAGE_NAMES[name]} | {stage['p50_ms']:.1f}ms | {stage['share']:.1%} |")
    lines.append(f"| 그 밖 | {summary['other_p50_ms']:.1f}ms | {summary['other_share']:.1%} |")
    return lines


def write_stage_report(record: dict, base: Path) -> Path:
    """melo-stages 기록을 base.json·base.md로 쓰고 .md 경로를 돌려준다. alternate는 짝지은 차이와 조건별 표 두
    벌을, off·on은 17절과 같은 표 한 벌을 쓴다. regex_cache_info가 있으면 데우기 뒤 새로 컴파일한 패턴 수(켬
    조건이 컴파일을 건너뛰었는지)를 적는다."""
    base.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    mode = record.get("regex_cache", "off")
    summary = record["summary"]
    count = summary["off"]["sentences"] if mode == "alternate" else summary["sentences"]
    lines = [
        f"# MeloTTS 단계별 시간 ({record['tag']})",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}, 장치 {record['device']}, g2pkk 정규식 캐시 {mode}, "
        f"문장 {count}개(서비스 측정의 영어 원문 마디가 받은 한국어 번역문)",
    ]
    info = record.get("regex_cache_info") or {}
    if info.get("after_warm_up") and info.get("end"):
        new = info["end"]["misses"] - info["after_warm_up"]["misses"]
        lines.append(
            f"- g2pkk 정규식 캐시: 데우기 뒤 새 컴파일 {new}, 적중 {info['end']['hits']}, "
            f"캐시 크기 {info['end']['currsize']}"
        )
    if mode == "alternate":
        lines.append(
            f"- 같은 문장의 끔 − 켬 중앙값({summary['paired_sentences']}문장): 전처리·BERT "
            f"{summary['text_saving_p50_ms']:.0f}ms, 합성 전체 {summary['total_saving_p50_ms']:.0f}ms"
        )
        for condition, title in (("off", "끔"), ("on", "켬")):
            lines += ["", f"## {title} ({condition})", "", *_stage_lines(summary[condition])]
    else:
        lines += _stage_lines(summary)
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".md")


# melo-stages 명령: 서비스와 같은 MeloTextToSpeech("ko")를 올리고, 첫 문장으로 데운 뒤 문장마다 ① 계측 없이 한
# 번(plain, 전체 시간만) ② 계측을 켜고 한 번(단계별) 합성한다(measure_melo_stages). 두 번 합성하는 것은
# 계측(동기화) 자체가 더한 시간을 함께 보려는 것이다. tts 그룹(transformers 4.27)이 있는 API 이미지 안에서
# 돌린다(17·18절 실행 방법). --regex-cache로 g2pkk 정규식 캐시 조건을 고른다(위 주석, 18절). 모델을 만든 뒤에
# melo.api를 import한다: 생성자가 pin_transformers_revisions()로 BERT 토크나이저의 리비전을 건 뒤 melo.api를
# 불러오는데, 이 도구가 먼저 import하면 언어 모듈의 토크나이저가 리비전 없이 받아진다(17절 측정 때의 순서,
# 받은 파일은 고정 커밋과 같았다 — docs/security.md 재현성 고정).
def melo_stages(args: argparse.Namespace) -> None:
    from app.services import tts

    texts = stage_texts(args.from_stream)
    if args.limit:
        texts = texts[: args.limit]
    model = tts.MeloTextToSpeech("ko", device=args.device, regex_cache=args.regex_cache == "on")
    import melo.api
    import melo.utils
    import torch

    def cache_info() -> dict:
        info = tts._G2PKK_RE.cache_info()
        return {"hits": info.hits, "misses": info.misses, "currsize": info.currsize}

    info: dict[str, dict] = {}
    rows = measure_melo_stages(
        model,
        texts,
        args.regex_cache,
        melo.api,
        melo.utils,
        torch,
        tts._cache_g2pkk_regex,
        after_warm_up=lambda: info.update(after_warm_up=cache_info()),
    )
    info["end"] = cache_info()
    summary = summarize_regex_cache(rows) if args.regex_cache == "alternate" else summarize_stages(rows)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    record = {
        "tag": args.tag,
        "device": args.device,
        "regex_cache": args.regex_cache,
        "sources": [str(p) for p in args.from_stream],
        "model_name": model.model_name,
        "regex_cache_info": info,
        "summary": summary,
        "rows": rows,
    }
    md = write_stage_report(record, REPORTS / f"tts_{args.tag}_melo_stages_{stamp}")
    print(f"written to {md}")


# 명령줄 진입점. 하위 명령 export·synth·score·timing·melo-stages를 고른다. synth의 --languages는 고른 언어만
# 확인 측정할 때(예: test에서 한국어만), --stt-load·--stt-load-device는 인식 부하 실행에 쓴다.
def main() -> None:
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    export_parser = phases.add_parser("export", help="pyarrow가 없는 환경을 위해 문장을 JSON으로 쓴다")
    export_parser.add_argument("--split", choices=["validation", "test"], default="validation")
    export_parser.add_argument("--languages", nargs="+", choices=["ko", "en"], default=["ko"])
    export_parser.add_argument("--out", type=Path, default=SENTENCES)
    synth_parser = phases.add_parser("synth", help="후보 하나로 합성한다")
    synth_parser.add_argument("--candidate", required=True)
    synth_parser.add_argument("--split", choices=["validation", "test"], default="validation")
    synth_parser.add_argument("--limit", type=int, default=None, help="언어마다 앞에서부터 N문장만 쓴다")
    synth_parser.add_argument("--languages", nargs="+", choices=["ko", "en"], help="기본: 후보의 모든 언어")
    synth_parser.add_argument("--tag", default="run")
    synth_parser.add_argument("--sentences-dir", type=Path, help="`export`가 쓴 JSON 파일에서 문장을 읽는다")
    synth_parser.add_argument(
        "--stt-load", type=Path, help="합성하는 동안 인식할 <language>_<id>.wav 클립 폴더"
    )
    synth_parser.add_argument("--stt-load-device", choices=["cuda", "cpu"], default="cuda")
    score_parser = phases.add_parser("score", help="한 태그의 모든 후보 음성을 다시 인식해 채점한다")
    score_parser.add_argument("--tag", default="run")
    timing_parser = phases.add_parser("timing", help="인식 없이 한 태그의 속도를 요약한다")
    timing_parser.add_argument("--tag", default="run")
    stages_parser = phases.add_parser(
        "melo-stages",
        help="MeloTTS 한국어 합성 한 번을 단계별(조각 나누기·전처리·생성·캐시 비우기·잇기)로 잰다"
        " (docs/experiments.md 17·18절)",
    )
    stages_parser.add_argument(
        "--from-stream",
        type=Path,
        nargs="+",
        required=True,
        help="stream_eval service 보고서(JSON). 영어 원문 마디의 한국어 번역문을 합성할 글로 쓴다",
    )
    stages_parser.add_argument("--tag", required=True)
    stages_parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    stages_parser.add_argument("--limit", type=int, default=None, help="앞에서부터 N문장만(시험용)")
    stages_parser.add_argument(
        "--regex-cache",
        choices=REGEX_CACHE_MODES,
        default="off",
        help="g2pkk 정규식 캐시(서버 설정 MELO_REGEX_CACHE) 조건. off(기본): 17절과 같이 끈 채로."
        " on: 처음부터 켠 채로. alternate: 문장마다 같은 글을 끔·켬 두 조건으로 재고 먼저 재는 조건을"
        " 문장마다 바꾼다 (docs/experiments.md 18절)",
    )
    args = parser.parse_args()
    commands = {
        "export": export,
        "synth": synth,
        "score": score,
        "timing": timing,
        "melo-stages": melo_stages,
    }
    commands[args.phase](args)


if __name__ == "__main__":
    main()
