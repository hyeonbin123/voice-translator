"""Speech recognition evaluation on FLEURS (tasks T2 and T14).

Usage, from backend/ with the gpu and eval groups installed (`uv sync --group gpu --group eval`):
    uv run python -m eval.stt_eval --split validation --models small large-v3-turbo large-v3 --tag t2_dev
    uv run python -m eval.stt_eval --split validation --models large-v3-turbo --variants a b c d --tag t14_dev

The candidates and the selection rule are written in docs/experiments.md before any run.
"""

# [파일 역할] 음성 인식 모델·디코딩 설정을 FLEURS로 재는 측정 스크립트다(T2: 모델 고르기, T14: 빈 결과를
# 줄이는 디코딩 변형 고르기, docs/experiments.md 1절·1-1절). 모델·변형마다 한국어 CER·영어 WER(전체 합산), 빈
# 결과 수, 환각 신호, 음성 1초당 처리 시간, GPU 메모리, 무음 입력 확인을 재서 reports/stt_<tag>_*.json·.md를
# 쓴다.
# [왜 필요한가] 측정 전에 적은 선택 규칙(예산 0.25초 안에서 두 언어 평균 오류율이 가장 낮은 모델, 1%p 안이면
# 빠른 쪽)에 넣을 숫자를 같은 조건으로 만든다. 이 측정으로 large-v3-turbo와 VAD 필터(변형 b)를 골랐다.
# [설계 이유] 언어는 고정해서 넘긴다. 화면에서 번역 방향을 고르므로 원문 언어를 이미 알고, 언어 판별 오류는
# 측정 대상이 아니다(1절).
# [연결] load_split·recognize는 eval/stt_decode_check.py가, error_rate·recognize는 eval/quality_eval.py가,
# error_rate는 eval/stt_rescore.py·eval/tts_eval.py가 가져다 쓴다. 저장된 결과는 eval/reanalysis.py(T80)가
# 다시 읽는다.

from __future__ import annotations

import argparse
import gc
import io
import json
import statistics
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

# jiwer: CER·WER 계산. numpy: 무음·잡음 테스트 음성 생성. pyarrow.parquet: FLEURS parquet 읽기.
# 인식은 서버의 WhisperSpeechToText를 그대로 써서 측정한 코드와 서비스 코드가 같게 한다.
import jiwer
import numpy as np
import pyarrow.parquet as pq

from app.services.interfaces import InvalidAudioError
from app.services.stt import WhisperSpeechToText
from eval.common import DATA, FLEURS_CONFIG, REPORTS, gpu_memory_mb
from eval.text_norm import normalize

# FLEURS 음성과 Whisper 입력의 표본화율(16 kHz). 음성 길이(초) = 표본 수 / 이 값.
SAMPLE_RATE = 16_000
# T14 후보(1-1절): a 기본값(기준), b vad_filter=True(Silero VAD로 말소리 구간만 넘김), c 결과가 비었을 때만
# 무음 판정 없이 한 번 더 인식, d 무음 판정을 늘 끔. 판정은 b였고 서버는 STT_VAD_FILTER(기본 켬)로 쓴다.
# Decoding variants compared in T14 (docs/experiments.md 1-1). "a" is faster-whisper's defaults, as in T2.
VARIANTS: dict[str, dict] = {
    "a": {},
    "b": {"vad_filter": True},
    "c": {"retry_without_no_speech": True},
    "d": {"no_speech_threshold": None},
}
# T14 무음 확인 입력: 말이 없는 녹음에서 문장을 지어내면 안 되므로 디지털 무음과 작은 백색 잡음을 넣어 본다.
NO_SPEECH_INPUTS = ("silence", "noise")


# 언어·분할의 FLEURS 행을 읽는다(앞 limit개만 쓸 수 있음). 음성 인식 평가는 녹음마다 따로 재므로 같은 문장의
# 여러 녹음을 모두 쓴다. 정답은 소문자·문장부호 없는 transcription 열이다.
def load_split(language: str, split: str, limit: int | None) -> list[dict]:
    path = DATA / "fleurs" / FLEURS_CONFIG[language] / f"{split}.parquet"
    rows = pq.read_table(path, columns=["id", "num_samples", "audio", "transcription"]).to_pylist()
    return rows[:limit] if limit else rows


# 한국어는 CER(글자), 영어는 WER(단어). 문장별 오류율의 평균이 아니라 전체 편집 수 ÷ 전체 참조 길이다(1절
# 지표). 긴 문장과 짧은 문장이 길이만큼 반영된다.
def error_rate(language: str, references: list[str], hypotheses: list[str]) -> float:
    # Corpus-level: total edits over total reference length, not a mean of per-sentence rates.
    return jiwer.cer(references, hypotheses) if language == "ko" else jiwer.wer(references, hypotheses)


# 3초짜리 무음 또는 진폭 0.01의 백색 잡음을 16비트 모노 WAV 바이트로 만든다. 잡음은 시드 0으로 고정해 실행마다
# 같은 입력이 된다.
def no_speech_wav(kind: str, seconds: float = 3.0) -> bytes:
    """Audio without speech: digital silence, or quiet white noise (peak amplitude 0.01, fixed seed)."""
    count = int(SAMPLE_RATE * seconds)
    samples = np.zeros(count) if kind == "silence" else np.random.default_rng(0).uniform(-0.01, 0.01, count)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


# 인식 결과 글자를 돌려주되, 말을 찾지 못해 InvalidAudioError(NoSpeechError 포함)가 나면 빈 문자열로 둔다.
# 서버가 '말 없음'으로 알리는 경우를 빈 결과로 세기 위해서다.
def recognize(stt: WhisperSpeechToText, audio: bytes, language: str) -> str:
    """The recognized text, or "" when nothing was found (what the server reports as no speech)."""
    try:
        return stt.transcribe(audio, language).text
    except InvalidAudioError:
        return ""


# 모델 크기 × 디코딩 변형 하나를 재서 결과 dict를 돌려준다. 입력: 모델 이름, 변형 키, 언어 목록, 분할, 앞 N개
# 제한, 빔 크기(기본 5).
# VRAM은 모델 적재 전후의 GPU 전체 사용량 차이다(1절 지표). 언어마다 첫 호출은 준비라 시간을 재지 않는다.
def evaluate_model(
    model_size: str, variant: str, languages: list[str], split: str, limit: int | None, beam: int
) -> dict:
    _, before = gpu_memory_mb()
    stt = WhisperSpeechToText(model_size, beam_size=beam, **VARIANTS[variant])
    _, after = gpu_memory_mb()
    result: dict = {
        "model": model_size,
        "variant": variant,
        "vram_mb": round(after - before),
        "languages": {},
        "no_speech": {},
    }

    for language in languages:
        rows = load_split(language, split, limit)
        recognize(stt, rows[0]["audio"]["bytes"], language)  # warm-up, not timed
        references, hypotheses, seconds_per_second, items = [], [], [], []
        for row in rows:
            # 발화마다 처리 시간 ÷ 음성 길이(음성 1초당 처리 초)를 잰다. 오류는 정답·결과를 같은 규칙으로
            # 정규화한 뒤 센다. 문장별 결과(정규화된 글)를 남겨 두면 나중에 모델 없이 다시 계산할 수 있다(T15
            # stt_rescore, T80 reanalysis).
            start = time.perf_counter()
            text = recognize(stt, row["audio"]["bytes"], language)
            elapsed = time.perf_counter() - start
            audio_seconds = row["num_samples"] / SAMPLE_RATE
            reference = normalize(row["transcription"], language)
            hypothesis = normalize(text, language)
            references.append(reference)
            hypotheses.append(hypothesis)
            seconds_per_second.append(elapsed / audio_seconds)
            items.append(
                {
                    "id": row["id"],
                    "audio_s": round(audio_seconds, 2),
                    "elapsed_s": round(elapsed, 3),
                    "error": round(error_rate(language, [reference], [hypothesis]), 4),
                    "reference": reference,
                    "hypothesis": hypothesis,
                }
            )
        # 언어별 요약: 전체 오류율, 빈 결과 수, 환각 신호(참조 길이의 2배를 넘는 결과), 속도 p50·p95. p95는
        # quantiles(n=20)의 19번째 경계다.
        pairs = list(zip(references, hypotheses, strict=True))
        result["languages"][language] = {
            "metric": "CER" if language == "ko" else "WER",
            "error": error_rate(language, references, hypotheses),
            "empty": sum(not hypothesis for _, hypothesis in pairs),
            # More than twice the reference's length: likely made-up text (T14's hallucination signal).
            "long": sum(len(hypothesis) > 2 * len(reference) for reference, hypothesis in pairs),
            "speed_p50": statistics.median(seconds_per_second),
            "speed_p95": statistics.quantiles(seconds_per_second, n=20)[18],
            "count": len(rows),
            "items": items,
        }
        # Inputs without speech must come back empty; anything else is text the model made up.
        for kind in NO_SPEECH_INPUTS:
            result["no_speech"][f"{language}/{kind}"] = recognize(stt, no_speech_wav(kind), language)

    # 다음 모델을 올리기 전에 이 모델을 지워 GPU 메모리를 돌려받는다.
    del stt
    gc.collect()
    return result


# 그 언어를 재지 않았으면 '-', 아니면 값을 형식에 맞춰 쓴 표 칸.
def cell(values: dict | None, key: str, spec: str) -> str:
    return "-" if values is None else format(values[key], spec)


# 결과를 reports/stt_<tag>_<UTC 시각>.json·.md로 쓰고 .md 경로를 돌려준다. 표에는 모델/변형별 VRAM, 두 언어
# 오류율과 평균(선택 규칙 값), 빈 결과·환각 신호, 속도, 무음 확인(빈 결과 수/전체)이 들어간다. 아래에는 말이
# 없는 입력에서 지어낸 글과 언어별 오류가 큰 5개를 적어 숫자 뒤의 실패를 바로 읽게 한다.
def write_report(results: list[dict], args: argparse.Namespace, gpu_name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = REPORTS / f"stt_{args.tag}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# 음성 인식 평가 ({args.tag})",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}",
        f"- 데이터: FLEURS {args.split}" + (f", 언어별 앞 {args.limit}개" if args.limit else ""),
        f"- GPU: {gpu_name}, float16, beam {args.beam}",
        "- 변형: a 기본값, b `vad_filter=True`, c 빈 결과면 무음 판정 없이 한 번 더, "
        "d `no_speech_threshold=None` (docs/experiments.md 1-1)",
        "- 속도: 음성 1초당 처리 시간(초). 언어별 첫 호출 제외",
        "- 빈 결과: 결과가 빈 문장 수. 환각 신호: 참조 길이의 2배를 넘는 결과 수. "
        "무음 확인: 무음·잡음 3초를 언어마다 넣어 빈 결과가 나온 수",
        "",
        "| 모델/변형 | VRAM (MB) | 한국어 CER | 영어 WER | 평균 | 빈 결과 (한/영) | 환각 신호 (한/영) "
        "| 한국어 속도 p50 | 영어 속도 p50 | 속도 p95 (최대) | 무음 확인 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        by_language = result["languages"]
        ko, en = by_language.get("ko"), by_language.get("en")
        mean = statistics.mean(v["error"] for v in by_language.values())
        p95 = max(v["speed_p95"] for v in by_language.values())
        no_speech = result["no_speech"]
        lines.append(
            f"| {result['model']}/{result['variant']} | {result['vram_mb']} "
            f"| {cell(ko, 'error', '.2%')} | {cell(en, 'error', '.2%')} | {mean:.2%} "
            f"| {cell(ko, 'empty', 'd')}/{cell(en, 'empty', 'd')} "
            f"| {cell(ko, 'long', 'd')}/{cell(en, 'long', 'd')} "
            f"| {cell(ko, 'speed_p50', '.3f')} | {cell(en, 'speed_p50', '.3f')} | {p95:.3f} "
            f"| {sum(not text for text in no_speech.values())}/{len(no_speech)} |"
        )
    for result in results:
        label = f"{result['model']}/{result['variant']}"
        made_up = {name: text for name, text in result["no_speech"].items() if text}
        if made_up:
            lines += ["", f"## {label}: 말이 없는 입력에서 나온 글자", ""]
            lines += [f"- {name}: `{text}`" for name, text in made_up.items()]
        for language, values in result["languages"].items():
            lines += ["", f"## {label} / {language}: 오류가 큰 5개", ""]
            for item in sorted(values["items"], key=lambda i: i["error"], reverse=True)[:5]:
                lines.append(
                    f"- id {item['id']} ({item['error']:.0%}): "
                    f"참조 `{item['reference']}` / 결과 `{item['hypothesis']}`"
                )

    report = base.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# 명령줄 진입점. --models × --variants의 모든 조합을 차례로 잰다(동시에 올리지 않아 GPU 메모리 측정이 섞이지
# 않음). 기본 모델 목록은 T2 후보(small, large-v3-turbo, 참고용 large-v3)다.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--models", nargs="+", default=["small", "large-v3-turbo", "large-v3"])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["a"])
    parser.add_argument("--languages", nargs="+", choices=["ko", "en"], default=["ko", "en"])
    parser.add_argument("--limit", type=int, default=None, help="first N utterances per language")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    gpu_name, _ = gpu_memory_mb()
    results = []
    for model_size in args.models:
        for variant in args.variants:
            print(f"evaluating {model_size}/{variant} ...", flush=True)
            results.append(
                evaluate_model(model_size, variant, args.languages, args.split, args.limit, args.beam)
            )
    print(f"written to {write_report(results, args, gpu_name)}")


if __name__ == "__main__":
    main()
