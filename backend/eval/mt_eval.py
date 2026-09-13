"""Translation evaluation on FLEURS parallel sentences (tasks T3 and T17).

Usage, from backend/ with the gpu and eval groups installed and the models converted (eval.mt_convert):
    uv run python -m eval.mt_eval --split validation --tag t3_dev
    uv run python -m eval.mt_eval --split validation --tag t17_dev --models opus-mt-tc-big-ko-en \
        opus-mt-tc-big-ko-en/split opus-mt-tc-big-en-ko opus-mt-tc-big-en-ko/split

The candidates and the selection rule are written in docs/experiments.md before any run.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pyarrow.parquet as pq
import sacrebleu

from app.services.interfaces import Language, ModelError, Translator
from app.services.translation import MarianTranslator, NllbTranslator, OllamaTranslator, split_sentences
from eval.common import DATA, FLEURS_CONFIG, MODELS, REPORTS, gpu_memory_mb

OLLAMA_URL = "http://localhost:11434"
QWEN = "qwen2.5:7b-instruct"
KO_EN: tuple[Language, Language] = ("ko", "en")
EN_KO: tuple[Language, Language] = ("en", "ko")
CT2 = MODELS / "ct2"

# name: (how to load it, directions it translates)
CANDIDATES: dict[str, tuple[Callable[[], Translator], list[tuple[Language, Language]]]] = {
    "opus-mt-ko-en": (lambda: MarianTranslator(CT2 / "opus-mt-ko-en", "ko", "en"), [KO_EN]),
    "opus-mt-tc-big-ko-en": (lambda: MarianTranslator(CT2 / "opus-mt-tc-big-ko-en", "ko", "en"), [KO_EN]),
    "opus-mt-tc-big-en-ko": (lambda: MarianTranslator(CT2 / "opus-mt-tc-big-en-ko", "en", "ko"), [EN_KO]),
    "nllb-200-distilled-600M": (lambda: NllbTranslator(CT2 / "nllb-200-distilled-600M"), [KO_EN, EN_KO]),
    "qwen2.5-7b": (lambda: OllamaTranslator(QWEN, OLLAMA_URL), [KO_EN, EN_KO]),
    # T17: the chosen models translating one sentence at a time (docs/experiments.md 2-1)
    "opus-mt-tc-big-ko-en/split": (
        lambda: MarianTranslator(CT2 / "opus-mt-tc-big-ko-en", "ko", "en", by_sentence=True),
        [KO_EN],
    ),
    "opus-mt-tc-big-en-ko/split": (
        lambda: MarianTranslator(CT2 / "opus-mt-tc-big-en-ko", "en", "ko", by_sentence=True),
        [EN_KO],
    ),
}


def load_pairs(split: str, limit: int | None) -> list[dict]:
    """One row per sentence ID; FLEURS repeats a sentence for each speaker."""
    text: dict[str, dict[int, str]] = {}
    for language, config in FLEURS_CONFIG.items():
        path = DATA / "fleurs" / config / f"{split}.parquet"
        rows = pq.read_table(path, columns=["id", "raw_transcription"]).to_pylist()
        text[language] = {row["id"]: row["raw_transcription"].strip() for row in rows}
    ids = sorted(text["ko"].keys() & text["en"].keys())
    pairs = [{"id": i, "ko": text["ko"][i], "en": text["en"][i]} for i in ids]
    return pairs[:limit] if limit else pairs


def unload_ollama() -> None:
    httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": QWEN, "keep_alive": 0}, timeout=60)
    time.sleep(3)  # let the runner release its memory before the next reading


def evaluate(name: str, pairs: list[dict]) -> dict:
    factory, directions = CANDIDATES[name]
    uses_ollama = name.startswith("qwen")
    if uses_ollama:
        unload_ollama()
    _, before = gpu_memory_mb()
    translator = factory()
    if uses_ollama:
        translator.translate(pairs[0]["ko"], "ko", "en")  # Ollama loads the model on the first call
    _, after = gpu_memory_mb()
    result: dict = {"model": name, "model_name": translator.model_name, "vram_mb": round(after - before)}
    result["directions"] = {}

    for source, target in directions:
        translator.translate(pairs[0][source], source, target)  # warm-up, not timed
        hypotheses, latencies, items, failures = [], [], [], 0
        for pair in pairs:
            start = time.perf_counter()
            try:
                hypothesis = translator.translate(pair[source], source, target)
            except ModelError:
                hypothesis, failures = "", failures + 1
            latencies.append(time.perf_counter() - start)
            hypotheses.append(hypothesis)
            items.append(
                {
                    "id": pair["id"],
                    "latency_s": round(latencies[-1], 3),
                    "chrf": round(sacrebleu.sentence_chrf(hypothesis, [pair[target]]).score, 1),
                    "sentences": len(split_sentences(pair[source])),
                    "source": pair[source],
                    "reference": pair[target],
                    "hypothesis": hypothesis,
                }
            )
        result["directions"][f"{source}-{target}"] = {
            "chrf": sacrebleu.corpus_chrf(hypotheses, [[pair[target] for pair in pairs]]).score,
            "latency_p50": statistics.median(latencies),
            "latency_p95": statistics.quantiles(latencies, n=20)[18],
            "failures": failures,
            "count": len(pairs),
            "items": items,
        }

    del translator
    gc.collect()
    if uses_ollama:
        unload_ollama()
    return result


def write_report(results: list[dict], args: argparse.Namespace, gpu_name: str, pairs: list[dict]) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = REPORTS / f"mt_{args.tag}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# 번역 평가 ({args.tag})",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}",
        f"- 데이터: FLEURS {args.split} 병렬 문장 {len(pairs)}쌍. 두 문장 이상으로 나뉘는 원문: "
        + ", ".join(
            f"{language} {sum(len(split_sentences(pair[language])) > 1 for pair in pairs)}개"
            for language in ("ko", "en")
        ),
        f"- GPU: {gpu_name}. seq2seq는 CTranslate2 float16·빔 4, Qwen은 Ollama temperature 0",
        f"- chrF: sacrebleu {sacrebleu.__version__} 기본 설정, 전체 합산",
        "- 지연: 문장당 초, 방향별 첫 호출 제외",
    ]
    for direction, title in (("ko-en", "한→영"), ("en-ko", "영→한")):
        lines += ["", f"## {title}", "", "| 후보 | VRAM (MB) | chrF | 지연 p50 | 지연 p95 | 실패 |"]
        lines.append("|---|---|---|---|---|---|")
        for result in results:
            if values := result["directions"].get(direction):
                lines.append(
                    f"| {result['model']} | {result['vram_mb']} | {values['chrf']:.1f} "
                    f"| {values['latency_p50']:.3f} | {values['latency_p95']:.3f} | {values['failures']} |"
                )
    for result in results:
        for direction, values in result["directions"].items():
            lines += ["", f"## {result['model']} / {direction}: chrF가 낮은 5개", ""]
            for item in sorted(values["items"], key=lambda i: i["chrf"])[:5]:
                lines.append(
                    f"- id {item['id']} ({item['chrf']}): 원문 `{item['source']}` / "
                    f"참조 `{item['reference']}` / 결과 `{item['hypothesis']}`"
                )

    report = base.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--models", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--limit", type=int, default=None, help="first N sentence pairs")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    gpu_name, _ = gpu_memory_mb()
    pairs = load_pairs(args.split, args.limit)
    results = []
    for name in args.models:
        print(f"evaluating {name} ...", flush=True)
        results.append(evaluate(name, pairs))
    print(f"written to {write_report(results, args, gpu_name, pairs)}")


if __name__ == "__main__":
    main()
