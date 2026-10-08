"""Translation evaluation on FLEURS parallel sentences (tasks T3, T17 and T83).

Usage, from backend/ with the gpu and eval groups installed and the models converted (eval.mt_convert):
    uv run python -m eval.mt_eval --split validation --tag t3_dev
    uv run python -m eval.mt_eval --split validation --tag t17_dev --models opus-mt-tc-big-ko-en \
        opus-mt-tc-big-ko-en/split opus-mt-tc-big-en-ko opus-mt-tc-big-en-ko/split
    uv run python -m eval.mt_eval --split validation --tag t83_dev --models opus-mt-tc-big-ko-en \
        opus-mt-tc-big-en-ko/split hy-mt2-q8 hy-mt2-q8/split

T83: the Hy-MT2 models must be in Ollama as eval.hymt_setup builds them, and each is refused unless Ollama
has the build recorded in docs/experiments.md 15.

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
from app.services.translation import (
    HyMtTranslator,
    MarianTranslator,
    NllbTranslator,
    OllamaTranslator,
    split_sentences,
)
from eval.common import DATA, FLEURS_CONFIG, MODELS, REPORTS, gpu_memory_mb
from eval.hymt_setup import QUANTS
from eval.mt_checks import check_items

OLLAMA_URL = "http://localhost:11434"
QWEN = "qwen2.5:7b-instruct"
KO_EN: tuple[Language, Language] = ("ko", "en")
EN_KO: tuple[Language, Language] = ("en", "ko")
CT2 = MODELS / "ct2"


def hymt(quant: str, by_sentence: bool = False) -> HyMtTranslator:
    """Hy-MT2 on Ollama (T83), refused unless Ollama has the build recorded in docs/experiments.md 15."""
    _, _, _, model, digest = QUANTS[quant]
    return HyMtTranslator(model, OLLAMA_URL, timeout_s=120, by_sentence=by_sentence, expected_digest=digest)


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
    # T83: Hy-MT2-1.8B served by Ollama (docs/experiments.md 15); Q4_K_M only as the latency fallback arm.
    "hy-mt2-q8": (lambda: hymt("q8_0"), [KO_EN, EN_KO]),
    "hy-mt2-q8/split": (lambda: hymt("q8_0", by_sentence=True), [EN_KO]),
    "hy-mt2-q4": (lambda: hymt("q4_k_m"), [KO_EN, EN_KO]),
    "hy-mt2-q4/split": (lambda: hymt("q4_k_m", by_sentence=True), [EN_KO]),
}
# The Ollama model behind each Ollama candidate, unloaded before and after it so VRAM is read without it.
OLLAMA_MODELS = {
    "qwen2.5-7b": QWEN,
    "hy-mt2-q8": QUANTS["q8_0"][3],
    "hy-mt2-q8/split": QUANTS["q8_0"][3],
    "hy-mt2-q4": QUANTS["q4_k_m"][3],
    "hy-mt2-q4/split": QUANTS["q4_k_m"][3],
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


def unload_ollama(model: str = QWEN) -> None:
    httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    time.sleep(3)  # let the runner release its memory before the next reading


def ollama_state(model: str) -> dict:
    """Ollama's version, the model's digest, and the memory Ollama places on the GPU for it (/api/ps)."""
    version = httpx.get(f"{OLLAMA_URL}/api/version", timeout=10).json()["version"]
    tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()["models"]
    running = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=10).json()["models"]
    digest = next((m["digest"] for m in tags if m.get("name") == model), None)
    loaded = next((m for m in running if m.get("name") == model), {})
    return {
        "ollama": version,
        "model": model,
        "digest": digest,
        "size_vram_mb": round(loaded.get("size_vram", 0) / 2**20),
        "context_length": loaded.get("context_length"),
    }


def evaluate(name: str, pairs: list[dict]) -> dict:
    factory, directions = CANDIDATES[name]
    ollama_model = OLLAMA_MODELS.get(name)
    if ollama_model:
        unload_ollama(ollama_model)
    _, before = gpu_memory_mb()
    translator = factory()
    if isinstance(translator, HyMtTranslator):
        translator.prepare()  # refuses another build, then loads the model
    if ollama_model:
        source, target = directions[0]
        translator.translate(pairs[0][source], source, target)  # Ollama loads the model on the first call
    _, after = gpu_memory_mb()
    result: dict = {"model": name, "model_name": translator.model_name, "vram_mb": round(after - before)}
    if ollama_model:
        result["ollama"] = ollama_state(ollama_model)
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
            if isinstance(translator, HyMtTranslator):
                items[-1]["ollama_calls"] = list(translator.calls)
        values = {
            "chrf": sacrebleu.corpus_chrf(hypotheses, [[pair[target] for pair in pairs]]).score,
            "latency_p50": statistics.median(latencies),
            "latency_p95": statistics.quantiles(latencies, n=20)[18],
            "failures": failures,
            "count": len(pairs),
            # T83 rule (b): other scripts or explanatory outputs, and empty outputs (eval/mt_checks.py).
            "checks": check_items(items, f"{source}-{target}"),
            "items": items,
        }
        if isinstance(translator, HyMtTranslator):
            calls = [call for item in items for call in item.get("ollama_calls", [])]
            values["cut_at_cap"] = sum(call.get("done_reason") == "length" for call in calls)
        result["directions"][f"{source}-{target}"] = values

    del translator
    gc.collect()
    if ollama_model:
        unload_ollama(ollama_model)
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
        f"- GPU: {gpu_name}. seq2seq는 CTranslate2 float16·빔 4, Qwen은 Ollama temperature 0, "
        "Hy-MT2는 Ollama(eval/hymt_setup.py의 Modelfile) temperature 0",
        f"- chrF: sacrebleu {sacrebleu.__version__} 기본 설정, 전체 합산",
        "- 지연: 문장당 초, 방향별 첫 호출 제외",
    ]
    for direction, title in (("ko-en", "한→영"), ("en-ko", "영→한")):
        lines += [
            "",
            f"## {title}",
            "",
            "| 후보 | VRAM (MB) | chrF | 지연 p50 | 지연 p95 | 실패 | 다른 문자·설명형 | 빈 출력 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for result in results:
            if values := result["directions"].get(direction):
                checks = values["checks"]
                lines.append(
                    f"| {result['model']} | {result['vram_mb']} | {values['chrf']:.1f} "
                    f"| {values['latency_p50']:.3f} | {values['latency_p95']:.3f} | {values['failures']} "
                    f"| {checks['flagged']} | {checks['empty']} |"
                )
    lines.append("")
    for result in results:
        if state := result.get("ollama"):
            lines.append(
                f"- {result['model']}: Ollama {state['ollama']}, {state['model']} "
                f"digest `{state['digest']}`, Ollama이 GPU에 둔 크기 {state['size_vram_mb']}MB, "
                f"문맥 {state['context_length']}"
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
