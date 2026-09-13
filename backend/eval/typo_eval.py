"""Translation of text with typing mistakes (task T32).

Usage, from backend/ with the gpu and eval groups installed, and the FLEURS files (eval.fleurs_download),
the translation models (eval.mt_convert) and the typo sets (eval.typo_noise) in place:
    uv run python -m eval.typo_eval convert                      # the correction models to CTranslate2
    uv run python -m eval.typo_eval run --split validation --tag t32_dev
    uv run python -m eval.typo_eval run --split test --directions ko-en --candidates B --tag t32_test

The candidates and the selection rule are in docs/experiments.md 6, written before any run. The
translation side is the server's setting: opus-mt-tc-big, English to Korean one sentence at a time.
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
import sacrebleu

from app.services.interfaces import Language, ModelError
from app.services.translation import MarianTranslator
from eval.common import DATA, MODELS, REPORTS, gpu_memory_mb

CT2 = MODELS / "ct2"
OLLAMA_URL = "http://localhost:11434"
LEVELS = ("clean", "light", "heavy")
DIRECTIONS: dict[str, tuple[Language, Language]] = {"ko-en": ("ko", "en"), "en-ko": ("en", "ko")}
# language: (Hugging Face repository, converted folder, text put before the input)
SEQ2SEQ = {
    "ko": ("j5ng/et5-typos-corrector", "et5-typos-corrector", "맞춤법을 고쳐주세요: "),
    "en": ("oliverguhr/spelling-correction-english-base", "spelling-correction-english-base", ""),
}
LANGUAGE_NAMES = {"ko": "Korean", "en": "English"}
SYSTEM = (
    "You fix typing mistakes. Correct only the spelling, typos and spacing in the user's {language} text. "
    "Do not change the meaning, do not translate and do not add anything. Reply with the corrected text only."
)


class Seq2SeqCorrector:
    """A correction model converted to CTranslate2, on the GPU like the translation models."""

    def __init__(self, language: Language) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        repo, folder, self.prefix = SEQ2SEQ[language]
        self.name = repo
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = ctranslate2.Translator(str(CT2 / folder), device="cuda", compute_type="float16")

    def correct(self, text: str) -> str:
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(self.prefix + text))
        result = self.model.translate_batch([tokens], beam_size=5, max_decoding_length=128)
        ids = self.tokenizer.convert_tokens_to_ids(result[0].hypotheses[0])
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def close(self) -> None:
        del self.model


class OllamaCorrector:
    """An instruction-tuned LLM served by Ollama with one fixed instruction (docs/experiments.md 6)."""

    def __init__(self, model: str, language: Language) -> None:
        self.name = f"ollama/{model}"
        self.model = model
        self.system = SYSTEM.format(language=LANGUAGE_NAMES[language])
        self.client = httpx.Client(base_url=OLLAMA_URL, timeout=120)

    def correct(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": text}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 256},
        }
        response = self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def close(self) -> None:
        unload_ollama(self.model)


def unload_ollama(model: str) -> None:
    httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    time.sleep(3)  # let the runner release its memory before the next reading


CANDIDATES: dict[str, Callable[[Language], Seq2SeqCorrector | OllamaCorrector] | None] = {
    "A": None,
    "B": Seq2SeqCorrector,
    "C": lambda language: OllamaCorrector("qwen2.5:1.5b-instruct", language),
    "D": lambda language: OllamaCorrector("qwen2.5:7b-instruct", language),
}


def load_rows(split: str, limit: int | None) -> list[dict]:
    path = DATA / "typo" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def translator_for(direction: str) -> MarianTranslator:
    source, target = DIRECTIONS[direction]
    folder = CT2 / f"opus-mt-tc-big-{source}-{target}"
    return MarianTranslator(folder, source, target, by_sentence=direction == "en-ko")


def evaluate(direction: str, candidate: str, rows: list[dict], translator: MarianTranslator) -> dict:
    source, target = DIRECTIONS[direction]
    factory = CANDIDATES[candidate]
    references = [row[target] for row in rows]
    if factory is not None and candidate in ("C", "D"):
        unload_ollama(factory(source).model)  # read VRAM without the model loaded
    _, before = gpu_memory_mb()
    corrector = factory(source) if factory else None
    if corrector is not None:
        corrector.correct(rows[0][source])  # loads the model (Ollama) and warms it up, not timed
    _, after = gpu_memory_mb()
    result: dict = {
        "direction": direction,
        "candidate": candidate,
        "corrector": corrector.name if corrector else None,
        "vram_mb": round(after - before),
        "levels": {},
    }
    for level in LEVELS:
        key = source if level == "clean" else f"{source}_{level}"
        hypotheses, corrected_texts, correction_s, total_s, items, empty, failures = [], [], [], [], [], 0, 0
        for row in rows:
            start = time.perf_counter()
            corrected = corrector.correct(row[key]) if corrector else row[key]
            if not corrected.strip():
                empty, corrected = empty + 1, row[key]  # a correction that returns nothing falls back
            middle = time.perf_counter()
            try:
                hypothesis = translator.translate(corrected, source, target)
            except ModelError:
                hypothesis, failures = "", failures + 1
            end = time.perf_counter()
            hypotheses.append(hypothesis)
            corrected_texts.append(corrected)
            correction_s.append(middle - start)
            total_s.append(end - start)
            items.append(
                {
                    "id": row["id"],
                    "input": row[key],
                    "corrected": corrected,
                    "hypothesis": hypothesis,
                    "chrf": round(sacrebleu.sentence_chrf(hypothesis, [row[target]]).score, 1),
                }
            )
        result["levels"][level] = {
            "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score,
            # Reference only (not in the rule): how close the corrected input comes to the clean original.
            "input_chrf": sacrebleu.corpus_chrf(corrected_texts, [[row[source] for row in rows]]).score,
            "correction_p50": statistics.median(correction_s),
            "total_p50": statistics.median(total_s),
            "empty_corrections": empty,
            "failures": failures,
            "items": items,
        }
    if corrector is not None:
        corrector.close()
    gc.collect()
    return result


def write_report(results: list[dict], args: argparse.Namespace, gpu_name: str, count: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = REPORTS / f"typo_{args.tag}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [
        f"# 오타 입력 번역 평가 ({args.tag})",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}",
        f"- 데이터: FLEURS {args.split} 병렬 문장 {count}쌍, 오타 세트 `data/typo/{args.split}.jsonl`",
        f"- GPU: {gpu_name}. 번역은 서버 설정(opus-mt-tc-big, 영→한 문장 단위)",
        "- chrF: sacrebleu 기본, 전체 합산. 지연: 문장당 초(교정 포함), 후보마다 첫 호출 제외",
        "- 입력 chrF: 교정 결과를 깨끗한 원문과 비교한 값 (참고용, 규칙에 없음)",
    ]
    for direction in args.directions:
        lines += [
            "",
            f"## {direction}",
            "",
            "| 후보 | 교정 모델 | 추가 VRAM (MB) | chrF 깨끗 | 가벼운 오타 | 심한 오타 | 오타 평균 "
            "| 교정 p50 | 전체 p50 | 입력 chrF (가벼움/심함) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for result in (r for r in results if r["direction"] == direction):
            levels = result["levels"]
            noisy = (levels["light"]["chrf"] + levels["heavy"]["chrf"]) / 2
            correction = max(levels[level]["correction_p50"] for level in LEVELS)
            lines.append(
                f"| {result['candidate']} | {result['corrector'] or '-'} | {result['vram_mb']} "
                f"| {levels['clean']['chrf']:.1f} | {levels['light']['chrf']:.1f} "
                f"| {levels['heavy']['chrf']:.1f} "
                f"| {noisy:.1f} | {correction:.3f} | {levels['heavy']['total_p50']:.3f} "
                f"| {levels['light']['input_chrf']:.1f} / {levels['heavy']['input_chrf']:.1f} |"
            )
    for result in results:
        if result["corrector"]:
            lines += ["", f"## {result['direction']} / {result['candidate']}: 심한 오타 교정 예 3개", ""]
            for item in result["levels"]["heavy"]["items"][:3]:
                lines.append(f"- id {item['id']}: `{item['input']}` → `{item['corrected']}`")
    report = base.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report)


def convert(force: bool) -> None:
    from ctranslate2.converters import TransformersConverter

    for repo, folder, _ in SEQ2SEQ.values():
        target = CT2 / folder
        if target.exists() and not force:
            print(f"{target} exists, skipped")
            continue
        print(f"converting {repo} ...", flush=True)
        TransformersConverter(repo).convert(str(target), quantization="float16", force=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser("convert")
    convert_parser.add_argument("--force", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--split", choices=["validation", "test"], default="validation")
    run_parser.add_argument("--directions", nargs="+", choices=list(DIRECTIONS), default=list(DIRECTIONS))
    run_parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    run_parser.add_argument("--limit", type=int, default=None, help="first N sentence pairs")
    run_parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    if args.command == "convert":
        convert(args.force)
        return
    gpu_name, _ = gpu_memory_mb()
    rows = load_rows(args.split, args.limit)
    results = []
    for direction in args.directions:
        translator = translator_for(direction)
        source, target = DIRECTIONS[direction]
        translator.translate(rows[0][source], source, target)  # warm-up, not timed
        for candidate in args.candidates:
            print(f"evaluating {direction} / {candidate} ...", flush=True)
            results.append(evaluate(direction, candidate, rows, translator))
        del translator
        gc.collect()
    print(f"written to {write_report(results, args, gpu_name, len(rows))}")


if __name__ == "__main__":
    main()
