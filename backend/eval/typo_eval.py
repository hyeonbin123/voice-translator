"""오타가 섞인 글자 입력의 번역(작업 T32).

사용법(backend/에서, gpu·eval 그룹을 설치하고 FLEURS 파일(eval.fleurs_download), 번역 모델(eval.mt_convert),
오타 세트(eval.typo_noise)를 준비한 상태로):
    uv run python -m eval.typo_eval convert                      # 교정 모델을 CTranslate2로 변환
    uv run python -m eval.typo_eval run --split validation --tag t32_dev
    uv run python -m eval.typo_eval run --split test --directions ko-en --candidates B --tag t32_test
    uv run python -m eval.typo_eval run --split validation --directions en-ko --candidates A \
        --translator hy-mt2-q8/split --tag t83_typo_dev                  (T83, docs/experiments.md 15)

후보와 선택 규칙은 실행 전에 docs/experiments.md 6절에 적었다. 번역 쪽은 기본으로 서버 설정과 같다:
opus-mt-tc-big, 영→한은 문장 단위. --translator를 주면 대신 eval.mt_eval의 Hy-MT2 후보 하나를 쓴다(T83,
docs/experiments.md 15절).
"""

# [파일 역할] 오타가 섞인 글자 입력의 번역 품질을 재고, 번역 전에 오타를 고치는 교정 후보를 비교하는 측정
# 스크립트다(T32, docs/experiments.md 6절). eval/typo_noise.py가 만든 세트로 깨끗·가벼운 오타·심한 오타 세
# 수준을 번역해 chrF, 교정 지연, 교정 모델의 추가 VRAM을 잰다.
# [후보] A 교정 없음(기준), B 전용 교정 모델(한국어 et5, 영어 BART), C qwen2.5 1.5B(Ollama), D qwen2.5
# 7B(참고). 판정은 영→한만 C 채택, 한→영은 A 유지였다(깨끗한 입력을 망가뜨리지 않으면서 오타 평균이 1점 넘게
# 오른 후보가 C뿐).
# [설계 이유] 교정이 멀쩡한 문장을 망가뜨리는지 보려고 깨끗한 입력도 교정을 거쳐 잰다(6절 지표). 번역은 서버
# 설정(opus-mt-tc-big, 영→한 문장 단위)을 그대로 쓴다. T83에서는 --translator로 Hy-MT2 후보를 교정 없이
# 번역하게 해, 오타에 강한 번역 모델이면 교정기를 은퇴시킬 수 있는지 보도록 했다(15절, 실제로는 앞 조건에서
# 떨어져 재지 않음).
# [연결] C의 지시문(SYSTEM)은 서버의 app/services/correction.py와 같다. 저장된 보고서는
# eval/reanalysis.py(T80)와 eval/mt_decide.py(T83)가 읽는다.

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime

# httpx: Ollama 교정 호출과 모델 내리기. sacrebleu: chrF. 서버의 번역기 구현(MarianTranslator, Hy-MT2용
# HyMtTranslator)을 그대로 쓴다.
import httpx
import sacrebleu

from app.services.interfaces import Language, ModelError, Translator
from app.services.translation import HyMtTranslator, MarianTranslator
from eval.common import DATA, MODELS, REPORTS, gpu_memory_mb

# CT2: CTranslate2로 변환한 번역·교정 모델 폴더. LEVELS: 세트의 세 수준(깨끗한 원문, 가벼운 오타, 심한 오타).
# DIRECTIONS: 방향 이름 → (원문 언어, 대상 언어).
CT2 = MODELS / "ct2"
OLLAMA_URL = "http://localhost:11434"
LEVELS = ("clean", "light", "heavy")
DIRECTIONS: dict[str, tuple[Language, Language]] = {"ko-en": ("ko", "en"), "en-ko": ("en", "ko")}
# 후보 B의 교정 모델(6절 후보): 한국어 j5ng/et5-typos-corrector(입력 앞에 '맞춤법을 고쳐주세요: '를 붙이도록
# 학습됨), 영어 oliverguhr/spelling-correction-english-base.
# 형식: 언어: (Hugging Face 저장소, 변환한 폴더 이름, 입력 앞에 붙이는 글)
SEQ2SEQ = {
    "ko": ("j5ng/et5-typos-corrector", "et5-typos-corrector", "맞춤법을 고쳐주세요: "),
    "en": ("oliverguhr/spelling-correction-english-base", "spelling-correction-english-base", ""),
}
# 후보 C·D의 고정 지시문: 오타·띄어쓰기만 고치고 뜻을 바꾸거나 번역·덧붙이지 말고 고친 문장만 답하라. 측정한
# 지시문 그대로 서버에서도 쓰므로 바꾸려면 둘을 함께 바꿔야 한다(correction.py의 주석).
LANGUAGE_NAMES = {"ko": "Korean", "en": "English"}
SYSTEM = (
    "You fix typing mistakes. Correct only the spelling, typos and spacing in the user's {language} text. "
    "Do not change the meaning, do not translate and do not add anything. Reply with the corrected text only."
)


# 후보 B: 교정 모델을 CTranslate2로 변환해 번역 모델처럼 GPU float16으로 돌리는 교정기. 토크나이저는 Hugging
# Face 원본을 쓴다.
class Seq2SeqCorrector:
    """CTranslate2로 변환한 교정 모델. 번역 모델처럼 GPU에서 돈다."""

    def __init__(self, language: Language) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        repo, folder, self.prefix = SEQ2SEQ[language]
        self.name = repo
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = ctranslate2.Translator(str(CT2 / folder), device="cuda", compute_type="float16")

    # 접두어를 붙여 토큰으로 바꾸고, beam 5·최대 128토큰(6절 후보 B 설정)으로 생성한 뒤 글로 되돌린다.
    def correct(self, text: str) -> str:
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(self.prefix + text))
        result = self.model.translate_batch([tokens], beam_size=5, max_decoding_length=128)
        ids = self.tokenizer.convert_tokens_to_ids(result[0].hypotheses[0])
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    # 모델 참조를 놓아 다음 후보 측정 전에 GPU 메모리를 돌려받게 한다.
    def close(self) -> None:
        del self.model


# 후보 C·D: Ollama의 지시 따름 LLM에 고정 지시문(system)과 입력(user)을 보내 교정문을 받는 교정기.
class OllamaCorrector:
    """Ollama가 제공하는 지시 따름 LLM에 고정 지시문 하나를 주어 쓰는 교정기(docs/experiments.md 6절)."""

    def __init__(self, model: str, language: Language) -> None:
        self.name = f"ollama/{model}"
        self.model = model
        self.system = SYSTEM.format(language=LANGUAGE_NAMES[language])
        self.client = httpx.Client(base_url=OLLAMA_URL, timeout=120)

    # temperature 0으로 같은 입력에 같은 답이 나오게 하고, num_predict 256으로 답 길이를 제한한다. 응답 앞뒤
    # 공백만 지운다. 서버의 교정(T38 안전장치: 정상 종료, 한글 섞임, 유사도 0.5)은 이 측정 뒤에 더해진 것이라
    # 여기에는 없다.
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

    # Ollama에서 모델을 내려 다음 측정의 메모리에 남지 않게 한다.
    def close(self) -> None:
        unload_ollama(self.model)


# keep_alive 0 요청으로 Ollama 모델을 내리고, 러너가 메모리를 실제로 놓을 때까지 3초
# 기다린다(eval/mt_eval.py와 같은 방식).
def unload_ollama(model: str) -> None:
    httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    time.sleep(3)  # 다음 메모리 읽기 전에 러너가 메모리를 놓을 시간을 준다


# 후보 이름 → 언어를 받아 교정기를 만드는 함수(A는 교정 없음이라 None).
CANDIDATES: dict[str, Callable[[Language], Seq2SeqCorrector | OllamaCorrector] | None] = {
    "A": None,
    "B": Seq2SeqCorrector,
    "C": lambda language: OllamaCorrector("qwen2.5:1.5b-instruct", language),
    "D": lambda language: OllamaCorrector("qwen2.5:7b-instruct", language),
}


# data/typo/<split>.jsonl을 읽는다(앞 limit쌍만 쓸 수 있음).
def load_rows(split: str, limit: int | None) -> list[dict]:
    path = DATA / "typo" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


# 방향의 번역기를 만든다. 기본 opus는 서버 설정과 같게 영→한만 문장 단위(T17). 다른 이름이면 eval/mt_eval.py의
# 후보 표에서 Hy-MT2 후보를 가져오고, 그 후보가 이 방향을 번역하지 않으면 멈춘다. Hy-MT2는 prepare에서 기록된
# 빌드인지 확인한다(T83).
def translator_for(direction: str, name: str = "opus") -> Translator:
    """서버의 opus 설정, 또는 준비·확인을 마친 eval.mt_eval의 Hy-MT2 후보(T83)."""
    source, target = DIRECTIONS[direction]
    if name == "opus":
        folder = CT2 / f"opus-mt-tc-big-{source}-{target}"
        return MarianTranslator(folder, source, target, by_sentence=direction == "en-ko")
    from eval.mt_eval import CANDIDATES

    factory, directions = CANDIDATES[name]
    if (source, target) not in directions:
        raise SystemExit(f"{name} does not translate {direction}")
    translator = factory()
    if isinstance(translator, HyMtTranslator):
        translator.prepare()  # 기록된 것과 다른 빌드면 거부한다
    return translator


# 한 방향 × 한 교정 후보를 세 수준 모두 재서 결과 dict를 돌려준다. 수준마다 corpus chrF, 교정 결과를 깨끗한
# 원문과 견준 입력 chrF(참고용, 규칙에 없음), 교정·전체 지연 p50, 빈 교정 수, 번역 실패 수, 문장별 항목을
# 남긴다.
def evaluate(direction: str, candidate: str, rows: list[dict], translator: Translator) -> dict:
    source, target = DIRECTIONS[direction]
    factory = CANDIDATES[candidate]
    references = [row[target] for row in rows]
    # VRAM은 교정 모델을 올리기 전후의 GPU 전체 사용량 차이다. Ollama 후보는 먼저 모델을 내린 상태에서 재고,
    # 첫 교정 호출(시간에 넣지 않는 준비)로 모델을 올린 뒤 다시 읽는다.
    if factory is not None and candidate in ("C", "D"):
        unload_ollama(factory(source).model)  # 모델을 내린 상태에서 VRAM을 읽는다
    _, before = gpu_memory_mb()
    corrector = factory(source) if factory else None
    if corrector is not None:
        corrector.correct(rows[0][source])  # 모델을 올리고(Ollama) 준비시킨다. 시간은 재지 않는다
    _, after = gpu_memory_mb()
    result: dict = {
        "direction": direction,
        "candidate": candidate,
        "corrector": corrector.name if corrector else None,
        "translator": translator.model_name,
        "vram_mb": round(after - before),
        "levels": {},
    }
    for level in LEVELS:
        key = source if level == "clean" else f"{source}_{level}"
        hypotheses, corrected_texts, correction_s, total_s, items, empty, failures = [], [], [], [], [], 0, 0
        for row in rows:
            # 교정 시간과 교정+번역 전체 시간을 따로 잰다. 교정이 빈 답을 내면 입력 그대로 번역하고 수를
            # 센다(서버도 교정 실패 때 입력대로 번역한다). 번역 실패는 빈 번역으로 두고 센다.
            start = time.perf_counter()
            corrected = corrector.correct(row[key]) if corrector else row[key]
            if not corrected.strip():
                empty, corrected = empty + 1, row[key]  # 빈 교정이면 원래 입력으로 되돌린다
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
            # 참고용(규칙에 없음): 교정한 입력이 깨끗한 원문에 얼마나 가까워졌는지.
            "input_chrf": sacrebleu.corpus_chrf(corrected_texts, [[row[source] for row in rows]]).score,
            "correction_p50": statistics.median(correction_s),
            "total_p50": statistics.median(total_s),
            "empty_corrections": empty,
            "failures": failures,
            "items": items,
        }
    # Ollama 후보는 모델이 아직 올라가 있을 때 버전·digest·GPU 크기를 기록하고, 그다음 내린다.
    if isinstance(corrector, OllamaCorrector):
        from eval.mt_eval import ollama_state

        result["ollama"] = ollama_state(corrector.model)  # 교정기가 아직 올라가 있을 때 읽는다
    if corrector is not None:
        corrector.close()
    gc.collect()
    return result


# 결과를 reports/typo_<tag>_<UTC 시각>.json·.md로 쓴다. 방향별 표: 교정 모델, 추가 VRAM, 세 수준 chrF와 오타
# 평균(가벼움·심함 평균, 선택 규칙 값), 교정 p50(세 수준 중 가장 큰 값), 심한 오타의 전체 p50, 입력 chrF. 교정
# 후보마다 심한 오타 교정 예 3개도 적는다.
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
        f"- GPU: {gpu_name}. 번역: "
        + ("서버 설정(opus-mt-tc-big, 영→한 문장 단위)" if args.translator == "opus" else args.translator),
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


# convert 명령: 후보 B 교정 모델 두 개를 Hugging Face에서 받아 CTranslate2 float16으로 변환한다. 이미 있으면
# --force 없이는 건너뛴다.
def convert(force: bool) -> None:
    from ctranslate2.converters import TransformersConverter

    for repo, folder, _ in SEQ2SEQ.values():
        target = CT2 / folder
        if target.exists() and not force:
            print(f"{target} exists, skipped")
            continue
        print(f"converting {repo} ...", flush=True)
        TransformersConverter(repo).convert(str(target), quantization="float16", force=True)


# 명령줄 진입점. run은 방향마다 번역기를 한 번 만들고 준비 호출을 한 뒤 후보를 차례로 잰다. 번역기는 같은
# 방향의 모든 후보가 공유하므로 후보 간 차이는 교정에서만 나온다.
def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser("convert")
    convert_parser.add_argument("--force", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--split", choices=["validation", "test"], default="validation")
    run_parser.add_argument("--directions", nargs="+", choices=list(DIRECTIONS), default=list(DIRECTIONS))
    run_parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    run_parser.add_argument("--limit", type=int, default=None, help="앞에서부터 N개 문장 쌍만 쓴다")
    run_parser.add_argument("--tag", default="run")
    run_parser.add_argument(
        "--translator",
        default="opus",
        choices=["opus", "hy-mt2-q8", "hy-mt2-q8/split", "hy-mt2-q4", "hy-mt2-q4/split"],
        help="서버의 opus 설정, 또는 eval.mt_eval의 Hy-MT2 후보(T83)",
    )
    args = parser.parse_args()

    if args.command == "convert":
        convert(args.force)
        return
    gpu_name, _ = gpu_memory_mb()
    rows = load_rows(args.split, args.limit)
    results = []
    for direction in args.directions:
        translator = translator_for(direction, args.translator)
        source, target = DIRECTIONS[direction]
        translator.translate(rows[0][source], source, target)  # 준비용 호출, 시간을 재지 않는다
        for candidate in args.candidates:
            print(f"evaluating {direction} / {candidate} ...", flush=True)
            results.append(evaluate(direction, candidate, rows, translator))
        del translator
        gc.collect()
    print(f"written to {write_report(results, args, gpu_name, len(rows))}")


if __name__ == "__main__":
    main()
