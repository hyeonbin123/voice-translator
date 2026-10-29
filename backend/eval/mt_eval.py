"""FLEURS 병렬 문장으로 번역을 평가한다.

backend/에서 gpu·eval 그룹을 설치하고 모델을 변환한(eval.mt_convert) 뒤 쓰는 법:
    uv run python -m eval.mt_eval --split validation --tag t3_dev
    uv run python -m eval.mt_eval --split validation --tag t17_dev --models opus-mt-tc-big-ko-en \
        opus-mt-tc-big-ko-en/split opus-mt-tc-big-en-ko opus-mt-tc-big-en-ko/split
    uv run python -m eval.mt_eval --split validation --tag t83_dev --models opus-mt-tc-big-ko-en \
        opus-mt-tc-big-en-ko/split hy-mt2-q8 hy-mt2-q8/split
    uv run --no-sync python -m eval.mt_eval --split validation --tag t108_dev --models opus-mt-tc-big-ko-en \
        opus-mt-tc-big-ko-en/b8 opus-mt-tc-big-ko-en/b8-mbr opus-mt-tc-big-en-ko/split \
        opus-mt-tc-big-en-ko/split/b8 opus-mt-tc-big-en-ko/split/b8-mbr

Hy-MT2 시험: Hy-MT2 모델은 eval.hymt_setup이 만드는 그대로 Ollama에 있어야 하고, Ollama에 있는 빌드가
docs/experiments.md 15절에 기록된 빌드가 아니면 그 후보는 거부된다.

디코딩 시험(docs/experiments.md 20절): '/b8'은 같은 모델의 빔 8 1-best, '/b8-mbr'은 빔 8의 8-best에서 문장
chrF MBR로 고른 번역(eval/mbr.py, MBR 계산은 문장당 시간에 들어감)이다. 판정은 eval.mbr decide.

후보와 선택 규칙은 실행 전에 docs/experiments.md에 적었다.
"""

# [파일 역할] 글자 입력 번역 모델을 FLEURS 병렬 문장으로 재는 측정 스크립트다. 후보마다 번역 품질(chrF),
# 문장당 지연(p50·p95), GPU 메모리, 출력 점검(다른 문자·설명형·빈 출력)을 재서
# backend/eval/reports/mt_<tag>_<시각>.json·.md로 남긴다.
# [왜 필요한가] 이 프로젝트는 모델을 측정해서 고른다. 후보와 판단 규칙은 측정 전에 docs/experiments.md에
# 적고(2절 T3, 2-1절 T17, 15절 T83, 20절 디코딩 시험), 이 스크립트가 같은 조건으로 모든 후보를 재서 그 규칙에
# 넣을 숫자를 만든다.
# [흐름] load_pairs로 문장 ID마다 한·영 쌍을 만든다 → 후보마다 evaluate가 모델을 올리고 메모리를 읽은 뒤,
# 방향마다 준비 호출 1번을 빼고 전 문장을 번역·채점한다 → write_report가 JSON(문장별 결과 포함)과 사람이 읽는
# Markdown 표를 쓴다.
# [연결] 저장된 JSON은 eval/mt_failures.py(누락 신호), eval/mt_checks.py, eval/mt_decide.py(T83 판정),
# eval/reanalysis.py(T80 재분석), eval/mbr.py(20절 판정)가 다시 읽는다. load_pairs는 eval/typo_noise.py가,
# CANDIDATES와 ollama_state는 eval/typo_eval.py가 가져다 쓴다.

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime

# httpx: Ollama HTTP API(모델 내리기, 버전·digest·GPU 메모리 조회)를 직접 부른다.
# pyarrow.parquet: data/fleurs/에 받아 둔 FLEURS parquet에서 필요한 열(id, raw_transcription)만 읽는다.
# sacrebleu: chrF 계산. 한국어는 띄어쓰기·어미 변화가 많아 단어 단위 BLEU보다 글자 단위 chrF가 맞다고
# 정했다(docs/experiments.md 2절 지표).
import httpx
import pyarrow.parquet as pq
import sacrebleu

# 서버가 실제로 쓰는 번역기 구현을 그대로 가져와 잰다. 측정한 코드와 서비스 코드가 달라지지 않게 하려는
# 것이다. split_sentences는 T17에서 서버가 영→한에 쓰는 문장 나누기와 같은 함수다.
# eval.common: 데이터·모델·리포트 경로와 GPU 전체 사용량 읽기. eval.hymt_setup.QUANTS: Hy-MT2 양자화별 Ollama
# 모델 이름과 측정한 빌드의 digest. eval.mt_checks.check_items: T83 규칙 (b)의 출력 점검.
# eval.mbr.MbrTranslator: 20절의 '/b8-mbr' 팔(빔 8의 n-best에서 문장 chrF MBR로 고르는 번역기).
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
from eval.mbr import MbrTranslator
from eval.mt_checks import check_items

# 호스트에서 도는 Ollama의 기본 주소. Qwen과 Hy-MT2 후보는 이 Ollama로 부른다.
OLLAMA_URL = "http://localhost:11434"
# T3의 일반 LLM 후보(7B, 4비트). 14B는 VRAM 약 9GB라 음성 인식·합성과 함께 올릴 수 없어 후보에서
# 뺐다(docs/experiments.md 2절).
QWEN = "qwen2.5:7b-instruct"
# 번역 방향을 (원문 언어, 대상 언어) 튜플로 둔다. 후보마다 지원하는 방향 목록을 적을 때 쓴다.
KO_EN: tuple[Language, Language] = ("ko", "en")
EN_KO: tuple[Language, Language] = ("en", "ko")
# eval/mt_convert.py가 CTranslate2로 변환해 둔 모델 폴더(data/models/ct2).
CT2 = MODELS / "ct2"


def hymt(quant: str, by_sentence: bool = False) -> HyMtTranslator:
    """Ollama의 Hy-MT2 후보 번역기를 만든다. 기록된 빌드가 아니면 거부된다(docs/experiments.md 15절).

    입력: 양자화 이름(q8_0·q4_k_m)과 문장 단위 여부. 출력: HyMtTranslator. expected_digest에 측정한 빌드의
    digest를 넘겨, Ollama에 다른 빌드가 들어 있으면 prepare에서 멈추게 한다. 같은 이름의 모델이라도
    Modelfile이 다르면 결과가 달라지므로 기록된 빌드로만 재기 위해서다.
    """
    _, _, _, model, digest = QUANTS[quant]
    return HyMtTranslator(model, OLLAMA_URL, timeout_s=120, by_sentence=by_sentence, expected_digest=digest)


# 후보 표: 이름 → (번역기를 만드는 함수, 이 후보가 번역하는 방향 목록). 만드는 함수를 람다로 둔 것은
# --models로 고른 후보만, 그리고 차례가 왔을 때만 GPU에 올리기 위해서다. 키는 --models 선택지가 된다.
# - opus-mt-ko-en, opus-mt-tc-big-*, nllb, qwen2.5-7b: T3 후보(docs/experiments.md 2절).
# - '/split' 이름: 같은 모델이 문장마다 따로 번역하는 T17 후보 B.
# - hy-mt2-*: T83 후보. Q4_K_M은 Q8_0 팔이 지연에서만 떨어졌을 때 쓰는 대체 팔이다.
# - '/b8', '/b8-mbr' 이름: 같은 모델의 디코딩만 바꾼 20절 팔(빔 8의 1-best, 빔 8의 8-best + MBR).
CANDIDATES: dict[str, tuple[Callable[[], Translator], list[tuple[Language, Language]]]] = {
    "opus-mt-ko-en": (lambda: MarianTranslator(CT2 / "opus-mt-ko-en", "ko", "en"), [KO_EN]),
    "opus-mt-tc-big-ko-en": (lambda: MarianTranslator(CT2 / "opus-mt-tc-big-ko-en", "ko", "en"), [KO_EN]),
    "opus-mt-tc-big-en-ko": (lambda: MarianTranslator(CT2 / "opus-mt-tc-big-en-ko", "en", "ko"), [EN_KO]),
    "nllb-200-distilled-600M": (lambda: NllbTranslator(CT2 / "nllb-200-distilled-600M"), [KO_EN, EN_KO]),
    "qwen2.5-7b": (lambda: OllamaTranslator(QWEN, OLLAMA_URL), [KO_EN, EN_KO]),
    # 고른 모델이 한 문장씩 번역하는 후보(docs/experiments.md 2-1절)
    "opus-mt-tc-big-ko-en/split": (
        lambda: MarianTranslator(CT2 / "opus-mt-tc-big-ko-en", "ko", "en", by_sentence=True),
        [KO_EN],
    ),
    "opus-mt-tc-big-en-ko/split": (
        lambda: MarianTranslator(CT2 / "opus-mt-tc-big-en-ko", "en", "ko", by_sentence=True),
        [EN_KO],
    ),
    # Ollama가 서비스하는 Hy-MT2-1.8B(docs/experiments.md 15절). Q4_K_M은 지연 대체 팔로만 쓴다.
    "hy-mt2-q8": (lambda: hymt("q8_0"), [KO_EN, EN_KO]),
    "hy-mt2-q8/split": (lambda: hymt("q8_0", by_sentence=True), [EN_KO]),
    "hy-mt2-q4": (lambda: hymt("q4_k_m"), [KO_EN, EN_KO]),
    "hy-mt2-q4/split": (lambda: hymt("q4_k_m", by_sentence=True), [EN_KO]),
    # 같은 opus-mt-tc-big의 디코딩만 바꾼 팔(docs/experiments.md 20절). 기준은 위의 지금 서비스 설정(한→영
    # 통째로, 영→한 문장 단위, 빔 4의 1-best)이고, 팔도 같은 방향 설정을 따른다. '/b8'은 빔 8의 1-best(빔 크기
    # 효과만), '/b8-mbr'은 빔 8의 가설 8개 중 문장 chrF MBR로 고른 번역이다(MBR 계산은 translate() 안이라
    # 문장당 시간에 들어감). MBR 팔의 안쪽 번역기도 빔 8로 만들어 두지만, 후보는 MbrTranslator가 넘기는 빔
    # 크기로 만든다.
    "opus-mt-tc-big-ko-en/b8": (
        lambda: MarianTranslator(CT2 / "opus-mt-tc-big-ko-en", "ko", "en", beam_size=8),
        [KO_EN],
    ),
    "opus-mt-tc-big-ko-en/b8-mbr": (
        lambda: MbrTranslator(
            MarianTranslator(CT2 / "opus-mt-tc-big-ko-en", "ko", "en", beam_size=8), beam=8, n=8
        ),
        [KO_EN],
    ),
    "opus-mt-tc-big-en-ko/split/b8": (
        lambda: MarianTranslator(CT2 / "opus-mt-tc-big-en-ko", "en", "ko", beam_size=8, by_sentence=True),
        [EN_KO],
    ),
    "opus-mt-tc-big-en-ko/split/b8-mbr": (
        lambda: MbrTranslator(
            MarianTranslator(CT2 / "opus-mt-tc-big-en-ko", "en", "ko", beam_size=8, by_sentence=True),
            beam=8,
            n=8,
        ),
        [EN_KO],
    ),
}
# Ollama 후보 → Ollama 안의 모델 이름. evaluate가 측정 앞뒤로 이 모델을 내려, GPU 메모리를 모델이 없는
# 상태에서 읽고 다음 후보 측정에 메모리가 남지 않게 한다.
OLLAMA_MODELS = {
    "qwen2.5-7b": QWEN,
    "hy-mt2-q8": QUANTS["q8_0"][3],
    "hy-mt2-q8/split": QUANTS["q8_0"][3],
    "hy-mt2-q4": QUANTS["q4_k_m"][3],
    "hy-mt2-q4/split": QUANTS["q4_k_m"][3],
}


def load_pairs(split: str, limit: int | None) -> list[dict]:
    """문장 ID마다 한 행. FLEURS는 화자마다 같은 문장을 되풀이한다.

    FLEURS의 한국어·영어 parquet에서 같은 문장 ID의 원문 쌍을 만든다. 입력: 분할 이름, 앞에서부터 쓸
    개수(None이면 전부). 출력: {id, ko, en} 목록. raw_transcription(문장부호가 있는 원문)을 쓰는 것은 번역
    참조로 쓰기 위해서다. 음성 인식 평가용 transcription은 소문자·문장부호 없음이라 번역 참조로 맞지
    않는다(docs/experiments.md 평가 데이터). 같은 문장을 여러 화자가 읽어 행이 겹치므로 dict로 ID마다
    하나만 남기고, 두 언어에 모두 있는 ID만 쓴다 → validation 129쌍, test 270쌍. ID로 정렬해 실행마다
    순서가 같다. 호출 위치: 이 파일의 main, eval/typo_noise.py.
    """
    text: dict[str, dict[int, str]] = {}
    for language, config in FLEURS_CONFIG.items():
        path = DATA / "fleurs" / config / f"{split}.parquet"
        rows = pq.read_table(path, columns=["id", "raw_transcription"]).to_pylist()
        text[language] = {row["id"]: row["raw_transcription"].strip() for row in rows}
    ids = sorted(text["ko"].keys() & text["en"].keys())
    pairs = [{"id": i, "ko": text["ko"][i], "en": text["en"][i]} for i in ids]
    return pairs[:limit] if limit else pairs


# Ollama에 keep_alive 0으로 빈 생성 요청을 보내 모델을 메모리에서 내린다. 그 뒤 3초 기다려 러너가 메모리를
# 실제로 돌려준 다음 GPU 사용량을 읽게 한다.
def unload_ollama(model: str = QWEN) -> None:
    httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    time.sleep(3)


def ollama_state(model: str) -> dict:
    """Ollama 버전, 모델 digest, Ollama가 그 모델을 위해 GPU에 둔 메모리(/api/ps).

    보고서에 남길 Ollama 상태다: Ollama 버전, 모델 digest(/api/tags), 모델이 올라가 있으면 GPU에 둔 크기와
    문맥 길이(/api/ps). 어떤 빌드·버전으로 쟀는지 기록해 두어야 나중에 같은 조건인지 확인할 수 있다.
    모델이 올라가 있지 않으면 크기는 0, 문맥은 None이 된다. 호출 위치: 이 파일의 evaluate,
    eval/typo_eval.py.
    """
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


# 후보 하나를 재서 결과 dict를 돌려준다. 결과에는 모델 이름, VRAM 차이, (Ollama면) Ollama 상태, 방향별
# chrF·지연·실패 수·출력 점검과 문장별 항목이 들어간다.
# VRAM은 장치 전체 사용량의 적재 전후 차이다. 그래서 다른 프로그램의 변화나 첫 모델의 CUDA 초기화분이 섞일 수
# 있고, 수백 MB 단위 차이는 비교하지 않는다(docs/experiments.md 2절 '결과를 읽을 때 본 것').
def evaluate(name: str, pairs: list[dict]) -> dict:
    factory, directions = CANDIDATES[name]
    ollama_model = OLLAMA_MODELS.get(name)
    # Ollama 후보는 먼저 모델을 내려 '모델이 없는 상태'의 메모리를 기준으로 삼는다.
    if ollama_model:
        unload_ollama(ollama_model)
    _, before = gpu_memory_mb()
    translator = factory()
    # Hy-MT2는 prepare에서 Ollama의 빌드 digest를 확인하고(다르면 예외로 측정을 멈춤) 모델을 올린다.
    if isinstance(translator, HyMtTranslator):
        translator.prepare()
    # Ollama는 첫 요청 때 모델을 올리므로, 한 번 번역을 시켜 모델이 GPU에 올라간 뒤에 메모리를 읽는다.
    if ollama_model:
        source, target = directions[0]
        translator.translate(pairs[0][source], source, target)
    _, after = gpu_memory_mb()
    result: dict = {"model": name, "model_name": translator.model_name, "vram_mb": round(after - before)}
    # opus 번역기(MarianTranslator)와 MBR 번역기는 디코딩 설정(빔 크기, 가설 수, 고르는 방법, 문장 단위)을
    # 함께 남긴다. 판정 도구(eval/mbr.py)가 팔이 등록한 설정으로 돌았는지 이것으로 확인한다.
    if decoding := getattr(translator, "decoding", None):
        result["decoding"] = dict(decoding)
    if ollama_model:
        result["ollama"] = ollama_state(ollama_model)
    result["directions"] = {}

    # 방향마다 측정한다. 첫 호출은 준비(커널 준비·캐시)라 느리므로 시간에 넣지 않는다(docs/experiments.md 2절
    # 지표).
    for source, target in directions:
        translator.translate(pairs[0][source], source, target)  # 준비 호출, 시간에 넣지 않음
        hypotheses, latencies, items, failures = [], [], [], 0
        for pair in pairs:
            # 문장 하나의 번역 시간을 잰다. 문장 단위 후보는 나뉜 문장들을 모두 번역한 합이 한 입력의 시간이
            # 된다.
            # 번역기가 ModelError를 내면 빈 번역으로 두고 실패 수를 센다. 측정 전체를 멈추지 않고, 빈 결과는
            # chrF에서 그대로 감점되며 출력 점검의 '빈 출력'으로도 잡힌다.
            start = time.perf_counter()
            try:
                hypothesis = translator.translate(pair[source], source, target)
            except ModelError:
                hypothesis, failures = "", failures + 1
            latencies.append(time.perf_counter() - start)
            hypotheses.append(hypothesis)
            # 문장별 결과를 남긴다. 문장 chrF는 나쁜 문장을 찾아보는 용도이고, 'sentences'(원문이 몇 문장으로
            # 나뉘는지)는 T17에서 여러 문장 입력이 몇 개인지 보려고 남긴다. 원문·참조·번역문은 뒤의
            # 재분석(T80)·COMET 채점이 모델을 다시 돌리지 않고 쓰게 한다.
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
            # Hy-MT2는 마지막 번역에서 Ollama가 알려 준 호출 정보(done_reason, 토큰 수)를 함께 남긴다.
            # 256토큰에서 잘린 답을 세는 데 쓴다.
            if isinstance(translator, HyMtTranslator):
                items[-1]["ollama_calls"] = list(translator.calls)
            # MBR 팔은 문장마다 후보 수와 MBR이 고른 위치(0 = 엔진 점수가 가장 높은 후보)를 남긴다. MBR이
            # 실제로 얼마나 바꿨는지 읽는 참고 수치이고, 번역이 실패한 입력은 빈 목록이다.
            if isinstance(translator, MbrTranslator):
                items[-1]["mbr"] = list(translator.picks)
        # 방향별 요약. chrF는 문장 점수의 평균이 아니라 전체 합산(corpus_chrf)이다. p95는 quantiles(n=20)의
        # 19번째 경계(인덱스 18)다.
        values = {
            "chrf": sacrebleu.corpus_chrf(hypotheses, [[pair[target] for pair in pairs]]).score,
            "latency_p50": statistics.median(latencies),
            "latency_p95": statistics.quantiles(latencies, n=20)[18],
            "failures": failures,
            "count": len(pairs),
            # 판단 규칙 (b): 다른 문자·설명형 출력과 빈 출력(eval/mt_checks.py).
            "checks": check_items(items, f"{source}-{target}"),
            "items": items,
        }
        # done_reason이 'length'인 호출 = num_predict(256토큰) 한도에서 잘린 답. 잘린 답도 번역으로 쓰되 수를
        # 센다(docs/experiments.md 15절 서빙 방식).
        if isinstance(translator, HyMtTranslator):
            calls = [call for item in items for call in item.get("ollama_calls", [])]
            values["cut_at_cap"] = sum(call.get("done_reason") == "length" for call in calls)
        result["directions"][f"{source}-{target}"] = values

    # 다음 후보를 올리기 전에 이 후보를 지우고 가비지 컬렉션을 돌려 GPU 메모리를 돌려받는다. Ollama 후보는
    # Ollama 쪽 모델도 내린다.
    del translator
    gc.collect()
    if ollama_model:
        unload_ollama(ollama_model)
    return result


# 결과를 reports/mt_<tag>_<UTC 시각>.json(문장별 결과까지 전부)과 .md(사람이 읽는 요약)로 쓰고 .md 경로를
# 돌려준다.
# Markdown에는 측정 조건(데이터·GPU·디코딩·chrF 설정), 방향별 후보 표, Ollama 빌드 정보, 후보·방향마다 chrF가
# 가장 낮은 5문장을 넣는다. 낮은 문장을 함께 적는 것은 숫자만으로 보이지 않는 실패(문장 누락, 다른 언어
# 섞임)를 사람이 바로 읽게 하려는 것이다.
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
    # 20절의 빔 8 팔을 쟀으면 그 디코딩을 밝힌다(위 '빔 4'는 기준 팔에만 맞으므로). 없으면 머리말은 지금까지와
    # 같다.
    if any("/b8" in result["model"] for result in results):
        lines.append(
            "- 디코딩 시험 팔(docs/experiments.md 20절): 이름이 '/b8'로 끝나면 같은 모델의 빔 8 1-best, "
            "'/b8-mbr'로 끝나면 빔 8의 가설 8개 중 문장 chrF MBR로 고른 번역(MBR 계산이 지연에 들어감)"
        )
    # 방향마다 표 한 개. 그 방향을 번역하지 않는 후보는 directions에 없으므로 행이 빠진다.
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


# 명령줄 진입점. --split으로 validation(고르기)·test(확인)를 고르고, --models로 잴 후보를, --limit로 앞
# N쌍만(빠른 점검용), --tag로 보고서 이름을 정한다.
# GPU 이름은 보고서에 적으려고 처음에 한 번 읽는다. 후보는 하나씩 차례로 재서 서로의 GPU 메모리가 겹치지 않게
# 한다.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--models", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--limit", type=int, default=None, help="앞 N개 문장 쌍만 쓴다")
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
