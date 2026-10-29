"""영→한 말투 후처리 시험(docs/experiments.md 21절)의 측정(run)과 판정(decide).

run은 저장된 eval.mt_eval 보고서의 영→한 번역문에 app/services/register.py의 후처리(서비스의
KO_REGISTER=formal과 같은 함수)를 걸어, 원래 출력(asis)과 후처리 출력(formal)을 mt_eval 보고서 모양 하나로
남긴다. 번역 모델은 돌리지 않는다. 같은 실행에서 사용자 확인용 바뀐 문장 목록(커밋하지 않는 work/)과 COMET-22
입력 파일을 쓴다. decide는 그 보고서와 COMET-22 점수에 21절의 자동 관문(말투 일관성·COMET-22 비열등·후처리
시간)을 적용한다. 사람 문법 확인(21절 기준표)은 이 도구가 하지 않으므로, 자동 관문을 통과해도 그것만으로
채택하지 않는다.

backend/에서(run은 형태소 분석기 python-mecab-ko가 필요해 uv.lock과 같은 판을 얹는다):
    uv run --no-sync --with python-mecab-ko==1.3.7 --with python-mecab-ko-dic==2.1.1.post2 \\
        python -m eval.register_eval run --report eval/reports/mt_t17_dev_20260913_054853.json \\
        --model opus-mt-tc-big-en-ko/split --tag t109_dev
    ../work/comet-venv/Scripts/python.exe eval/comet_score.py score \\
        --input ../work/register/comet_input_t109_dev.jsonl \\
        --output eval/reports/register_t109_dev_<시각>_comet22.json --device cuda --precision fp16
    uv run --no-sync python -m eval.register_eval decide \\
        --report eval/reports/register_t109_dev_<시각>.json \\
        --comet eval/reports/register_t109_dev_<시각>_comet22.json --tag t109_dev
"""

# [파일 역할] 말투 후처리 시험의 두 단계: (1) run — 저장된 번역문에 후처리를 걸고 출력마다 걸린 시간을 재어
# 보고서·바뀐 문장 목록·COMET 입력을 쓴다 (2) decide — 측정 전에 21절에 적은 판정 규칙을 그 결과에 적용한다.
# [설계 이유] 후처리는 번역 뒤에 붙는 결정적 단계라 같은 번역문에 걸면 서비스와 같은 결과가 나온다. 그래서
# 번역을 다시 돌리지 않고 기록된 출력(2-1절 보고서, 또는 20절이 고른 팔의 출력)을 쓴다. 말투
# 분류(classify_end)는 후처리가 쓰는 형태소 분석기를 쓰지 않는 글자 규칙이다: 분석기가 틀려 바꾸지 못한 문장
# 끝이 지표에서 그대로 드러나게(같은 분석기로 세면 분석기의 오류가 지표에서도 가려진다). COMET 키·점수 확인과
# 구간은 15·20절 도구(eval/mt_decide.py, eval/mbr.py)와 13절의 짝지은 붓스트랩(eval/significance.py,
# 10,000번·시드 12345)을 그대로 쓴다.
# [연결] 입력은 eval/mt_eval.py 보고서, 후처리는 app/services/register.py, COMET 채점은
# eval/comet_score.py(별도 환경 work/comet-venv). 결과는 eval/reports/register_<태그>_<시각>.json·.md(run)와
# register_decide_<태그>_<시각>.json·.md(decide). 테스트: tests/test_register_eval.py.

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

# app.services.register: 서비스와 같은 후처리 함수(load_formalizer, 분석기는 함수 안에서 import).
# split_sentences: 서비스가 영→한을 문장으로 나누는 규칙(2-1절 B) — 말투를 문장마다 셀 때 같은 경계를 쓴다.
# eval.mbr: 20절 도구의 COMET 메타 확인과 문장 일치 확인. eval.mt_decide: 15절 도구의 COMET 입력 모으기·점수
# 찾기·구간 dict·글 모양. eval.significance: 13절의 붓스트랩 반복 수·시드와 짝지은 구간 함수. sacrebleu는 eval
# 그룹에만 있어 쓰는 함수 안에서 import한다.
from app.services.interfaces import ModelError
from app.services.register import load_formalizer
from app.services.translation import split_sentences
from eval.common import PROJECT, REPORTS
from eval.mbr import _check_comet_meta, _same_sentences
from eval.mt_decide import _comet, _fmt, _interval, comet_input
from eval.significance import ROUNDS, SEED, corpus_difference, mean_difference

# 21절에 등록한 값. MODEL: 기본 대상(2-1절 보고서의 지금 서비스 설정). FORMAL: 후처리 출력의 이름 꼬리.
MODEL = "opus-mt-tc-big-en-ko/split"
DIRECTION = "en-ko"
FORMAL = "+formal"
# 판정 문턱(21절): (a) 모든 문장 끝이 높임인 출력 ≥ 98%이고 섞인 출력 0, (c) COMET-22(×100) 차이 구간 하한 >
# −0.5, (d) 출력 하나의 후처리 시간 p50 ≤ 5ms. 비율은 부동소수점 대신 정수로 견준다(높임 출력 × 100 ≥ 98 ×
# 출력 수).
MIN_HONORIFIC_PERCENT = 98
MAX_MIXED = 0
MIN_COMET_LOW = -0.5
MAX_P50_MS = 5.0
# 후처리 시간을 재기 전에 한 번 돌리는 문장(분석기의 첫 호출 비용을 잰 시간에서 빼려고). 시간에 넣지 않는다.
WARM_UP = "그는 집에 갔다."

# 문장 끝 분류에서 떼어 보는 끝 글자들: 문장부호, 닫는 따옴표·괄호, 공백. 남은 글 끝의 한글 덩어리를 본다.
_END_MARKS = ".!?…。！？\"'”’)]}」』〉》】 \t\r\n"
_HANGUL_TAIL = re.compile(r"[가-힣]+$")
_SYLLABLE_FIRST = 0xAC00
_FINAL_B = 17
_KINDS = ("formal", "polite", "plain", "other")


def _has_final_b(syllable: str) -> bool:
    """한글 음절의 받침이 ㅂ인지(습니다·갑니다·갑시다의 '습'·'갑'을 가르는 데 씀)."""
    return (ord(syllable) - _SYLLABLE_FIRST) % 28 == _FINAL_B


def classify_end(sentence: str) -> str:
    """문장 하나의 끝 말투: formal(합쇼체), polite(해요체), plain(해라체 평서), other(그 밖).

    끝의 문장부호·닫는 따옴표·괄호·공백을 떼고 글 끝의 한글 덩어리를 본다(형태소 분석기를 쓰지 않음):
    - formal: …니다·…니까 앞 음절이 '습'이거나 받침 ㅂ(갑니다·입니까. '아니다'·'다니다'는 아님), …십시오,
      받침 ㅂ 음절 + 시다(청유 …ㅂ시다)
    - polite: …요·…죠
    - plain: 그 밖에 …다로 끝남(갔다·아니다·이다)
    - other: 한글로 끝나지 않음(숫자·로마자), 의문·명령·청유의 해라체(…까·…라·…자)와 명사로 끝남
    '다'로 끝나는 명사(바다)는 plain으로 센다(글자 규칙의 한계, 21절 한계에 적음).
    """
    match = _HANGUL_TAIL.search(sentence.rstrip(_END_MARKS))
    if not match:
        return "other"
    tail = match.group()
    # 합쇼체 평서·의문은 '니다'·'니까' 앞 음절이 받침 ㅂ이다('습'도 받침 ㅂ). 청유 '…ㅂ시다'도 같은 모양이다.
    before_b = len(tail) >= 3 and _has_final_b(tail[-3])
    if (tail.endswith(("니다", "니까", "시다")) and before_b) or tail.endswith("십시오"):
        return "formal"
    if tail.endswith(("요", "죠")):
        return "polite"
    if tail.endswith("다"):
        return "plain"
    return "other"


def consistency(outputs: Sequence[str]) -> dict:
    """출력들의 말투 일관성(21절 지표 (a)).

    출력마다 서비스와 같은 split_sentences로 문장을 나눠 문장 끝을 classify_end로 센다. all_honorific은 모든
    문장 끝이 높임(formal 또는 polite)인 출력 수, ratio는 그 비율, mixed는 높임 끝과 해라체 평서 끝을 함께
    가진 출력 수, ends는 문장 끝 종류별 수다.
    """
    ends = dict.fromkeys(_KINDS, 0)
    honorific = mixed = 0
    for output in outputs:
        kinds = [classify_end(sentence) for sentence in split_sentences(output)] or ["other"]
        for kind in kinds:
            ends[kind] += 1
        polite = [kind in ("formal", "polite") for kind in kinds]
        honorific += all(polite)
        mixed += any(polite) and "plain" in kinds
    count = len(outputs)
    return {
        "outputs": count,
        "all_honorific": honorific,
        "ratio": honorific / count if count else 0.0,
        "mixed": mixed,
        "ends": ends,
    }


def _values(results: list[dict], model: str) -> dict:
    """mt_eval 보고서에서 모델 하나의 영→한 결과. 없으면 측정하지 않고 멈춘다."""
    for result in results:
        if result.get("model") == model and DIRECTION in result.get("directions", {}):
            return result["directions"][DIRECTION]
    raise SystemExit(f"the report has no {model} {DIRECTION} run")


def run(
    results: list[dict],
    model: str,
    formalize: Callable[[str], str],
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict]:
    """저장된 번역문에 후처리를 걸어 두 출력(asis, formal)을 mt_eval 보고서 모양 목록으로 돌려준다.

    파일은 쓰지 않는다(main이 씀). 출력마다 후처리 한 번의 걸린 시간(ms, 형태소 분석 포함)을 잰다. 재기 전에
    WARM_UP 문장으로 한 번 돌려 둔다. 항목에는 ID·원문·참조·번역문(·문장 수)만 옮긴다: 원래 보고서의 문장
    chrF·번역 시간은 후처리 출력에 맞지 않아 뺀다. formal 쪽에는 출력마다 바뀌었는지(changed)와 시간, 전체의
    바뀐 출력 수, 바뀐 출력의 글자 수 증가 중앙값과 전체 글자 수 증가율(합성 시간 영향 참고), 시간 p50·p95를
    더한다. 두 쪽 모두 말투 일관성(consistency)을 적는다(판정은 decide가 출력 글에서 다시 센다).
    """
    items = _values(results, model).get("items", [])
    if not items:
        raise SystemExit(f"{model} {DIRECTION} has no outputs")
    formalize(WARM_UP)
    asis_items, formal_items, times_ms, increases = [], [], [], []
    for item in items:
        kept = {
            key: item[key] for key in ("id", "source", "reference", "hypothesis", "sentences") if key in item
        }
        started = clock()
        formal = formalize(item["hypothesis"])
        elapsed_ms = round((clock() - started) * 1000, 3)
        changed = formal != item["hypothesis"]
        times_ms.append(elapsed_ms)
        if changed:
            increases.append(len(formal) - len(item["hypothesis"]))
        asis_items.append(kept)
        formal_items.append({**kept, "hypothesis": formal, "changed": changed, "postprocess_ms": elapsed_ms})
    before_chars = sum(len(item["hypothesis"]) for item in asis_items)
    after_chars = sum(len(item["hypothesis"]) for item in formal_items)
    asis = {
        "count": len(asis_items),
        "items": asis_items,
        "consistency": consistency([item["hypothesis"] for item in asis_items]),
    }
    formal_values = {
        "count": len(formal_items),
        "items": formal_items,
        "consistency": consistency([item["hypothesis"] for item in formal_items]),
        "changed": len(increases),
        "char_increase_median": float(statistics.median(increases)) if increases else 0.0,
        "char_increase_rate": (after_chars - before_chars) / before_chars if before_chars else 0.0,
        "postprocess_p50_ms": statistics.median(times_ms),
        "postprocess_p95_ms": statistics.quantiles(times_ms, n=20)[18] if len(times_ms) >= 2 else times_ms[0],
    }
    return [
        {"model": model, "register": "asis", "directions": {DIRECTION: asis}},
        {"model": model + FORMAL, "register": "formal", "directions": {DIRECTION: formal_values}},
    ]


# 표 칸 안의 '|'는 칸 나눔으로 읽히지 않게 막는다.
def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def changes_lines(report: list[dict]) -> list[str]:
    """사용자 확인용 바뀐 출력 목록(마크다운): 바뀐 출력마다 원문(영어)·바뀌기 전·바뀐 뒤와 빈 판정 칸.

    21절의 문법 판정 기준표를 머리에 적는다. 이 목록은 커밋하지 않는 work/에 둔다(사람 확인은 사용자 몫).
    """
    asis, formal = (entry["directions"][DIRECTION]["items"] for entry in report)
    lines = [
        "# 말투 후처리로 바뀐 출력 (docs/experiments.md 21절 사람 확인용)",
        "",
        "판정 칸에 오류면 종류(활용 오류 / 높임 중복·어색함 / 뜻 바뀜)를, 아니면 '정상'을 적는다. "
        "셋 중 하나라도 있으면 오류다. 바뀌기 전 번역 자체의 오역은 세지 않는다(후처리가 만든 차이만 본다).",
        "",
        "| ID | 원문 | 바뀌기 전 | 바뀐 뒤 | 판정 |",
        "|---|---|---|---|---|",
    ]
    for before, after in zip(asis, formal, strict=True):
        if after.get("changed"):
            lines.append(
                f"| {after['id']} | {_cell(after['source'])} | {_cell(before['hypothesis'])} "
                f"| {_cell(after['hypothesis'])} |  |"
            )
    return lines


def judge(after: dict, comet_low: float, p50_ms: float) -> dict:
    """21절 자동 관문 셋을 본다.

    consistency(높임 출력 ≥ 98%이고 섞임 0), comet(COMET-22 차이 구간 하한 > −0.5), latency(p50 ≤ 5ms).
    p50은 ms 소수 셋째 자리(µs)로 반올림해 견준다(20절과 같이, 표기 오차로 경계값이 떨어지지 않게).
    """
    return {
        "consistency": after["all_honorific"] * 100 >= MIN_HONORIFIC_PERCENT * after["outputs"]
        and after["mixed"] <= MAX_MIXED,
        "comet": comet_low > MIN_COMET_LOW,
        "latency": round(p50_ms, 3) <= MAX_P50_MS,
    }


# spBLEU 지표: sacrebleu tokenize='flores200'(13절 재분석과 같은 FLORES-200 관례). 처음 쓸 때 토크나이저
# 모델을 SACREBLEU 환경 변수의 폴더(기본 ~/.sacrebleu)에 한 번 받는다. 테스트는 내려받기를 피하려고 13a로 바꿔
# 넣는다.
def bleu():
    from sacrebleu.metrics import BLEU

    return BLEU(tokenize="flores200")


def _pair(results: list[dict]) -> tuple[dict, dict]:
    """run 보고서의 asis·formal 두 출력. 각각 정확히 하나여야 한다(같은 실행 하나만 판정)."""
    by_register: dict[str, list[dict]] = {}
    for result in results:
        by_register.setdefault(result.get("register"), []).append(result)
    if len(by_register.get("asis", [])) != 1 or len(by_register.get("formal", [])) != 1:
        raise SystemExit("the report must hold one asis and one formal output (eval.register_eval run)")
    return by_register["asis"][0], by_register["formal"][0]


def decide(results: list[dict], comet: dict[str, float], rounds: int = ROUNDS) -> dict:
    """run 보고서와 COMET-22 점수에 21절 판정을 적용한다.

    두 출력은 같은 문장(ID·원문·참조)을 같은 순서로 가져야 하고, 모든 번역문의 COMET 점수가 있어야 한다(아니면
    판정하지 않고 멈춤). 말투 일관성은 보고서에 적힌 수가 아니라 출력 글에서 다시 센다. COMET-22 차이(formal −
    asis, 문장 점수 ×100 평균)의 95% 짝지은 구간으로 비열등을 보고, chrF·spBLEU 차이(corpus 점수, 같은 표본)는
    적기만 한다(참조도 말투가 섞여 있어 판정에 쓰지 않음). 사람 문법 확인은 하지 않았으므로 human_check는
    None이고, 자동 관문을 모두 통과해도 verdict는 'automatic-pass-pending-human-check'(채택 보류)다.
    """
    from sacrebleu.metrics import CHRF

    asis_result, formal_result = _pair(results)
    asis, formal = asis_result["directions"][DIRECTION], formal_result["directions"][DIRECTION]
    _same_sentences(formal, asis)
    references = [item["reference"] for item in asis["items"]]
    groups = [
        ([i["hypothesis"] for i in formal["items"]], [i["hypothesis"] for i in asis["items"]], references)
    ]
    comet_formal, comet_asis = _comet(comet, formal["items"]), _comet(comet, asis["items"])
    comet_diff = mean_difference([(comet_formal, comet_asis)], rounds)
    chrf_metric, bleu_metric = CHRF(), bleu()
    chrf = corpus_difference(chrf_metric, groups, rounds)
    spbleu = corpus_difference(bleu_metric, groups, rounds)
    before = consistency([item["hypothesis"] for item in asis["items"]])
    after = consistency([item["hypothesis"] for item in formal["items"]])
    rules = judge(after, comet_diff.low, formal["postprocess_p50_ms"])
    passes = all(rules.values())
    return {
        "model": asis_result["model"],
        "before": before,
        "after": after,
        "comet": {**_interval(comet_diff), "a": float(comet_formal.mean()), "b": float(comet_asis.mean())},
        "chrf": {
            **_interval(chrf),
            "a": chrf_metric.corpus_score(groups[0][0], [references]).score,
            "b": chrf_metric.corpus_score(groups[0][1], [references]).score,
        },
        "spbleu": {
            **_interval(spbleu),
            "a": bleu_metric.corpus_score(groups[0][0], [references]).score,
            "b": bleu_metric.corpus_score(groups[0][1], [references]).score,
        },
        "changed": formal.get("changed"),
        "char_increase_median": formal.get("char_increase_median"),
        "char_increase_rate": formal.get("char_increase_rate"),
        "postprocess_p50_ms": formal["postprocess_p50_ms"],
        "postprocess_p95_ms": formal.get("postprocess_p95_ms"),
        "rules": rules,
        "automatic_passes": passes,
        "human_check": None,
        "verdict": "automatic-pass-pending-human-check" if passes else "rejected",
    }


# 판정 결과의 마크다운: 지표 (a)~(d)와 자동 관문 O/X, 사람 확인 상태, 결론 한 줄.
def decide_lines(result: dict) -> list[str]:
    before, after, rules = result["before"], result["after"], result["rules"]

    def mark(rule: str) -> str:
        return "O" if rules[rule] else "X"

    conclusion = (
        "자동 관문 통과 — 사람 문법 확인(21절 기준표) 전이라 채택하지 않음"
        "(보류, KO_REGISTER 기본 asis 그대로)"
        if result["automatic_passes"]
        else "자동 관문 불통과 — 채택하지 않음(KO_REGISTER 기본 asis 그대로)"
    )
    return [
        f"- 대상: {result['model']} (영→한 {after['outputs']}개)",
        f"- (a) 모든 문장 끝이 높임인 출력: 후처리 전 {before['all_honorific']}/{before['outputs']} "
        f"({before['ratio']:.1%}), 뒤 {after['all_honorific']}/{after['outputs']} ({after['ratio']:.1%}); "
        f"섞인 출력: 전 {before['mixed']}, 뒤 {after['mixed']} → {mark('consistency')}",
        "  - 문장 끝 종류(전 → 뒤): "
        + ", ".join(f"{kind} {before['ends'][kind]} → {after['ends'][kind]}" for kind in _KINDS),
        "- (b) 바뀐 출력의 문법 오류율: 사람 문법 확인은 하지 않음(사용자가 정함) — 판정에 넣지 "
        "않았고, 하기 전에는 채택하지 않음",
        f"- (c) COMET-22 차이(후처리 − 원래) {_fmt(result['comet'])} (후처리 {result['comet']['a']:.2f}, "
        f"원래 {result['comet']['b']:.2f}) → {mark('comet')}",
        f"- (d) 후처리 시간 p50 {result['postprocess_p50_ms']:.3f}ms, "
        f"p95 {result['postprocess_p95_ms']:.3f}ms "
        f"→ {mark('latency')}",
        f"  - 적기만: chrF {result['chrf']['a']:.1f} (원래 {result['chrf']['b']:.1f}) "
        f"차이 {_fmt(result['chrf'])}, spBLEU {result['spbleu']['a']:.1f} "
        f"(원래 {result['spbleu']['b']:.1f}) 차이 {_fmt(result['spbleu'])}, "
        f"바뀐 출력 {result['changed']}개, "
        f"바뀐 출력의 글자 수 증가 중앙값 {result['char_increase_median']:.1f}, "
        f"전체 글자 수 증가율 {result['char_increase_rate']:.2%}",
        f"- 결론: {conclusion}",
    ]


# run 보고서의 사람이 읽을 요약(마크다운): 대상, 바뀐 출력 수, 두 출력의 일관성, 시간, 글자 수.
def run_lines(report: list[dict], source: Path) -> list[str]:
    asis, formal = (entry["directions"][DIRECTION] for entry in report)
    before, after = asis["consistency"], formal["consistency"]
    return [
        f"- 원래 보고서: {source} — {report[0]['model']}",
        f"- 바뀐 출력: {formal['changed']}/{formal['count']}",
        f"- 모든 문장 끝이 높임인 출력: 전 {before['all_honorific']}, 뒤 {after['all_honorific']}; "
        f"섞인 출력: 전 {before['mixed']}, 뒤 {after['mixed']}",
        f"- 후처리 시간 p50 {formal['postprocess_p50_ms']:.3f}ms, p95 {formal['postprocess_p95_ms']:.3f}ms",
        f"- 바뀐 출력의 글자 수 증가 중앙값 {formal['char_increase_median']:.1f}, "
        f"전체 글자 수 증가율 {formal['char_increase_rate']:.2%}",
        "- 판정은 COMET-22 채점 뒤 `eval.register_eval decide`(이 요약은 판정이 아님)",
    ]


# 분석기 판(재현용). 설치되지 않았으면 None.
def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


# 결과를 만든 시각과 함께 JSON으로, 사람이 읽을 요약을 마크다운으로 남긴다. prefix는 파일 이름의 앞부분.
def _write(prefix: str, title: str, tag: str, result, lines: list[str], stamp: datetime) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = REPORTS / f"{prefix}_{tag}_{stamp:%Y%m%d_%H%M%S}"
    base.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text("\n".join([f"# {title} ({tag})", "", *lines]) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")
    return base


def _run_command(args: argparse.Namespace) -> None:
    try:
        formalize = load_formalizer()
    except ModelError as exc:
        raise SystemExit(
            f"{exc}: run with `uv run --no-sync --with python-mecab-ko==1.3.7 "
            "--with python-mecab-ko-dic==2.1.1.post2 python -m eval.register_eval run ...`"
        ) from exc
    stamp = datetime.now(UTC)
    results = json.loads(args.report.read_text(encoding="utf-8"))
    report = run(results, args.model, formalize)
    meta = {
        "source_report": str(args.report),
        "created": stamp.isoformat(timespec="seconds"),
        "analyzer": {name: _version(name) for name in ("python-mecab-ko", "python-mecab-ko-dic")},
    }
    report = [{**entry, **meta} for entry in report]
    base = _write("register", "말투 후처리 적용", args.tag, report, run_lines(report, args.report), stamp)
    changes = args.changes or PROJECT / "work" / "register" / f"changed_{args.tag}_{stamp:%Y%m%d_%H%M%S}.md"
    changes.parent.mkdir(parents=True, exist_ok=True)
    changes.write_text("\n".join(changes_lines(report)) + "\n", encoding="utf-8")
    comet_path = args.comet_input or PROJECT / "work" / "register" / f"comet_input_{args.tag}.jsonl"
    rows = comet_input([base.with_suffix(".json")], None)
    comet_path.parent.mkdir(parents=True, exist_ok=True)
    comet_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"changed outputs -> {changes}")
    print(f"{len(rows)} segments -> {comet_path}")


def _decide_command(args: argparse.Namespace) -> None:
    results = json.loads(args.report.read_text(encoding="utf-8"))
    comet = json.loads(args.comet.read_text(encoding="utf-8"))
    _check_comet_meta(comet["meta"])
    stamp = datetime.now(UTC)
    result = decide(results, comet["scores"], ROUNDS)
    result = {
        "created": stamp.isoformat(timespec="seconds"),
        "rounds": ROUNDS,
        "seed": SEED,
        **result,
        "report": str(args.report),
        "comet_meta": comet["meta"],
    }
    _write("register_decide", "말투 후처리 판정", args.tag, result, decide_lines(result), stamp)


# 하위 명령 run / decide. 붓스트랩 반복 수는 13절의 10,000번(ROUNDS)이다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run",
        help="저장된 mt_eval 보고서의 영→한 번역문에 말투 후처리를 걸어 보고서·바뀐 출력 목록·"
        "COMET 입력을 쓴다",
    )
    run_parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="eval.mt_eval 보고서 .json(21절: 2-1절 보고서 또는 20절이 고른 팔의 보고서)",
    )
    run_parser.add_argument(
        "--model", default=MODEL, help=f"보고서 안의 영→한 결과 이름(기본 {MODEL}, 지금 서비스 설정)"
    )
    run_parser.add_argument(
        "--tag", required=True, help="결과 파일 이름의 태그: eval/reports/register_<태그>_<시각>.json·.md"
    )
    run_parser.add_argument(
        "--changes",
        type=Path,
        help="사용자 확인용 바뀐 출력 목록 .md"
        "(기본 ../work/register/changed_<태그>_<시각>.md, 커밋하지 않음)",
    )
    run_parser.add_argument(
        "--comet-input",
        type=Path,
        help="COMET-22 채점 입력 .jsonl"
        "(기본 ../work/register/comet_input_<태그>.jsonl, 두 출력의 겹치지 않는 문장)",
    )
    decide_parser = commands.add_parser(
        "decide", help="run 보고서와 COMET-22 점수에 21절 판정 규칙(자동 관문)을 적용한다"
    )
    decide_parser.add_argument(
        "--report", type=Path, required=True, help="eval.register_eval run이 쓴 보고서 .json"
    )
    decide_parser.add_argument(
        "--comet",
        type=Path,
        required=True,
        help="그 보고서의 문장을 eval/comet_score.py로 채점한 COMET-22 점수 .json(13절의 고정 모델이어야 함)",
    )
    decide_parser.add_argument(
        "--tag",
        required=True,
        help="결과 파일 이름의 태그: eval/reports/register_decide_<태그>_<시각>.json·.md",
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        _run_command(args)
    else:
        _decide_command(args)


if __name__ == "__main__":
    main()
