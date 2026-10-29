"""같은 opus-mt의 빔 탐색 n-best에서 문장 chrF로 고르는 MBR 디코딩과 그 판정(docs/experiments.md 20절).

MBR(minimum Bayes risk) 디코딩: 빔 탐색이 준 번역 후보 n개마다, 나머지 후보들을 의사 참조로 두고 sacrebleu
문장 chrF(2절과 같은 기본 설정)의 평균(기대 효용)을 구해 그 값이 가장 큰 후보를 번역으로 고른다. 이 파일은
번역 모델을 돌리지 않는다: 후보는 app.services.translation의 MarianTranslator.candidates가 만들고, 여기서는
고르기(측정 때 eval.mt_eval의 '/b8-mbr' 팔 안에서)와 저장된 보고서의 판정만 한다.

backend/에서 gpu·eval 그룹 환경으로(측정 순서와 조건은 20절):
    uv run --no-sync python -m eval.mt_eval --split validation --tag t108_dev --models opus-mt-tc-big-ko-en \\
        opus-mt-tc-big-ko-en/b8 opus-mt-tc-big-ko-en/b8-mbr opus-mt-tc-big-en-ko/split \\
        opus-mt-tc-big-en-ko/split/b8 opus-mt-tc-big-en-ko/split/b8-mbr
    uv run --no-sync python -m eval.mt_decide comet-input --reports eval/reports/mt_t108_dev_<시각>.json \\
        --output ../work/mbr/comet_input_dev.jsonl
    ../work/comet-venv/Scripts/python.exe eval/comet_score.py score \\
        --input ../work/mbr/comet_input_dev.jsonl \\
        --output eval/reports/mt_t108_dev_<시각>_comet22.json --device cuda --precision fp16
    uv run --no-sync python -m eval.mbr decide --report eval/reports/mt_t108_dev_<시각>.json \\
        --comet eval/reports/mt_t108_dev_<시각>_comet22.json --tag t108_dev
"""

# [파일 역할] MBR 디코딩 시험(docs/experiments.md 20절)의 두 부분: (1) 측정 때 쓰는 MBR 고르기(select와 효용
# 표 chrf_matrix)와 그것을 번역기 모양으로 감싼 MbrTranslator(eval/mt_eval.py의 '/b8-mbr' 팔), (2) 저장된 측정
# 결과에 측정 전에 적은 판정 규칙을 적용하는 decide(명령 `decide`).
# [설계 이유] MBR 계산을 번역기의 translate() 안에서 해, mt_eval이 재는 문장당 번역 시간에 MBR 계산 시간이
# 그대로 들어가게 했다(채택하면 서버도 같은 계산을 더하므로 시간 조건이 그 비용까지 봐야 한다). 효용은
# sacrebleu 공개 함수 sentence_score와 같은 값을 후보마다 n-gram을 한 번만 뽑아 계산한다(20절 '측정 전 확인'의
# CPU 시간 참고). 판정은 15절 도구(eval/mt_decide.py)와 같은 COMET 키·점수 확인과, 13절의 짝지은
# 붓스트랩(eval/significance.py, 10,000번·시드 12345)을 쓴다. 측정 전에 적은 규칙을 코드로 고정해 같은
# 입력이면 늘 같은 판정이 나오게 한다.
# [연결] 후보는 app/services/translation.py의 MarianTranslator.candidates(_Ct2Model.generate_nbest)가 만들고,
# eval/mt_eval.py의 CANDIDATES가 MbrTranslator로 팔을 만든다. 판정 입력은 mt_eval 보고서와
# eval/comet_score.py의 COMET-22 점수, 결과는 eval/reports/mbr_decide_<태그>_<시각>.json·.md. 테스트:
# tests/test_mbr.py.

from __future__ import annotations

import argparse
import functools
import json
import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

# eval.comet_score: 13절에 고정한 COMET-22 모델·리비전·체크포인트 해시(점수 파일이 그 모델로 매겨졌는지 확인).
# 맨 위에서 표준 라이브러리만 import하는 파일이라 이 환경에서 읽어도 된다. eval.mt_decide: 15절 판정 도구와
# 같은 COMET 점수 찾기(키가 없거나 유한하지 않으면 멈춤)·구간 dict·구간 글 모양을 그대로 쓴다.
# eval.significance: 13절의 붓스트랩 반복 수·시드와 짝지은 구간 함수. sacrebleu는 eval 그룹에만 있어 쓰는 함수
# 안에서 import한다.
from app.services.interfaces import Language
from eval import comet_score
from eval.common import REPORTS
from eval.mt_decide import _comet, _fmt, _interval
from eval.significance import ROUNDS, SEED, corpus_difference, mean_difference

# 효용 표 함수의 모양: 후보 글 목록을 받아 표[i][j] = 효용(가설 = 글 i, 의사 참조 = 글 j)을 돌려준다.
# 대각선(자기 자신)은 쓰지 않는다. 테스트는 정해 둔 표를 돌려주는 대역을 넣는다.
UtilityMatrix = Callable[[Sequence[str]], list[list[float]]]


# sacrebleu chrF 지표 하나(2절과 같은 기본 설정: 문자 6-gram, 단어 n-gram 없음, beta 2, eps 평활 없음)를
# 만들어 두고 다시 쓴다. 측정 도구의 첫 번역은 준비 호출이라 이 import와 생성 시간은 잰 시간에 들지 않는다.
@functools.cache
def _chrf():
    from sacrebleu.metrics import CHRF

    metric = CHRF()
    # chrf_matrix의 빠른 길은 문자 n-gram만 센다. 단어 n-gram을 더하는 chrF++ 설정이면 공개 함수와 같은 값이
    # 나오지 않으므로 멈춘다(기본 설정을 쓰는 한 일어나지 않는다).
    if metric.word_order != 0:
        raise RuntimeError("the fast chrF path counts character n-grams only")
    return metric


def chrf_matrix(texts: Sequence[str]) -> list[list[float]]:
    """후보끼리의 문장 chrF 표: 표[i][j] = texts[i]를 가설, texts[j]를 참조로 둔 sacrebleu 문장 chrF.

    sacrebleu `CHRF().sentence_score(texts[i], [texts[j]]).score`와 소수점 끝자리까지 같은 값이다
    (tests/test_mbr.py가 확인). 공개 함수는 짝마다 두 글의 문자 n-gram을 다시 뽑지만, 여기서는 후보마다 한
    번만 뽑아(공개 함수가 안에서 부르는 sacrebleu 함수 `extract_all_char_ngrams`·`_get_match_statistics`·
    `_compute_f_score` 그대로) 짝마다 일치 수만 센다. sacrebleu의 내부 함수에 기대므로 sacrebleu 판이 바뀌면
    그 테스트로 다시 확인한다(13절이 쓰는 eval/significance.py와 같은 처지, 지금 판은 2.6.0). chrF는 대칭이
    아니다(beta 2라 재현율 쪽 무게가 크다). 대각선은 0.0으로 두고 쓰지 않는다.
    """
    from sacrebleu.metrics.helpers import extract_all_char_ngrams

    metric = _chrf()
    grams = [
        extract_all_char_ngrams(metric._preprocess_segment(text), metric.char_order, metric.whitespace)
        for text in texts
    ]
    table: list[list[float]] = []
    for i, hypothesis in enumerate(grams):
        row: list[float] = []
        for j, reference in enumerate(grams):
            if i == j:
                row.append(0.0)
                continue
            stats: list[int] = []
            for hypothesis_order, reference_order in zip(hypothesis, reference, strict=True):
                stats.extend(metric._get_match_statistics(hypothesis_order, reference_order))
            row.append(metric._compute_f_score(stats))
        table.append(row)
    return table


def select(candidates: Sequence[tuple[str, float]], matrix: UtilityMatrix = chrf_matrix) -> int:
    """MBR로 고른 후보의 위치. 입력: (번역문, 엔진 점수) 목록, 효용 표 함수.

    후보 i의 기대 효용 = 자기 자신을 뺀 다른 후보 j 모두를 의사 참조로 둔 효용(표[i][j])의 평균. 이 값이 가장
    큰 후보를 고르고, 같으면 엔진 점수가 높은 후보, 그것도 같으면 앞 후보(엔진이 준 순서)다. 평균은
    math.fsum으로 더해, 같은 값들이면 더하는 순서와 상관없이 같은 평균이 되게 한다(그래야 같은 효용을 동점으로
    본다). 겹치는 글은 따로 센다: 엔진의 가설 하나가 의사 참조 하나다(같은 글을 낸 가설이 많을수록 그 글의
    효용이 커진다). 후보가 하나면 표를 만들지 않고 0, 없으면 ValueError(후보 함수는 빈 목록을 내지 않는다).
    """
    if not candidates:
        raise ValueError("MBR needs at least one candidate")
    if len(candidates) == 1:
        return 0
    texts = [text for text, _ in candidates]
    table = matrix(texts)
    others = len(texts) - 1
    utilities = [
        math.fsum(table[i][j] for j in range(len(texts)) if j != i) / others for i in range(len(texts))
    ]
    return max(range(len(texts)), key=lambda i: (utilities[i], candidates[i][1], -i))


class NbestTranslator(Protocol):
    """MbrTranslator가 감싸는 번역기의 모양. app.services.translation.MarianTranslator가 이 모양이다."""

    model_name: str
    by_sentence: bool

    def candidates(
        self, text: str, source: Language, target: Language, beam_size: int, n: int
    ) -> list[list[tuple[str, float]]]: ...


class MbrTranslator:
    """빔 탐색의 n-best에서 문장마다 MBR로 고르는 번역기(측정 전용, 20절의 '/b8-mbr' 팔).

    translate()는 감싼 번역기의 candidates로 문장마다 후보 n개를 받고(by_sentence면 서비스처럼 나눈 문장마다),
    문장마다 select로 고른 뒤 공백 하나로 잇는다(서비스의 문장 단위 번역과 같은 이음). MBR 계산이 translate()
    안에 있어 eval/mt_eval.py가 재는 문장당 번역 시간에 들어간다. picks는 마지막 번역에서 문장마다 후보 수와
    고른 위치(0 = 엔진 점수가 가장 높은 후보)를 남긴다(측정 보고서용, 번역할 때마다 새로 씀). 후보 함수의
    ModelError는 그대로 올라간다. model_name은 감싼 모델 이름 뒤에 '/b<빔>-mbr<n>'을 붙이고, decoding은
    측정 보고서에 남길 디코딩 설정이다(MarianTranslator.decoding과 같은 모양).
    """

    def __init__(
        self, inner: NbestTranslator, beam: int = 8, n: int = 8, matrix: UtilityMatrix = chrf_matrix
    ) -> None:
        if not 1 <= n <= beam:
            raise ValueError(f"need between 1 and beam={beam} hypotheses, got {n}")
        self.inner, self.beam, self.n = inner, beam, n
        self._matrix = matrix
        self.model_name = f"{inner.model_name}/b{beam}-mbr{n}"
        self.decoding = {
            "beam_size": beam,
            "num_hypotheses": n,
            "selection": "mbr-chrf",
            "by_sentence": inner.by_sentence,
        }
        self.picks: list[dict] = []

    def translate(self, text: str, source: Language, target: Language) -> str:
        self.picks = []
        chosen: list[str] = []
        for candidates in self.inner.candidates(text, source, target, self.beam, self.n):
            index = select(candidates, self._matrix)
            self.picks.append({"candidates": len(candidates), "chosen": index})
            chosen.append(candidates[index][0])
        return " ".join(chosen)


# ---- 판정(20절 '판정'). 입력은 eval.mt_eval 보고서 하나(여섯 팔을 같은 세션에 잰 것)와 COMET-22 점수.

# 20절에 등록한 팔: 방향마다 (기준 = 지금 서비스 설정, B8 = 빔 8의 1-best, B8-MBR = 빔 8의 8-best + MBR).
# 이름은 eval/mt_eval.py의 CANDIDATES 키다. 영→한은 서비스처럼 셋 다 문장 단위다.
ARMS: dict[str, tuple[str, str, str]] = {
    "ko-en": ("opus-mt-tc-big-ko-en", "opus-mt-tc-big-ko-en/b8", "opus-mt-tc-big-ko-en/b8-mbr"),
    "en-ko": (
        "opus-mt-tc-big-en-ko/split",
        "opus-mt-tc-big-en-ko/split/b8",
        "opus-mt-tc-big-en-ko/split/b8-mbr",
    ),
}
# 팔마다 보고서의 decoding 기록(번역기의 decoding)이 이것과 같아야 판정한다. 팔 이름과 실제 설정이 어긋난
# 측정(예: 영→한 B8을 문장 단위 없이 잼)은 빔 크기 효과와 다른 효과가 섞이므로 판정하지 않는다.
DECODING: dict[str, dict] = {
    "opus-mt-tc-big-ko-en": {
        "beam_size": 4,
        "num_hypotheses": 1,
        "selection": "1-best",
        "by_sentence": False,
    },
    "opus-mt-tc-big-ko-en/b8": {
        "beam_size": 8,
        "num_hypotheses": 1,
        "selection": "1-best",
        "by_sentence": False,
    },
    "opus-mt-tc-big-ko-en/b8-mbr": {
        "beam_size": 8,
        "num_hypotheses": 8,
        "selection": "mbr-chrf",
        "by_sentence": False,
    },
    "opus-mt-tc-big-en-ko/split": {
        "beam_size": 4,
        "num_hypotheses": 1,
        "selection": "1-best",
        "by_sentence": True,
    },
    "opus-mt-tc-big-en-ko/split/b8": {
        "beam_size": 8,
        "num_hypotheses": 1,
        "selection": "1-best",
        "by_sentence": True,
    },
    "opus-mt-tc-big-en-ko/split/b8-mbr": {
        "beam_size": 8,
        "num_hypotheses": 8,
        "selection": "mbr-chrf",
        "by_sentence": True,
    },
}
# 시간 조건: 문장당 번역 시간 p50이 기준보다 +50ms 넘게 늘지 않고, p95가 0.8초 이하(20절. 동시통역 영어 원문의
# 말 끝 → 번역 음성 1.5초 목표와의 여유가 작아 15절의 절대값 0.35초 대신 증가 폭으로 둠).
MAX_P50_INCREASE_MS = 50.0
MAX_P95_MS = 800.0


# 보고서에서 팔 하나의 한 방향 결과를 꺼낸다. 팔이 없거나 그 방향이 없거나, 디코딩 기록이 등록한 것과 다르면
# 판정하지 않고 멈춘다.
def _values(by_name: dict[str, dict], name: str, direction: str) -> dict:
    result = by_name.get(name)
    if result is None or direction not in result["directions"]:
        raise SystemExit(f"the report has no {name} {direction} run")
    if result.get("decoding") != DECODING[name]:
        raise SystemExit(f"{name} was decoded with {result.get('decoding')}, not {DECODING[name]}")
    return result["directions"][direction]


# 짝지은 비교의 전제: 두 실행이 같은 문장(ID·원문·참조)을 같은 순서로 가져야 한다.
def _same_sentences(a: dict, b: dict) -> None:
    def key(values: dict) -> list[tuple]:
        return [(item["id"], item["source"], item["reference"]) for item in values["items"]]

    if key(a) != key(b):
        raise SystemExit("the two runs must hold the same sentences in the same order")


def difference(a: dict, b: dict, comet: dict[str, float], rounds: int = ROUNDS) -> dict:
    """한 방향 두 실행의 a − b 차이: chrF와 COMET-22, 각각 95% 짝지은 붓스트랩 구간(13절과 같은 함수).

    chrF는 문장 통계를 다시 더하는 corpus 점수의 차이(eval/significance.py corpus_difference, sacrebleu 기본
    설정), COMET-22는 문장 점수(×100) 평균의 차이(mean_difference)다. 두 구간은 같은 문장 표본(시드 12345)을
    쓴다. both_above_zero는 두 구간 하한이 모두 0보다 큰지(20절 품질 조건)다. 문장이 어긋나거나 COMET 점수가
    빠졌으면 멈춘다.
    """
    from sacrebleu.metrics import CHRF

    _same_sentences(a, b)
    references = [item["reference"] for item in b["items"]]
    chrf = corpus_difference(
        CHRF(),
        [([i["hypothesis"] for i in a["items"]], [i["hypothesis"] for i in b["items"]], references)],
        rounds,
    )
    comet_a, comet_b = _comet(comet, a["items"]), _comet(comet, b["items"])
    comet_diff = mean_difference([(comet_a, comet_b)], rounds)
    return {
        "chrf": {**_interval(chrf), "a": a["chrf"], "b": b["chrf"]},
        "comet": {**_interval(comet_diff), "a": float(comet_a.mean()), "b": float(comet_b.mean())},
        "both_above_zero": chrf.low > 0 and comet_diff.low > 0,
    }


def judge(arm: dict, base: dict, comet: dict[str, float], rounds: int = ROUNDS) -> dict:
    """한 방향에서 팔 하나를 기준(지금 서비스 설정)과 견준다: 품질·출력 점검·시간(20절 판정).

    품질: chrF 차이와 COMET-22 차이(팔 − 기준)의 95% 짝지은 구간 하한이 둘 다 0보다 큼. 출력 점검
    (eval/mt_checks.py): 다른 문자·설명형 출력 수와 빈 출력 수가 각각 기준보다 늘지 않음. 시간: 문장당 번역
    시간 p50이 기준보다 50ms 넘게 늘지 않고 p95가 0.8초 이하(MBR 계산 포함, 같은 세션). 시간은 ms 소수 셋째
    자리(µs)로 반올림해 견준다: 초 단위 부동소수점의 표기 오차(0.55 − 0.5 = 0.050000000000000044)로 경계값이
    떨어지지 않게 하려는 것이고, 1µs보다 작은 차이는 판정에 뜻이 없다. 세 조건을 모두 지키면 passes.
    """
    diff = difference(arm, base, comet, rounds)
    checks = {key: arm["checks"][key] for key in ("flagged", "empty")}
    baseline_checks = {key: base["checks"][key] for key in ("flagged", "empty")}
    increase_ms = round((arm["latency_p50"] - base["latency_p50"]) * 1000, 3)
    p95_ms = round(arm["latency_p95"] * 1000, 3)
    rules = {
        "quality": diff["both_above_zero"],
        "checks": all(checks[key] <= baseline_checks[key] for key in checks),
        "latency": increase_ms <= MAX_P50_INCREASE_MS and p95_ms <= MAX_P95_MS,
    }
    return {
        **diff,
        "checks": checks,
        "baseline_checks": baseline_checks,
        "latency_p50": arm["latency_p50"],
        "baseline_latency_p50": base["latency_p50"],
        "p50_increase_ms": increase_ms,
        "latency_p95": arm["latency_p95"],
        "rules": rules,
        "passes": all(rules.values()),
    }


def choose(b8_passes: bool, mbr_passes: bool, mbr_beats_b8: bool) -> str | None:
    """20절의 고르기: 고른 팔의 종류("b8", "b8-mbr") 또는 None(기준 유지).

    B8과 B8-MBR이 모두 기준 대비 조건을 통과하면, B8-MBR − B8도 두 지표 조건(구간 하한 둘 다 > 0)을 통과할
    때만 B8-MBR이고 아니면 더 단순한 B8. 하나만 통과하면 그 팔, 아무것도 통과하지 못하면 None.
    """
    if b8_passes and mbr_passes:
        return "b8-mbr" if mbr_beats_b8 else "b8"
    if mbr_passes:
        return "b8-mbr"
    if b8_passes:
        return "b8"
    return None


# 참고(판정에 쓰지 않음): MBR 팔에서 MBR이 n-best의 첫 후보(엔진 점수가 가장 높은 후보)가 아닌 것을 고른 문장
# 수와, 출력이 B8과 글자까지 같은 입력 수. MBR이 실제로 얼마나 바꿨는지 읽는 데 쓴다.
def _mbr_choices(mbr: dict, b8: dict) -> dict:
    picks = [pick for item in mbr["items"] for pick in item.get("mbr", [])]
    return {
        "inputs": len(mbr["items"]),
        "sentences": len(picks),
        "not_first": sum(pick["chosen"] != 0 for pick in picks),
        "same_as_b8": sum(
            a["hypothesis"] == b["hypothesis"] for a, b in zip(mbr["items"], b8["items"], strict=True)
        ),
    }


def decide(results: list[dict], comet: dict[str, float], rounds: int = ROUNDS) -> dict:
    """mt_eval 보고서 하나의 결과에 20절 판정을 방향마다 적용한다.

    방향마다 등록한 세 팔(ARMS)이 모두 등록한 디코딩(DECODING)으로 있어야 하고, 같은 팔이 두 번 있으면 안
    된다(같은 세션의 측정 하나만 판정한다). 보고서의 다른 후보는 보지 않는다. 방향마다 B8·B8-MBR을 기준과
    견주고(judge), B8-MBR − B8 차이를 구해(difference, 둘 다 통과할 때만 고르기에 씀) choose로 고른다. pick은
    고른 팔 이름 또는 None(기준 유지)이다.
    """
    names = [result["model"] for result in results]
    if len(set(names)) != len(names):
        raise SystemExit("each arm must appear once in the report")
    by_name = {result["model"]: result for result in results}
    out: dict = {"directions": {}}
    for direction, (base_name, b8_name, mbr_name) in ARMS.items():
        base, b8, mbr = (_values(by_name, name, direction) for name in (base_name, b8_name, mbr_name))
        arms = {b8_name: judge(b8, base, comet, rounds), mbr_name: judge(mbr, base, comet, rounds)}
        mbr_vs_b8 = difference(mbr, b8, comet, rounds)
        b8_passes, mbr_passes = arms[b8_name]["passes"], arms[mbr_name]["passes"]
        kind = choose(b8_passes, mbr_passes, mbr_vs_b8["both_above_zero"])
        out["directions"][direction] = {
            "baseline": base_name,
            "arms": arms,
            "mbr_vs_b8": {**mbr_vs_b8, "used": b8_passes and mbr_passes},
            "pick": {"b8": b8_name, "b8-mbr": mbr_name}.get(kind) if kind else None,
            "mbr_choices": _mbr_choices(mbr, b8),
        }
    return out


# 판정 결과의 마크다운: 팔마다 chrF·두 차이 구간·출력 점검·시간과 세 조건(O/X), 끝에 방향마다 고른 팔, B8-MBR
# − B8 차이(고르기에 썼는지), 참고 수치.
def decide_lines(result: dict) -> list[str]:
    lines = [
        "| 방향 | 팔 | chrF (기준) | chrF 차이 [95%] | COMET-22 차이 [95%] | 다른 문자·설명형 / 빈 (기준) | "
        "p50 s (기준) | p50 증가 ms | p95 s | 품질 | 출력 | 시간 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for direction, values in result["directions"].items():
        for name, arm in values["arms"].items():
            rules = arm["rules"]
            marks = " | ".join("O" if rules[rule] else "X" for rule in ("quality", "checks", "latency"))
            lines.append(
                f"| {direction} | {name} | {arm['chrf']['a']:.1f} ({arm['chrf']['b']:.1f}) "
                f"| {_fmt(arm['chrf'])} | {_fmt(arm['comet'])} "
                f"| {arm['checks']['flagged']} / {arm['checks']['empty']} "
                f"({arm['baseline_checks']['flagged']} / {arm['baseline_checks']['empty']}) "
                f"| {arm['latency_p50']:.3f} ({arm['baseline_latency_p50']:.3f}) "
                f"| {arm['p50_increase_ms']:+.1f} | {arm['latency_p95']:.3f} | {marks} |"
            )
    lines.append("")
    for direction, values in result["directions"].items():
        versus, choices = values["mbr_vs_b8"], values["mbr_choices"]
        lines += [
            f"- {direction}: 고른 팔 {values['pick'] or '없음(기준 유지)'} (기준 {values['baseline']})",
            f"  - B8-MBR − B8: chrF {_fmt(versus['chrf'])}, COMET-22 {_fmt(versus['comet'])}, "
            f"두 하한 > 0: {'예' if versus['both_above_zero'] else '아니오'} "
            f"({'고르기에 씀' if versus['used'] else '고르기에 안 씀: 두 팔이 모두 통과할 때만 씀'})",
            f"  - 참고(판정 밖): MBR이 n-best 첫 후보가 아닌 것을 고른 문장 {choices['not_first']}/"
            f"{choices['sentences']}, B8과 출력이 같은 입력 {choices['same_as_b8']}/{choices['inputs']}",
        ]
    return lines


# COMET 점수 파일이 13절에 고정한 COMET-22(모델·리비전·채점 때 해시한 체크포인트·인코더 리비전)로 매겨졌는지
# 본다. 다르면 판정하지 않는다. 장치·정밀도(규칙은 GPU fp16)는 결과에 그대로 남긴다.
def _check_comet_meta(meta: dict) -> None:
    expected = {
        "model": comet_score.MODEL,
        "revision": comet_score.REVISION,
        "checkpoint_sha256": comet_score.CHECKPOINT_SHA256,
        "encoder": comet_score.ENCODER,
        "encoder_revision": comet_score.ENCODER_REVISION,
    }
    wrong = {key: meta.get(key) for key, value in expected.items() if meta.get(key) != value}
    if wrong:
        raise SystemExit(
            f"the COMET scores are not from the fixed COMET-22 model (docs/experiments.md 13): {wrong}"
        )


# 판정 결과를 만든 시각과 붓스트랩 설정(반복 수·시드)과 함께 JSON으로, 사람이 읽을 요약을 마크다운으로 남긴다.
def _write(tag: str, result: dict, lines: list[str]) -> Path:
    stamp = datetime.now(UTC)
    result = {"created": stamp.isoformat(timespec="seconds"), "rounds": ROUNDS, "seed": SEED, **result}
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = REPORTS / f"mbr_decide_{tag}_{stamp:%Y%m%d_%H%M%S}"
    base.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text(
        "\n".join([f"# MBR 디코딩 판정 ({tag})", "", *lines]) + "\n", encoding="utf-8"
    )
    print(f"written to {base.with_suffix('.md')}")
    return base


# 하위 명령 decide: 보고서와 COMET 점수를 읽어 점수 파일의 모델을 확인하고, 판정해 결과를 쓴다. 붓스트랩 반복
# 수는 13절의 10,000번(ROUNDS)이다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    decide_parser = commands.add_parser(
        "decide", help="저장된 mt_eval 보고서와 COMET-22 점수에 20절 판정 규칙을 적용한다"
    )
    decide_parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="여섯 팔을 같은 세션에 한 번에 잰 eval.mt_eval 보고서 .json(같은 세션만 판정하므로 하나만 받음)",
    )
    decide_parser.add_argument(
        "--comet",
        type=Path,
        required=True,
        help="그 보고서의 문장을 eval/comet_score.py로 채점한 COMET-22 점수 .json(13절의 고정 모델이어야 함)",
    )
    decide_parser.add_argument(
        "--tag", required=True, help="결과 파일 이름의 태그: eval/reports/mbr_decide_<태그>_<시각>.json·.md"
    )
    args = parser.parse_args(argv)

    results = json.loads(args.report.read_text(encoding="utf-8"))
    comet = json.loads(args.comet.read_text(encoding="utf-8"))
    _check_comet_meta(comet["meta"])
    result = decide(results, comet["scores"], ROUNDS)
    result["report"] = str(args.report)
    result["comet_meta"] = comet["meta"]
    _write(args.tag, result, decide_lines(result))


if __name__ == "__main__":
    main()
