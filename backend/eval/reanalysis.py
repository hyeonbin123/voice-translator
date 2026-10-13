"""Re-analysis of recorded translation and speech recognition decisions with paired intervals.

docs/experiments.md 13 (task T80) registers the comparisons, the intervals and how they are read before
anything is computed. Nothing is re-run, re-selected or rewritten: every number comes from the per-sentence
items of the stored reports in eval/reports/, and the COMET-22 scores come from eval/comet_score.py, which
runs in its own environment.

From backend/ with the eval group. `run` scores spBLEU with sacrebleu's flores200 tokenizer, which sacrebleu
downloads once into the folder named by the SACREBLEU environment variable (default: ~/.sacrebleu).
    uv run --no-sync python -m eval.reanalysis check
    uv run --no-sync python -m eval.reanalysis comet-input --output ../work/vt-n3/comet_input.jsonl
    uv run --no-sync python -m eval.reanalysis comet-input --subset precision --output CHECK.jsonl
    uv run --no-sync python -m eval.reanalysis comet-agreement CHECK_FP16.json CHECK_FP32.json
    uv run --no-sync python -m eval.reanalysis run --comet ../work/vt-n3/comet_fp16.json --tag n3
"""

# [파일 역할] 이미 기록된 번역·음성 인식 결정(1·1-1·2·2-1·6·9절)마다 차이 a − b의 95% 짝지은 붓스트랩 구간과
# COMET-22·spBLEU 차이를 계산해 보고서로 남기는 재분석 도구다(T80, docs/experiments.md 13절).
# [왜 필요한가] 지난 결정들은 0.3~1.0점 chrF 차이를 점 추정만으로 판정했다. 이 도구는 그 차이가 표본 잡음
# 밖인지(견고/문턱 걸침)와, 모델 기반 지표(COMET-22)가 chrF와 같은 쪽을 가리키는지를 본다. 모델을 다시 돌리지
# 않고, 아무것도 다시 고르지 않으며, 기록된 값을 고치지 않는다. 결과는 README '얇은 판정 여유폭'에 요약돼
# 있다.
# [흐름] check: 저장된 문장별 결과로 기록된 corpus 점수 66개를 다시 계산해 같은지 보는 관문 → comet-input:
# COMET-22로 채점할 (원문, 번역문, 참조) 중복 없는 목록을 쓴다(채점은 별도 환경의 eval/comet_score.py) →
# comet-agreement: GPU fp16과 CPU fp32 점수가 충분히 같은지 확인 → run: 관문을 다시 통과시킨 뒤 비교표 전체의
# 구간을 계산해 reports/reanalysis_<tag>_*.json·.md를 쓴다.
# [설계 이유] 비교할 보고서 이름과 비교 목록(문턱·규칙 문장 포함)을 코드 상수로 고정해, 측정 전에 13절에 적은
# 비교표와 실제 계산이 한 줄씩 대응하게 했다.

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

# jiwer: 발화마다 편집 수와 참조 길이를 다시 센다. numpy: 배열 계산. sacrebleu: chrF·spBLEU 점수와 서명(설정
# 기록). 구간 계산은 eval.significance의 함수를 그대로 쓴다.
import jiwer
import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

from eval.common import REPORTS
from eval.significance import (
    ROUNDS,
    SEED,
    TRIALS,
    Interval,
    by_group,
    classify,
    corpus_difference,
    mean_difference,
    paired_test,
    rate_difference,
)

# 재분석에 쓰는 저장 보고서 파일 이름. 결정을 내릴 때 썼던 바로 그 보고서로 고정해 다른 실행의 결과가 섞이지
# 않게 한다. T2는 T15의 한국어 아포스트로피 보정을 다시 적용한 _rescored 보고서다.
# The stored reports the recorded decisions were made from (docs/experiments.md 1, 1-1, 2, 2-1, 6, 9).
T2_DEV = "stt_t2_dev_20260912_223312_rescored.json"
T2_TEST = "stt_t2_test_20260912_224129_rescored.json"
T14_DEV = "stt_t14_dev_20260913_054613.json"
T14_TEST = "stt_t14_test_20260913_055810.json"
T3_DEV = "mt_t3_dev_20260912_231928.json"
T3_TEST = "mt_t3_test_20260912_232115.json"
T17_DEV = "mt_t17_dev_20260913_054853.json"
T17_TEST = "mt_t17_test_20260913_055845.json"
T32_DEV = "typo_t32_dev_20260913_120900.json"
T32_DEV_D = "typo_t32_dev_d_20260913_121811.json"
T32_TEST = "typo_t32_test_20260913_122908.json"
T65_TEST = "quality_t65_test_20260914_125149.json"

# TC_BIG: 방향별 서버 번역 모델 이름. T32 보고서에 없는 깨끗한 원문·참조를 T3 보고서의 tc-big 항목에서 찾을 때
# 쓴다.
# LABELS: 분류 결과를 보고서 표에 쓸 한국어 이름. 문턱이 없는 비교(None)는 '참고'.
TC_BIG = {"ko-en": "opus-mt-tc-big-ko-en", "en-ko": "opus-mt-tc-big-en-ko"}
LABELS = {"robust": "견고", "straddles": "문턱 걸침", "opposite": "반대쪽", None: "참고"}
# COMET-22 fp16 관문의 한도: 문장 점수 차이 최대 0.01, 시스템 평균 차이 0.001(13절 'COMET-22 정밀도'). 넘으면
# 모든 문장을 CPU fp32로 채점하기로 했다(실제로는 최대 0.0063으로 통과).
# COMET fp16 against fp32 on the precision subset (docs/experiments.md 13): within these, fp16 scores stand.
COMET_MAX_SEGMENT_GAP = 0.01
COMET_MAX_MEAN_GAP = 0.001

# 보고서 이름 → 읽은 JSON을 돌려주는 함수의 타입. 기본은 디스크를 읽는 stored이고,
# 테스트(tests/test_reanalysis.py)는 가짜 보고서를 주는 함수를 넘겨 파일 없이 검증한다.
Reports = Callable[[str], object]


# reports/의 보고서를 읽어 파싱한다. @cache로 같은 파일은 한 번만 읽는다(비교마다 같은 보고서를 여러 번
# 부르므로).
@cache
def stored(name: str) -> object:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


# 번역 비교의 한쪽 데이터: 문장 ID, 깨끗한 원문, 참조, 이 시스템의 번역문. 원문을 따로 두는 것은 COMET-22가
# (원문, 번역문, 참조)를 입력으로 받기 때문이다.
@dataclass(frozen=True)
class Segments:
    """One system's translations of a sentence list, with the clean source and the reference."""

    ids: tuple[int, ...]
    sources: tuple[str, ...]
    references: tuple[str, ...]
    hypotheses: tuple[str, ...]


# mt_eval 보고서에서 모델·방향이 맞는 결과를 찾는다. 없으면 KeyError로 멈춰 잘못된 비교가 조용히 넘어가지 않게
# 한다.
def _mt_entry(reports: Reports, name: str, model: str, direction: str) -> dict:
    for entry in reports(name):
        if entry["model"] == model and direction in entry["directions"]:
            return entry["directions"][direction]
    raise KeyError(f"{name}: no {model} {direction}")


# mt_eval 보고서(T3, T17)의 번역 결과를 Segments로 읽는 함수를 만들어 돌려준다. 바로 읽지 않고 함수를 돌려주는
# 것은 비교 목록을 모듈 상수로 미리 적어 두고, 실제 읽기는 reports를 받았을 때(테스트면 가짜 보고서) 하려는
# 것이다.
def mt(name: str, model: str, direction: str) -> Callable[[Reports], Segments]:
    """Translations stored by eval.mt_eval (T3, T17)."""

    def load(reports: Reports) -> Segments:
        items = _mt_entry(reports, name, model, direction)["items"]
        return Segments(
            tuple(item["id"] for item in items),
            tuple(item["source"] for item in items),
            tuple(item["reference"] for item in items),
            tuple(item["hypothesis"] for item in items),
        )

    return load


# typo_eval 보고서에서 방향·후보가 맞는 결과를 찾는다.
def _typo_entry(reports: Reports, name: str, direction: str, candidate: str) -> dict:
    for entry in reports(name):
        if entry["direction"] == direction and entry["candidate"] == candidate:
            return entry
    raise KeyError(f"{name}: no {direction} {candidate}")


# typo_eval 보고서(T32)의 한 오타 수준(clean·light·heavy) 번역을 읽는다. T32 항목에는 오타 입력만 있고 깨끗한
# 원문과 참조가 없으므로, 같은 분할의 T3 보고서에서 문장 ID로 가져온다(13절 '데이터'). COMET-22의 원문도 오타
# 입력이 아니라 깨끗한 원문이 된다(참조가 그 원문의 번역이므로).
def typo(name: str, direction: str, candidate: str, level: str) -> Callable[[Reports], Segments]:
    """Translations stored by eval.typo_eval (T32). Those items keep the noisy input but not the clean source
    or the reference, so both come from the T3 report of the same split (same FLEURS sentence IDs)."""
    plain = T3_TEST if "test" in name else T3_DEV

    def load(reports: Reports) -> Segments:
        items = _typo_entry(reports, name, direction, candidate)["levels"][level]["items"]
        clean = {
            item["id"]: item for item in _mt_entry(reports, plain, TC_BIG[direction], direction)["items"]
        }
        return Segments(
            tuple(item["id"] for item in items),
            tuple(clean[item["id"]]["source"] for item in items),
            tuple(clean[item["id"]]["reference"] for item in items),
            tuple(item["hypothesis"] for item in items),
        )

    return load


# T65 보고서(eval/quality_eval.py)에서 음성 경로(from_speech) 또는 전사 경로(from_text) 번역을 읽는다. 원문은
# 두 경로 모두 정답 전사로 둔다. 그래야 두 경로가 같은 원문·참조로 짝지어진다.
def quality(direction: str, path: str) -> Callable[[Reports], Segments]:
    """T65 translations of the recognised speech (`from_speech`) or of the true transcription (`from_text`);
    the source is the true transcription for both."""

    def load(reports: Reports) -> Segments:
        items = reports(T65_TEST)["results"][direction]["items"]
        return Segments(
            tuple(item["id"] for item in items),
            tuple(item["transcription"] for item in items),
            tuple(item["reference"] for item in items),
            tuple(item[path] for item in items),
        )

    return load


# 번역 비교 한 행의 정의: 키, 단계(T3 등), 분할, 방향, 표에 쓸 이름, a·b 쪽 데이터를 읽는 함수들, 문턱(a − b
# 눈금), 판단 규칙 문장. a·b가 튜플인 것은 오타 평균처럼 여러 그룹(light, heavy)을 함께 비교하기 위해서다.
@dataclass(frozen=True)
class MtComparison:
    """chrF of a minus b over the same sentences; several groups (typo levels) are averaged."""

    key: str
    stage: str
    split: str
    direction: str
    label: str
    a: tuple[Callable[[Reports], Segments], ...]
    b: tuple[Callable[[Reports], Segments], ...]
    threshold: float | None
    rule: str


# 한 T32 후보의 여러 오타 수준을 읽는 함수 튜플을 만든다.
def _levels(name: str, direction: str, candidate: str, levels: tuple[str, ...]) -> tuple:
    return tuple(typo(name, direction, candidate, level) for level in levels)


# 오타 수준 묶음과 규칙 문장. 규칙 문장은 각 절의 판단 규칙을 a − b 눈금으로 옮긴 것으로, 보고서에 함께 적혀
# 어떤 규칙과 견줬는지 보이게 한다.
CLEAN = ("clean",)
TYPOS = ("light", "heavy")
T3_TIE = "1점 이내면 빠른 쪽: 상대가 1점 넘게 높지 않아야 결정 유지"
T17_RULE = "B 채택 조건 chrF가 A보다 0.3점 넘게 떨어지지 않음"
T32_CLEAN = "깨끗한 입력 chrF가 A보다 0.5점 넘게 떨어지지 않아야 함"
T32_TYPO = "오타 평균이 A보다 1.0점 넘게 높을 때만 채택"

# 13절 비교표의 번역 행 21개를 그대로 옮긴 목록. 순서: T3 validation 5행(tc-big − 다른 후보, 문턱 −1.0), T17
# 3행(문장 단위 B − 통째로 A, 문턱 −0.3), T32 validation 8행(두 방향 × 후보 B·C × 깨끗/오타 평균, 문턱
# −0.5/+1.0), T32 D − C 참고 1행, T32 test 2행, T65 2행(음성 경로 − 전사 경로, 문턱 없음).
# test 행은 확인 측정의 차이를 같은 문턱에 견줘 적기만 한다. 2-1절 test 비교는 서로 다른 실행(T17 test의 B, T3
# test의 A)끼리다.
MT_COMPARISONS: tuple[MtComparison, ...] = (
    # 2절 T3: 두 방향에서 고른 tc-big과 떨어진 후보의 차이. 규칙 '1점 이내면 빠른 쪽'을 a − b 눈금으로 옮겨
    # 문턱 −1.0(상대가 1점 넘게 높지 않아야 결정 유지).
    *(
        MtComparison(
            f"t3-{direction}-{short}",
            "T3",
            "validation",
            direction,
            f"tc-big − {short}",
            (mt(T3_DEV, TC_BIG[direction], direction),),
            (mt(T3_DEV, other, direction),),
            -1.0,
            T3_TIE,
        )
        for direction, short, other in [
            ("ko-en", "nllb", "nllb-200-distilled-600M"),
            ("ko-en", "qwen2.5-7b", "qwen2.5-7b"),
            ("ko-en", "opus-mt-ko-en", "opus-mt-ko-en"),
            ("en-ko", "nllb", "nllb-200-distilled-600M"),
            ("en-ko", "qwen2.5-7b", "qwen2.5-7b"),
        ]
    ),
    # 2-1절 T17: 문장 단위 번역(B)과 통째로(A). 채택 조건 'chrF가 0.3점 넘게 떨어지지 않음' → 문턱 −0.3.
    # 한→영은 누락 신호 때문에 A를 유지했으므로 chrF 조건만 본다.
    MtComparison(
        "t17-ko-en",
        "T17",
        "validation",
        "ko-en",
        "B 문장 단위 − A 통째로",
        (mt(T17_DEV, "opus-mt-tc-big-ko-en/split", "ko-en"),),
        (mt(T17_DEV, "opus-mt-tc-big-ko-en", "ko-en"),),
        -0.3,
        T17_RULE + " (A 유지는 누락 신호가 줄지 않아서)",
    ),
    MtComparison(
        "t17-en-ko",
        "T17",
        "validation",
        "en-ko",
        "B 문장 단위 − A 통째로",
        (mt(T17_DEV, "opus-mt-tc-big-en-ko/split", "en-ko"),),
        (mt(T17_DEV, "opus-mt-tc-big-en-ko", "en-ko"),),
        -0.3,
        T17_RULE,
    ),
    # T17 test 확인 행. B(T17 test)와 A(T3 test의 통째로 번역)는 서로 다른 실행의 결과다.
    MtComparison(
        "t17-en-ko-test",
        "T17",
        "test",
        "en-ko",
        "B 문장 단위 − A 통째로(T3 test)",
        (mt(T17_TEST, "opus-mt-tc-big-en-ko/split", "en-ko"),),
        (mt(T3_TEST, "opus-mt-tc-big-en-ko", "en-ko"),),
        -0.3,
        T17_RULE + " (test 확인, 결정을 바꾸지 않음)",
    ),
    # 6절 T32 validation: 두 방향 × 후보 B·C × (깨끗한 입력 문턱 −0.5, 오타 평균 문턱 +1.0) = 8행. 오타 평균은
    # light·heavy 두 그룹을 같은 문장 표본으로 평균한다.
    *(
        MtComparison(
            f"t32-{direction}-{candidate}-{kind}",
            "T32",
            "validation",
            direction,
            f"{candidate} − A {'깨끗' if kind == 'clean' else '오타 평균'}",
            _levels(T32_DEV, direction, candidate, levels),
            _levels(T32_DEV, direction, "A", levels),
            threshold,
            rule,
        )
        for direction in ("ko-en", "en-ko")
        for candidate in ("B", "C")
        for kind, levels, threshold, rule in [
            ("clean", CLEAN, -0.5, T32_CLEAN),
            ("typo", TYPOS, 1.0, T32_TYPO),
        ]
    ),
    # T32 참고 행: 7B 교정기(D, 다른 실행)와 고른 1.5B 교정기(C)의 오타 평균 차이. 문턱이 없다.
    MtComparison(
        "t32-en-ko-D-C-typo",
        "T32",
        "validation",
        "en-ko",
        "D(7B 참고) − C 오타 평균",
        _levels(T32_DEV_D, "en-ko", "D", TYPOS),
        _levels(T32_DEV, "en-ko", "C", TYPOS),
        None,
        "참고: 작은 모델의 한계로 본 1.6점",
    ),
    # T32 test 확인 2행(영→한 C − A, 깨끗/오타 평균).
    *(
        MtComparison(
            f"t32-en-ko-C-{kind}-test",
            "T32",
            "test",
            "en-ko",
            f"C − A {'깨끗' if kind == 'clean' else '오타 평균'}",
            _levels(T32_TEST, "en-ko", "C", levels),
            _levels(T32_TEST, "en-ko", "A", levels),
            threshold,
            rule + " (test 확인, 결정을 바꾸지 않음)",
        )
        for kind, levels, threshold, rule in [
            ("clean", CLEAN, -0.5, T32_CLEAN),
            ("typo", TYPOS, 1.0, T32_TYPO),
        ]
    ),
    # 9절 T65 2행: 인식 오류가 번역 품질을 깎는 몫(음성 경로 − 전사 경로). 확인 측정이라 문턱이 없다.
    *(
        MtComparison(
            f"t65-{direction}",
            "T65",
            "test",
            direction,
            "음성 경로 − 전사 경로",
            (quality(direction, "from_speech"),),
            (quality(direction, "from_text"),),
            None,
            "확인 측정, 판단 규칙 없음",
        )
        for direction in ("ko-en", "en-ko")
    ),
)


# 음성 인식 비교의 한쪽 데이터: 녹음 ID(문장 ID), 정규화된 참조, 발화별 편집 수와 참조 길이 배열.
@dataclass(frozen=True)
class Utterances:
    """One recognizer's results on a list of recordings, with per-utterance edits and reference lengths."""

    ids: tuple[int, ...]
    references: tuple[str, ...]
    edits: np.ndarray
    lengths: np.ndarray


# 발화 하나의 편집 수(대체+삭제+삽입)와 참조 길이(일치+대체+삭제)를 jiwer로 센다. 한국어는 글자 단위(CER),
# 영어는 단어 단위(WER)로, jiwer.cer·wer가 세는 방식과 같다. 발화별로 따로 세어 두어야 재표집마다 합산
# 오류율을 다시 계산할 수 있다.
def utterance_edits(language: str, reference: str, hypothesis: str) -> tuple[int, int]:
    """Edits and reference length of one utterance, as jiwer.cer (ko) or jiwer.wer (en) count them."""
    process = jiwer.process_characters if language == "ko" else jiwer.process_words
    out = process(reference, hypothesis)
    return out.substitutions + out.deletions + out.insertions, out.hits + out.substitutions + out.deletions


# stt_eval·stt_rescore 보고서에서 모델·변형(T14의 a·b 등, T2는 None)·언어가 맞는 결과를 읽는 함수를 만든다.
# 보고서의 reference·hypothesis는 이미 정규화된 문자열이라 그대로 센다.
def stt(name: str, model: str, variant: str | None, language: str) -> Callable[[Reports], Utterances]:
    """Recognition results stored by eval.stt_eval / eval.stt_rescore (normalized texts)."""

    def load(reports: Reports) -> Utterances:
        for entry in reports(name):
            if entry["model"] == model and entry.get("variant") == variant:
                items = entry["languages"][language]["items"]
                counted = [utterance_edits(language, item["reference"], item["hypothesis"]) for item in items]
                return Utterances(
                    tuple(item["id"] for item in items),
                    tuple(item["reference"] for item in items),
                    np.array([edits for edits, _ in counted], dtype=np.int64),
                    np.array([length for _, length in counted], dtype=np.int64),
                )
        raise KeyError(f"{name}: no {model} {variant}")

    return load


# 음성 인식 비교 한 행의 정의. a·b는 언어별 읽기 함수 dict이고, 두 언어면 (한국어 CER + 영어 WER) / 2 차이를
# %p로 본다.
@dataclass(frozen=True)
class SttComparison:
    """Error rate (%p) of a minus b; with both languages, the mean of Korean CER and English WER."""

    key: str
    stage: str
    split: str
    label: str
    a: dict[str, Callable[[Reports], Utterances]]
    b: dict[str, Callable[[Reports], Utterances]]
    threshold: float | None
    rule: str


# 두 쪽(a, b)의 언어별 읽기 함수 dict를 한 번에 만든다. 기본은 두 언어, 언어 하나만 주면 그 언어의 오류율
# 비교가 된다.
def _stt_pair(
    name_a, model_a, variant_a, name_b, model_b, variant_b, languages=("ko", "en")
) -> tuple[dict, dict]:
    return (
        {language: stt(name_a, model_a, variant_a, language) for language in languages},
        {language: stt(name_b, model_b, variant_b, language) for language in languages},
    )


# 13절 비교표의 음성 인식 6행: T2 turbo − large-v3·small − large-v3(문턱 +1.0%p), T14 B − A 평균(문턱
# +0.2%p)과 언어별 참고 2행, T14 test 확인 1행(기준은 T2 test의 T15 보정 결과).
T2_TIE = "가장 낮은 평균과 1%p 이내면 가장 빠른 모델"
T14_RULE = "평균 오류율이 A보다 0.2%p 넘게 나빠지지 않아야 함"
STT_COMPARISONS: tuple[SttComparison, ...] = (
    SttComparison(
        "t2-turbo-large",
        "T2",
        "validation",
        "large-v3-turbo − large-v3 (평균)",
        *_stt_pair(T2_DEV, "large-v3-turbo", None, T2_DEV, "large-v3", None),
        1.0,
        T2_TIE + ": turbo가 1%p 이내라 더 빠른 turbo",
    ),
    SttComparison(
        "t2-small-large",
        "T2",
        "validation",
        "small − large-v3 (평균)",
        *_stt_pair(T2_DEV, "small", None, T2_DEV, "large-v3", None),
        1.0,
        T2_TIE + ": small은 1%p 밖",
    ),
    SttComparison(
        "t14-b-a",
        "T14",
        "validation",
        "B VAD − A 기본값 (평균)",
        *_stt_pair(T14_DEV, "large-v3-turbo", "b", T14_DEV, "large-v3-turbo", "a"),
        0.2,
        T14_RULE,
    ),
    SttComparison(
        "t14-b-a-ko",
        "T14",
        "validation",
        "B VAD − A 기본값 (한국어 CER)",
        *_stt_pair(T14_DEV, "large-v3-turbo", "b", T14_DEV, "large-v3-turbo", "a", ("ko",)),
        None,
        "참고",
    ),
    SttComparison(
        "t14-b-a-en",
        "T14",
        "validation",
        "B VAD − A 기본값 (영어 WER)",
        *_stt_pair(T14_DEV, "large-v3-turbo", "b", T14_DEV, "large-v3-turbo", "a", ("en",)),
        None,
        "참고: 영어만 0.16%p 나빠짐",
    ),
    SttComparison(
        "t14-b-default-test",
        "T14",
        "test",
        "B VAD − 기본값(T2 test, T15 보정) (평균)",
        *_stt_pair(T14_TEST, "large-v3-turbo", "b", T2_TEST, "large-v3-turbo", None),
        0.2,
        T14_RULE + " (test 확인, 결정을 바꾸지 않음)",
    ),
)


# 번역 비교의 두 쪽을 읽어 그룹별 (a, b) 목록으로 돌려준다. 두 쪽의 문장 ID·원문·참조가 모두 같아야 하고,
# 그룹끼리도 같은 문장이어야 한다. 하나라도 다르면 짝지은 비교가 성립하지 않으므로 ValueError로 멈춘다(13절
# 관문의 '같은 문장' 조건).
def _paired_mt(comparison: MtComparison, reports: Reports) -> list[tuple[Segments, Segments]]:
    groups = []
    for load_a, load_b in zip(comparison.a, comparison.b, strict=True):
        a, b = load_a(reports), load_b(reports)
        if (a.ids, a.sources, a.references) != (b.ids, b.sources, b.references):
            raise ValueError(f"{comparison.key}: the two sides are not the same sentences")
        groups.append((a, b))
    if len({a.ids for a, _ in groups}) != 1:
        raise ValueError(f"{comparison.key}: groups cover different sentences")
    return groups


# 음성 인식 비교의 두 쪽을 언어별로 읽는다. 같은 녹음 순서(ID와 참조가 모두 같음)가 아니면 멈춘다.
def _paired_stt(comparison: SttComparison, reports: Reports) -> list[tuple[str, Utterances, Utterances]]:
    strata = []
    for language in comparison.a:
        a, b = comparison.a[language](reports), comparison.b[language](reports)
        if (a.ids, a.references) != (b.ids, b.references):
            raise ValueError(f"{comparison.key}: the two sides are not the same recordings")
        strata.append((language, a, b))
    return strata


# 구간을 계산하기 전의 관문. 저장된 문장별 결과로 기록된 corpus 점수(chrF·CER·WER)를 모두 다시 계산해 기록값과
# 1e-9 미만으로 같은지 행마다 돌려준다. 끝으로 모든 비교를 짝지어 보며 문장이 어긋나면 예외를 낸다.
# 왜: 보고서 해석(어느 필드가 무엇인지, 어느 참조를 쓰는지)이 틀리면 구간도 틀리므로, 기록된 값을 먼저 똑같이
# 재현해 읽는 방법이 맞는지 확인한다. 준비 단계와 run 직전에 66/66 통과했다(13절).
def check(reports: Reports = stored) -> list[dict]:
    """Recompute every recorded corpus score used here from the stored items (gate before any interval)."""
    rows = []

    def add(name: str, recorded: float, recomputed: float) -> None:
        rows.append(
            {
                "name": name,
                "recorded": recorded,
                "recomputed": recomputed,
                "ok": abs(recorded - recomputed) < 1e-9,
            }
        )

    for name in (T3_DEV, T3_TEST, T17_DEV, T17_TEST):
        for entry in reports(name):
            for direction, result in entry["directions"].items():
                segments = mt(name, entry["model"], direction)(reports)
                score = sacrebleu.corpus_chrf(list(segments.hypotheses), [list(segments.references)]).score
                add(f"{name} {entry['model']} {direction}", result["chrf"], score)
    # T32 chrF는 T3 보고서에서 가져온 참조로 계산해도 기록값과 같아야 한다(가져온 참조가 맞다는 확인).
    for name in (T32_DEV, T32_DEV_D, T32_TEST):
        for entry in reports(name):
            for level, result in entry["levels"].items():
                segments = typo(name, entry["direction"], entry["candidate"], level)(reports)
                score = sacrebleu.corpus_chrf(list(segments.hypotheses), [list(segments.references)]).score
                add(f"{name} {entry['direction']} {entry['candidate']} {level}", result["chrf"], score)
    for direction, result in reports(T65_TEST)["results"].items():
        for path, field in [("from_speech", "chrf_speech"), ("from_text", "chrf_text")]:
            segments = quality(direction, path)(reports)
            score = sacrebleu.corpus_chrf(list(segments.hypotheses), [list(segments.references)]).score
            add(f"{T65_TEST} {direction} {path}", result[field], score)
    # 음성 인식은 발화별 편집 수의 합 / 참조 길이의 합으로 다시 계산해 기록된 전체 오류율과 견준다.
    for name in (T2_DEV, T2_TEST, T14_DEV, T14_TEST):
        for entry in reports(name):
            for language, result in entry["languages"].items():
                utterances = stt(name, entry["model"], entry.get("variant"), language)(reports)
                rate = utterances.edits.sum() / utterances.lengths.sum()
                add(
                    f"{name} {entry['model']} {entry.get('variant') or ''} {language}",
                    result["error"],
                    float(rate),
                )
    for comparison in MT_COMPARISONS:
        _paired_mt(comparison, reports)
    for comparison in STT_COMPARISONS:
        _paired_stt(comparison, reports)
    return rows


# (원문, 번역문, 참조) 세 문자열로 만든 SHA-1 키. 같은 세 문자열은 한 번만 채점하고(13절: 3,876개) 점수를 찾을
# 때 이 키를 쓴다. 보안 용도가 아니라 식별자다. eval/mt_decide.py도 같은 키로 COMET 점수를 찾는다.
def comet_key(source: str, translation: str, reference: str) -> str:
    return hashlib.sha1(json.dumps([source, translation, reference], ensure_ascii=False).encode()).hexdigest()


# fp16 정밀도 점검에 쓸 비교. 이 두 비교의 a 쪽이 T3 validation tc-big 두 방향(258문장)이다.
PRECISION_SUBSET = (
    "t3-ko-en-nllb",
    "t3-en-ko-nllb",
)  # their `a` sides: T3 validation tc-big, both directions


# 모든 번역 비교에서 COMET-22로 채점할 (원문, 번역문, 참조)를 키로 중복 없이 모은다(처음 본 순서).
# subset='precision'이면 정밀도 점검 비교의 a 쪽만 모은다. comet-input 명령이 이 목록을 JSONL로 써서
# eval/comet_score.py에 넘긴다.
def comet_segments(reports: Reports = stored, subset: str | None = None) -> list[dict]:
    """Unique (source, translation, reference) triples of every MT comparison, in first-seen order."""
    seen: dict[str, dict] = {}
    for comparison in MT_COMPARISONS:
        if subset == "precision" and comparison.key not in PRECISION_SUBSET:
            continue
        sides = [a for a, _ in _paired_mt(comparison, reports)]
        if subset is None:
            sides += [b for _, b in _paired_mt(comparison, reports)]
        for segments in sides:
            for source, translation, reference in zip(
                segments.sources, segments.hypotheses, segments.references, strict=True
            ):
                key = comet_key(source, translation, reference)
                seen.setdefault(key, {"key": key, "src": source, "mt": translation, "ref": reference})
    return list(seen.values())


# 같은 문장들을 두 방식(GPU fp16, CPU fp32)으로 채점한 점수를 견준다: 문장 점수 차이의 최댓값, 방향별 시스템
# 평균의 차이. 두 한도를 모두 지키면 passes가 True이고, 그러면 fp16 점수를 쓴다.
# 두 파일의 문장 집합이 다르거나 NaN·무한대가 있으면 비교하지 않고 멈춘다.
def comet_agreement(first: dict[str, float], second: dict[str, float], reports: Reports = stored) -> dict:
    """fp16 against fp32 on the precision subset: largest segment gap and the gap of each system mean."""
    keys = sorted(first.keys() & second.keys())
    if not keys or len(keys) != len(first) or len(keys) != len(second):
        raise ValueError("the two score files must hold the same segments")
    if not all(math.isfinite(scores[key]) for scores in (first, second) for key in keys):
        raise ValueError("a COMET score is not a finite number")
    segment_gap = max(abs(first[key] - second[key]) for key in keys)
    mean_gaps = {}
    for comparison in MT_COMPARISONS:
        if comparison.key in PRECISION_SUBSET:
            segments = _paired_mt(comparison, reports)[0][0]
            system = [
                comet_key(*triple)
                for triple in zip(segments.sources, segments.hypotheses, segments.references, strict=True)
            ]
            mean_gaps[comparison.direction] = abs(
                np.mean([first[k] for k in system]) - np.mean([second[k] for k in system])
            )
    worst_mean = max(mean_gaps.values())
    return {
        "segments": len(keys),
        "max_segment_gap": float(segment_gap),
        "mean_gaps": {direction: float(gap) for direction, gap in mean_gaps.items()},
        # bool(): worst_mean is a numpy float, so the comparison is a numpy bool that json cannot print.
        "passes": bool(segment_gap <= COMET_MAX_SEGMENT_GAP and worst_mean <= COMET_MAX_MEAN_GAP),
    }


# Interval을 보고서용 dict로 바꾼다: 점 추정, 구간, 구간이 0을 포함하는지, 문턱에 견준 분류.
def _interval_dict(interval: Interval, threshold: float | None = None) -> dict:
    return {
        "point": interval.point,
        "low": interval.low,
        "high": interval.high,
        "contains_zero": interval.contains(0.0),
        "classification": classify(interval, threshold),
    }


# sacrebleu 지표의 설정 서명 문자열(버전·토크나이저 등)을 얻는다. 참조 개수는 한 번 채점해야 알기 때문에 더미
# 문장으로 한 번 채점한 뒤 서명을 읽는다. 보고서에 남겨 같은 설정으로 다시 계산할 수 있게 한다.
def signature(metric) -> str:
    """sacrebleu's signature string; it knows the reference count only after scoring something."""
    metric.corpus_score(["a"], [["a"]])
    return str(metric.get_signature())


# spBLEU: tokenize='flores200'(FLORES-200 관례). 13a 토크나이저는 한국어를 어절로만 나눠 거칠어서 두 방향 모두
# 이것을 쓴다(13절 '지표와 구간'). 처음 쓸 때 토크나이저 모델을 내려받는다.
def bleu() -> BLEU:
    return BLEU(tokenize="flores200")  # spBLEU, the FLORES-200 convention, for both target languages


# 번역 비교 한 행의 결과를 만든다: chrF 차이 구간과 분류(문턱 기준), spBLEU 차이 구간(문턱 없음), 그룹이
# 하나면 sacrebleu PairedTest의 p값 두 가지, COMET 점수가 있으면 COMET-22 차이 구간과 chrF와의 부호 일치.
# COMET 점수는 0~1이라 ×100 해서 차이를 낸다(13절 표의 눈금). 오타 평균처럼 그룹이 여럿이면 p값은 계산하지
# 않는다(13절: 한 수준짜리 비교에만 적음).
def analyse_mt(
    comparison: MtComparison, reports: Reports, comet: dict[str, float] | None, rounds: int, trials: int
) -> dict:
    groups = _paired_mt(comparison, reports)
    # 그룹마다 (a 번역문, b 번역문, 참조). 같은 문장 표본을 모든 그룹에 쓰는 corpus_difference로 chrF·spBLEU
    # 차이 구간을 낸다.
    texts = [(a.hypotheses, b.hypotheses, a.references) for a, b in groups]
    chrf = corpus_difference(CHRF(), texts, rounds)
    result = {
        "key": comparison.key,
        "stage": comparison.stage,
        "split": comparison.split,
        "direction": comparison.direction,
        "label": comparison.label,
        "groups": len(groups),
        "sentences": len(groups[0][0].ids),
        "threshold": comparison.threshold,
        "rule": comparison.rule,
        "chrf": _interval_dict(chrf, comparison.threshold),
        "bleu": _interval_dict(corpus_difference(bleu(), texts, rounds)),
        "comet": None,
    }
    # 그룹이 하나인 비교만 sacrebleu PairedTest의 p값(붓스트랩, approximate randomization)을 함께 적는다.
    if len(groups) == 1:
        a, b = groups[0]
        tests = paired_test(
            {"chrf": CHRF(), "bleu": bleu()}, a.hypotheses, b.hypotheses, a.references, rounds, trials
        )
        for metric in ("chrf", "bleu"):
            result[metric].update(tests[metric])
    # 그룹마다 a·b의 문장 순서대로 COMET 점수 배열을 만든다. same_sign_as_chrf는 COMET 차이와 chrF 차이의
    # 부호가 같은지다('다시 볼 후보'를 가리는 데 씀).
    if comet is not None:
        scored = [
            tuple(
                np.array(
                    [comet[comet_key(*t)] for t in zip(s.sources, s.hypotheses, s.references, strict=True)]
                )
                for s in pair
            )
            for pair in groups
        ]
        interval = mean_difference([(a * 100, b * 100) for a, b in scored], rounds)
        result["comet"] = _interval_dict(interval)
        result["comet"]["score_a"] = float(np.mean([a.mean() for a, _ in scored]) * 100)
        result["comet"]["score_b"] = float(np.mean([b.mean() for _, b in scored]) * 100)
        result["comet"]["same_sign_as_chrf"] = math.copysign(1, interval.point) == math.copysign(
            1, chrf.point
        )
    return result


# 음성 인식 비교 한 행의 결과를 만든다. 편집 수에 100을 곱해 넘기므로 구간이 %p 단위로 나온다.
# 두 가지 구간을 함께 낸다: 발화 단위(계획대로)와 문장 묶음(by_group으로 같은 문장 ID의 녹음을 합쳐 뽑음).
# 분류는 문장 묶음 구간으로 한다. 같은 문장을 여러 사람이 읽어 녹음끼리 독립이 아니기 때문이다(13절 '계획 변경
# (측정 전)').
# by_language는 언어별 오류율 차이(%p)의 점 추정이다.
def analyse_stt(comparison: SttComparison, reports: Reports, rounds: int) -> dict:
    strata = _paired_stt(comparison, reports)
    utterance = rate_difference([(a.edits * 100, b.edits * 100, a.lengths) for _, a, b in strata], rounds)
    grouped = []
    for _, a, b in strata:
        edits_a, edits_b, lengths = by_group(a.ids, a.edits, b.edits, a.lengths)
        grouped.append((edits_a * 100, edits_b * 100, lengths))
    sentence = rate_difference(grouped, rounds)
    return {
        "key": comparison.key,
        "stage": comparison.stage,
        "split": comparison.split,
        "label": comparison.label,
        "languages": {language: len(a.ids) for language, a, _ in strata},
        "sentences": {language: len(set(a.ids)) for language, a, _ in strata},
        "threshold": comparison.threshold,
        "rule": comparison.rule,
        "utterance": _interval_dict(utterance, comparison.threshold),
        "sentence": _interval_dict(sentence, comparison.threshold),
        "classification": classify(sentence, comparison.threshold),
        "by_language": {
            language: 100 * (a.edits.sum() - b.edits.sum()) / a.lengths.sum() for language, a, b in strata
        },
    }


# run 명령. 관문(check)을 먼저 다시 통과해야 하고, 하나라도 어긋나면 계산하지 않고 멈춘다. COMET 점수 파일을
# 주면 NaN이 없고 모든 비교 문장의 점수가 있는지 확인한 뒤 쓴다.
# 출력: reports/reanalysis_<tag>_<UTC 시각>.json(설정·서명·모든 행), .md(표), COMET를 쓴 경우 실제로 쓴 문장
# 점수만 담은 _comet22.json. 기본 경로의 보고서 파일 이름을 돌려준다.
def run(args: argparse.Namespace, reports: Reports = stored) -> Path:
    # 관문: 기록값을 재현하지 못하면 구간을 계산하지 않는다(처음 몇 개만 보여 준다).
    failed = [row for row in check(reports) if not row["ok"]]
    if failed:
        raise SystemExit(f"stored items do not reproduce the recorded scores: {failed[:3]}")
    comet = meta = None
    # COMET 점수 파일 확인: 모든 점수가 유한한 수이고, 비교에 쓰는 모든 (원문, 번역문, 참조)의 점수가 있어야
    # 한다. 빠진 문장이 있으면 평균이 다른 문장 묶음으로 계산되므로 멈춘다.
    if args.comet:
        data = json.loads(Path(args.comet).read_text(encoding="utf-8"))
        comet, meta = data["scores"], data["meta"]
        if not all(math.isfinite(value) for value in comet.values()):
            raise SystemExit("a COMET score is not a finite number")
        missing = [s["key"] for s in comet_segments(reports) if s["key"] not in comet]
        if missing:
            raise SystemExit(f"{len(missing)} segments have no COMET score")
    # 번역 21행, 음성 인식 6행을 계산하고, 재현에 필요한 설정(반복 수, 시드, sacrebleu 버전과 서명, COMET
    # 메타데이터)을 결과에 함께 담는다.
    mt_rows = [analyse_mt(c, reports, comet, args.rounds, args.trials) for c in MT_COMPARISONS]
    stt_rows = [analyse_stt(c, reports, args.rounds) for c in STT_COMPARISONS]
    stamp = datetime.now(UTC)
    result = {
        "created": stamp.isoformat(timespec="seconds"),
        "rounds": args.rounds,
        "trials": args.trials,
        "seed": SEED,
        "sacrebleu": sacrebleu.__version__,
        "signatures": {"chrf": signature(CHRF()), "bleu": signature(bleu())},
        "comet": meta,
        "mt": mt_rows,
        "stt": stt_rows,
    }
    base = REPORTS / f"reanalysis_{args.tag}_{stamp:%Y%m%d_%H%M%S}"
    base.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    # 보고서와 함께 쓴 COMET 점수(메타데이터 포함)를 남겨, 나중에 같은 구간을 모델 없이 다시 계산할 수 있게
    # 한다.
    if comet is not None:
        used = {s["key"] for s in comet_segments(reports)}
        kept = {"meta": meta, "scores": {key: comet[key] for key in sorted(used)}}
        Path(f"{base}_comet22.json").write_text(json.dumps(kept, indent=0), encoding="utf-8")
    base.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    return base


# 구간을 '+점 [+하한, +상한]' 형식의 문자열로 만든다(부호를 늘 붙임).
def _fmt(interval: dict, digits: int = 2) -> str:
    return f"{interval['point']:+.{digits}f} [{interval['low']:+.{digits}f}, {interval['high']:+.{digits}f}]"


# 결과 dict를 사람이 읽는 Markdown 보고서로 바꾼다. 번역 표(chrF·p값·spBLEU·COMET-22)와 음성 인식 표(발화 단위
# 구간, 문장 묶음 구간, 분류)를 쓴다. 13절의 결과 표는 이 보고서에서 옮겼다.
def markdown(result: dict) -> str:
    lines = [
        "# 기록된 결정의 짝지은 구간 (재분석, T80)",
        "",
        f"- 날짜: {result['created']}, 붓스트랩 {result['rounds']:,}번, 무작위 교환 {result['trials']:,}번, "
        f"시드 {result['seed']}",
        f"- sacrebleu {result['sacrebleu']}: {result['signatures']['chrf']} / {result['signatures']['bleu']}",
        "- 고르지 않는다. 차이는 a − b, [ ]는 95% 구간. "
        "분류: 견고 = 구간 전체가 점 추정과 같은 쪽, 문턱 걸침 = 문턱이 구간 안",
    ]
    if result["comet"]:
        meta = result["comet"]
        lines.append(
            f"- COMET-22: {meta['model']} @ {meta['revision'][:7]}, {meta['device']} {meta['precision']}, "
            f"segment {meta['segments']}, ×100"
        )
    lines += [
        "",
        "## 번역",
        "",
        "| 단계 | 분할 | 방향 | 비교 (a − b) | 문장 | chrF 차이 [구간] | 문턱 | 분류 | p (bs / ar) "
        "| spBLEU 차이 [구간] | COMET-22 차이 [구간] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["mt"]:
        chrf = row["chrf"]
        p = f"{chrf['p_bootstrap']:.4f} / {chrf['p_randomization']:.4f}" if "p_bootstrap" in chrf else "-"
        threshold = "-" if row["threshold"] is None else f"{row['threshold']:+.1f}"
        comet = _fmt(row["comet"]) if row["comet"] else "-"
        groups = "" if row["groups"] == 1 else f" ×{row['groups']}"
        label = LABELS[chrf["classification"]]
        lines.append(
            f"| {row['stage']} | {row['split']} | {row['direction']} | {row['label']} "
            f"| {row['sentences']}{groups} | {_fmt(chrf)} | {threshold} | {label} | {p} "
            f"| {_fmt(row['bleu'])} | {comet} |"
        )
    lines += [
        "",
        "## 음성 인식 (%p, 한국어 CER·영어 WER)",
        "",
        "| 단계 | 분할 | 비교 (a − b) | 녹음 (문장) | 차이 [발화 단위 구간] | 문장 묶음 구간 | 문턱 "
        "| 분류 (문장 묶음) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["stt"]:
        counts = ", ".join(f"{lang} {n} ({row['sentences'][lang]})" for lang, n in row["languages"].items())
        sentence = row["sentence"]
        threshold = "-" if row["threshold"] is None else f"{row['threshold']:+.1f}"
        label = LABELS[row["classification"]]
        lines.append(
            f"| {row['stage']} | {row['split']} | {row['label']} | {counts} | {_fmt(row['utterance'])} "
            f"| [{sentence['low']:+.2f}, {sentence['high']:+.2f}] | {threshold} | {label} |"
        )
    return "\n".join(lines) + "\n"


# 명령줄 진입점. 하위 명령: check(관문만, 실패하면 종료 코드 1), comet-input(채점할 문장 JSONL 쓰기),
# comet-agreement(fp16·fp32 점수 비교 결과 출력), run(구간 계산과 보고서).
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="recompute the recorded scores from the stored items")
    segments = commands.add_parser("comet-input", help="write the unique segments to score with COMET-22")
    segments.add_argument("--output", required=True)
    segments.add_argument("--subset", choices=["precision"], help="only the fp16/fp32 check subset")
    agreement = commands.add_parser(
        "comet-agreement", help="compare fp16 and fp32 scores of the check subset"
    )
    agreement.add_argument("first")
    agreement.add_argument("second")
    analysis = commands.add_parser("run", help="compute the intervals and write the report")
    analysis.add_argument("--comet", help="scores from eval/comet_score.py")
    analysis.add_argument("--tag", required=True)
    analysis.add_argument("--rounds", type=int, default=ROUNDS)
    analysis.add_argument("--trials", type=int, default=TRIALS)
    args = parser.parse_args()

    if args.command == "check":
        rows = check()
        for row in rows:
            if not row["ok"]:
                print(f"MISMATCH {row['name']}: recorded {row['recorded']} recomputed {row['recomputed']}")
        print(f"{sum(row['ok'] for row in rows)}/{len(rows)} recorded scores reproduced, pairs aligned")
        if not all(row["ok"] for row in rows):
            raise SystemExit(1)
    elif args.command == "comet-input":
        rows = comet_segments(subset=args.subset)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        print(f"{len(rows)} segments -> {args.output}")
    elif args.command == "comet-agreement":
        first, second = (
            json.loads(Path(p).read_text(encoding="utf-8"))["scores"] for p in (args.first, args.second)
        )
        print(json.dumps(comet_agreement(first, second), indent=1))
    else:
        print(f"report: {run(args)}")


if __name__ == "__main__":
    main()
