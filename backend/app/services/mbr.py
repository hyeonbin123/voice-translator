"""빔 탐색의 n-best에서 문장 chrF MBR로 고르는 서비스용 디코딩(docs/experiments.md 20절).

MBR(minimum Bayes risk) 디코딩: 빔 탐색이 준 번역 후보 n개마다 나머지 후보들을 의사 참조로 두고 문장 chrF의
평균(기대 효용)을 구해, 그 값이 가장 큰 후보를 번역으로 고른다. 20절의 validation 측정에서 한→영은 빔 8의 가설
8개에 이 고르기를 건 것이 빔 4의 1-best보다 chrF +0.87 [+0.03, +1.79]·COMET-22 +0.40 [+0.09, +0.74] 높고 번역
시간 p50이 12ms 늘어 채택됐다(영→한은 COMET-22 차이가 유의하지 않아 채택되지 않았고 빔 4의 1-best 그대로다).
서버는 KO_EN_DECODING=mbr8(기본)일 때 이 모듈의 MbrTranslator로 한→영 번역기를 감싼다(services/models.py).
"""

# [파일 역할] 서비스 안의 MBR 고르기 세 부분: 문장 chrF 효용 표(chrf_matrix), 고르기(select), 번역기
# 감싸개(MbrTranslator).
# [설계 이유] 측정 도구 eval/mbr.py는 효용을 sacrebleu(eval 그룹)로 계산하지만 서버 이미지에는 sacrebleu가
# 없다. 그래서 sacrebleu 2.6.0 CHRF의 기본 설정(문자 1~6-gram, 단어 n-gram 없음, beta 2, 공백 제외,
# 소문자화·eps 평활 없음)을 표준 라이브러리만으로 같은 계산 순서로 옮겼다: 값이 소수점 끝자리까지 같고, 그래서
# 측정 때 고른 후보를 서버도 똑같이 고른다(tests/test_mbr_service.py가 sacrebleu 공개 함수·eval/mbr.py와
# 견준다. CI의 sacrebleu 단계). 측정 도구(eval/mbr.py)는 측정 때 쓴 그대로 두고 고치지 않았다. 서비스는 모델
# 스레드가 여럿일 수 있어 번역기에 마지막 결과를 남기지 않는다(측정 도구의 picks 같은 상태 없음).
# [연결] 후보는 translation.MarianTranslator.candidates(_Ct2Model.generate_nbest)가 만든다.
# services/models.load_translation이 설정(config.Settings.ko_en_decoding)에 따라 감싼다. 테스트:
# tests/test_mbr_service.py.

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Protocol

from app.services.interfaces import Language, ModelError
from app.services.translation import _nonempty

# sacrebleu CHRF의 기본값과 같은 설정(20절 '효용' = 2절과 같은 chrF). CHAR_ORDER는 문자 n-gram의 최대 차수,
# BETA는 F_beta에서 재현율에 두는 무게다.
CHAR_ORDER = 6
BETA = 2

# 효용 표 함수의 모양: 후보 글 목록을 받아 표[i][j] = 효용(가설 = 글 i, 의사 참조 = 글 j)을 돌려준다.
# 대각선(자기 자신)은 쓰지 않는다. 테스트는 정해 둔 표를 돌려주는 대역을 넣는다.
UtilityMatrix = Callable[[Sequence[str]], list[list[float]]]


def _char_ngrams(text: str) -> list[Counter[str]]:
    """글 하나의 문자 n-gram 수: 차수 1~CHAR_ORDER 순서의 Counter 목록.

    sacrebleu `extract_all_char_ngrams`(whitespace=False)와 같이 모든 공백 문자를 지운 뒤 센다. 소문자로
    바꾸지 않는다(기본 설정). 글이 차수보다 짧으면 그 차수의 Counter는 빈 것이다.
    """
    line = "".join(text.split())
    return [Counter(line[i : i + n] for i in range(len(line) - n + 1)) for n in range(1, CHAR_ORDER + 1)]


def _match_statistics(hypothesis: Counter[str], reference: Counter[str]) -> tuple[int, int, int]:
    """한 차수의 (가설 n-gram 수, 참조 n-gram 수, 일치 수). sacrebleu `_get_match_statistics`와 같다.

    참조에 그 차수의 n-gram이 하나도 없으면(참조가 차수보다 짧음) 가설 수를 0으로 보고해 그 차수가 평균에서
    빠지게 한다(sacrebleu의 처리 그대로). 일치 수는 n-gram마다 두 쪽 수의 작은 값의 합이다.
    """
    matched = total = 0
    for ngram, count in hypothesis.items():
        total += count
        if ngram in reference:
            matched += min(count, reference[ngram])
    return (total if reference else 0, sum(reference.values()), matched)


def _f_score(statistics: Sequence[tuple[int, int, int]]) -> float:
    """차수별 통계에서 chrF(0~100). sacrebleu `_compute_f_score`(eps_smoothing=False)와 같은 계산 순서다.

    가설·참조 n-gram이 둘 다 있는 차수만 세어(실효 차수) 정밀도·재현율을 차수마다 더한 뒤 실효 차수로 나누고,
    F_beta = (1 + beta²)·P·R / (beta²·P + R)에 100을 곱한다. 실효 차수가 0이거나 P + R이 0이면 0.0이다.
    부동소수점 연산을 sacrebleu와 같은 순서로 해 값이 끝자리까지 같다(sacrebleu의 eps 평활 누적은
    eps_smoothing=True에서만 쓰여 여기서는 계산하지 않는다).
    """
    factor = BETA**2
    effective_order = 0
    precision = recall = 0.0
    for hypothesis_count, reference_count, matched in statistics:
        if hypothesis_count > 0 and reference_count > 0:
            precision += matched / hypothesis_count
            recall += matched / reference_count
            effective_order += 1
    if effective_order == 0:
        return 0.0
    precision /= effective_order
    recall /= effective_order
    if precision + recall:
        score = (1 + factor) * precision * recall
        score /= (factor * precision) + recall
        return 100 * score
    return 0.0


def sentence_chrf(hypothesis: str, reference: str) -> float:
    """문장 chrF 하나(가설과 참조 하나).

    sacrebleu `CHRF().sentence_score(hypothesis, [reference]).score`와 같은 값이다.
    """
    pairs = zip(_char_ngrams(hypothesis), _char_ngrams(reference), strict=True)
    return _f_score([_match_statistics(hyp, ref) for hyp, ref in pairs])


def chrf_matrix(texts: Sequence[str]) -> list[list[float]]:
    """후보끼리의 문장 chrF 표: 표[i][j] = texts[i]를 가설, texts[j]를 참조로 둔 문장 chrF.

    측정 도구 `eval.mbr.chrf_matrix`(sacrebleu 내부 함수로 계산)와 같은 값이다(tests/test_mbr_service.py).
    후보마다 n-gram을 한 번만 뽑아 짝마다 일치 수만 센다. chrF는 대칭이 아니다(beta 2라 재현율 쪽 무게가
    크다). 대각선은 0.0으로 두고 쓰지 않는다.
    """
    grams = [_char_ngrams(text) for text in texts]
    table: list[list[float]] = []
    for i, hypothesis in enumerate(grams):
        row: list[float] = []
        for j, reference in enumerate(grams):
            if i == j:
                row.append(0.0)
                continue
            pairs = zip(hypothesis, reference, strict=True)
            row.append(_f_score([_match_statistics(hyp, ref) for hyp, ref in pairs]))
        table.append(row)
    return table


def select(candidates: Sequence[tuple[str, float]], matrix: UtilityMatrix = chrf_matrix) -> int:
    """MBR로 고른 후보의 위치. 입력: (번역문, 엔진 점수) 목록, 효용 표 함수.

    측정 도구 `eval.mbr.select`와 같은 규칙이다(tests/test_mbr_service.py가 무작위 후보 묶음으로 견준다).
    후보 i의 기대 효용 = 자기 자신을 뺀 다른 후보 j 모두를 의사 참조로 둔 효용(표[i][j])의 평균. 이 값이 가장
    큰 후보를 고르고, 같으면 엔진 점수가 높은 후보, 그것도 같으면 앞 후보(엔진이 준 순서)다. 평균은
    math.fsum으로 더해, 같은 값들이면 더하는 순서와 상관없이 같은 평균이 되게 한다(그래야 같은 효용을 동점으로
    본다). 겹치는 글은 따로 센다: 엔진의 가설 하나가 의사 참조 하나다. 후보가 하나면 표를 만들지 않고 0,
    없으면 ValueError(후보 함수는 빈 목록을 내지 않는다).
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

    def candidates(
        self, text: str, source: Language, target: Language, beam_size: int, n: int
    ) -> list[list[tuple[str, float]]]: ...


class MbrTranslator:
    """빔 탐색의 n-best에서 문장마다 MBR로 고르는 번역기(서비스용, 20절에서 채택한 한→영 디코딩).

    translate()는 감싼 번역기의 candidates로 문장마다 후보 n개를 받고(감싼 번역기가 문장 단위면 그 문장마다,
    아니면 입력 전체에 하나), 문장마다 select로 고른 뒤 공백 하나로 잇는다(서비스의 문장 단위 번역과 같은
    이음). 측정 도구 eval/mbr.py의 MbrTranslator와 같은 계산이지만 효용은 이 모듈의 chrf_matrix(sacrebleu
    없이)로 구하고, 모델 스레드가 여럿일 수 있어 번역마다 남기는 상태가 없다. 후보 함수의 ModelError(엔진
    오류, 빈 번역, 다른 방향)는 그대로 올라가고, 고르기 안의 다른 예외는 번역 실패(ModelError, API 503)로
    바꾼다. 빈 결과는 빈 번역 실패다. model_name은 감싼 모델 이름 뒤에 '/b<빔>-mbr<n>'을 붙여(측정 보고서의 팔
    이름과 같은 꼴) 번역 기록·/api/health에 어떤 디코딩인지 남긴다. 가설 수가 1~빔 크기가 아니면 설정 실수라
    만들 때 ValueError다.
    """

    def __init__(
        self, inner: NbestTranslator, beam: int = 8, n: int = 8, matrix: UtilityMatrix = chrf_matrix
    ) -> None:
        if not 1 <= n <= beam:
            raise ValueError(f"need between 1 and beam={beam} hypotheses, got {n}")
        self._inner, self.beam, self.n = inner, beam, n
        self._matrix = matrix
        self.model_name = f"{inner.model_name}/b{beam}-mbr{n}"

    def translate(self, text: str, source: Language, target: Language) -> str:
        groups = self._inner.candidates(text, source, target, self.beam, self.n)
        try:
            chosen = [candidates[select(candidates, self._matrix)][0] for candidates in groups]
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - 고르기를 감싸는 예외 경계(번역 실패로 바꿈)
            raise ModelError(f"the MBR selection failed: {type(exc).__name__}") from exc
        return _nonempty(" ".join(chosen))
