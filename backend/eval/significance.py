"""Paired bootstrap intervals for re-analysing recorded decisions (docs/experiments.md 13, task T80).

Every function returns the difference a - b on the full data and the 2.5th and 97.5th percentiles of the
same difference over resamples that draw the same items for both systems.

- chrF and BLEU are corpus-level scores. A resample sums sacrebleu's per-sentence statistics and recomputes
  the score, so the n-gram counts are re-aggregated instead of averaging sentence scores. The draws are the
  ones sacrebleu's paired bootstrap (`significance.PairedTest`, `test_type="bs"`) makes for the same seed
  and size, so the per-system intervals agree with sacrebleu's.
- CER and WER are corpus-level rates (total edits over total reference length). A resample sums the edits
  and lengths of the drawn utterances, or of the drawn sentences after `by_group`.
- COMET-22 system scores are means of segment scores.
"""

# [파일 역할] 두 시스템(a, b)의 점수 차이에 95% 짝지은 붓스트랩 구간을 붙이는 계산 함수 모음이다. 모델을
# 돌리지 않고 저장된 문장별 결과만으로 계산한다.
# [왜 필요한가] 2·2-1·6절은 0.3~1.0점 chrF 차이를 점 추정만으로 판정했는데, 정해 둔 여유폭이 표본 잡음보다
# 작을 수 있었다(docs/experiments.md 13절 배경). T80 재분석(13절)은 기록된 결정마다 이
# 구간을 붙여 '견고'와 '문턱 걸침'으로 나눴고, T83(15절)은 같은 함수로 Hy-MT2 채택 규칙 (a)를 판정했다.
# [설계 이유] '짝지음'은 한 번 뽑은 문장 표본을 두 시스템에 똑같이 쓰는 것이다. 같은 문장끼리 비교하므로 문장
# 난이도 차이가 빠지고 구간이 좁아진다. 시드를 sacrebleu 기본값(12345)으로 고정해 sacrebleu PairedTest와 같은
# 표본을 뽑게 했다(단위 테스트로 확인).
# [연결] eval/reanalysis.py(T80), eval/mt_decide.py(T83)가 쓰고,
# tests/test_significance.py·test_significance_corpus.py가 검증한다.

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

# numpy: 붓스트랩 10,000번을 반복문 대신 행렬 연산(뽑힌 횟수 행렬 @ 문장별 값)으로 한 번에 계산한다.
import numpy as np

# sacrebleu는 eval 그룹에만 있다. 타입 표기용으로만 가져와, sacrebleu 없이도(CI) 오류율·평균 구간 함수는 쓸 수
# 있게 한다.
if TYPE_CHECKING:  # sacrebleu is in the eval group; the rate and mean intervals work without it (CI)
    from sacrebleu.metrics.base import Metric

# ROUNDS: 붓스트랩 반복 수, TRIALS: approximate randomization 시행 수(둘 다 13절에 적은 10,000번).
# SEED: sacrebleu의 기본 SACREBLEU_SEED와 같은 값. 그래야 PairedTest가 뽑는 표본과 이 파일이 뽑는 표본이
# 같아진다.
ROUNDS = 10_000
TRIALS = 10_000  # approximate randomization
SEED = 12345  # sacrebleu's default SACREBLEU_SEED, so PairedTest draws the same resamples


# 구간 결과: point는 전체 데이터에서 계산한 차이(점 추정), low·high는 붓스트랩 차이의 2.5·97.5 백분위.
# frozen이라 만든 뒤 바꿀 수 없다. contains는 값이 구간 안(경계 포함)에 있는지 본다.
@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


# 붓스트랩 표본을 '항목별 뽑힌 횟수' 행렬로 만든다. 모양 (rounds, count): 각 행이 한 번의 재표집이고, 값은 그
# 항목이 몇 번 뽑혔는지다.
# rng.choice로 복원 추출한 위치를 np.add.at으로 세어 넣는다(같은 위치가 여러 번 뽑혀도 모두 더해짐). 이 행렬에
# 문장별 값(통계·편집 수·점수)을 곱하면 재표집마다의 합이 한 번에 나온다.
def _counts(rng: np.random.Generator, count: int, rounds: int) -> np.ndarray:
    picks = rng.choice(count, size=(rounds, count), replace=True)
    counts = np.zeros((rounds, count), dtype=np.int64)
    np.add.at(counts, (np.arange(rounds)[:, None], picks), 1)
    return counts


# 시드로 새 난수 생성기를 만들어 _counts를 부른다. 호출마다 같은 시드면 같은 표본이 나오므로, 같은 문장 수의
# 비교끼리는 같은 재표집을 쓴다.
def resample_counts(count: int, rounds: int = ROUNDS, seed: int = SEED) -> np.ndarray:
    """How often each of `count` items is drawn in each resample, shape (rounds, count)."""
    return _counts(np.random.default_rng(seed), count, rounds)


# 재표집 차이들의 2.5·97.5 백분위로 95% 백분위 구간을 만든다.
def _interval(point: float, draws: np.ndarray) -> Interval:
    low, high = np.percentile(draws, [2.5, 97.5])
    return Interval(float(point), float(low), float(high))


# sacrebleu 지표(chrF·BLEU)의 충분 통계(문자·단어 n-gram 일치 수 등)를 문장마다 한 행으로 뽑는다. 참조는
# 문장마다 하나다.
# sacrebleu의 내부 메서드(_extract_corpus_statistics)를 쓰므로 sacrebleu 버전이 바뀌면 확인이 필요하다(13절은
# 2.6.0으로 계산).
def segment_statistics(metric: Metric, hypotheses: Sequence[str], references: Sequence[str]) -> np.ndarray:
    """sacrebleu's sufficient statistics, one row per sentence (one reference each)."""
    return np.asarray(metric._extract_corpus_statistics(list(hypotheses), [list(references)]), dtype=np.int64)


# 재표집마다 뽑힌 문장들의 통계를 더한 뒤(counts @ stats) 그 합으로 corpus 점수를 다시 계산한다. 문장 점수의
# 평균이 아니라 기록된 chrF와 같은 '전체 합산' 점수를 재표집마다 만드는 것이 핵심이다.
def corpus_scores(metric: Metric, stats: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The corpus score of every resample: the drawn sentences' statistics summed, then scored."""
    return np.array([metric._compute_score_from_stats(row).score for row in counts @ stats])


# 짝지은 입력들이 모두 같은 길이이고 비어 있지 않은지 확인하고 그 길이를 돌려준다. 길이가 다르면 문장이 어긋난
# 것이라 계산하지 않고 ValueError를 낸다.
def _same_length(lengths: Sequence[int]) -> int:
    if len(set(lengths)) != 1 or lengths[0] == 0:
        raise ValueError(f"paired inputs must be non-empty and of equal length, got {sorted(set(lengths))}")
    return lengths[0]


# chrF·BLEU 같은 corpus 지표의 차이(a − b) 구간. 출력: Interval.
# groups를 여러 개 주면(예: 오타 가벼움·심함) 문장 위치를 한 번만 뽑아 모든 그룹에 똑같이 쓰고, 그룹별 차이를
# 평균한다. 그래서 한 문장의 두 수준이 늘 함께 뽑힌다(13절 '오타 평균은 문장을 한 번 뽑아 함께 쓴다').
def corpus_difference(
    metric: Metric,
    groups: Sequence[tuple[Sequence[str], Sequence[str], Sequence[str]]],
    rounds: int = ROUNDS,
    seed: int = SEED,
) -> Interval:
    """Corpus score of a minus b, averaged over groups.

    Each group is (hypotheses_a, hypotheses_b, references) over the same sentences in the same order, for
    example one group per typo level. A resample draws sentence positions once and uses them in every group,
    so a sentence keeps all its levels.
    """
    count = _same_length([len(part) for group in groups for part in group])
    counts = resample_counts(count, rounds, seed)
    # full: 모든 문장을 한 번씩 쓴 '재표집 없음' 행. 같은 함수로 점 추정을 계산하려고 둔다.
    full = np.ones((1, count), dtype=np.int64)
    point, draws = 0.0, np.zeros(rounds)
    for hypotheses_a, hypotheses_b, references in groups:
        stats_a = segment_statistics(metric, hypotheses_a, references)
        stats_b = segment_statistics(metric, hypotheses_b, references)
        point += corpus_scores(metric, stats_a, full)[0] - corpus_scores(metric, stats_b, full)[0]
        draws += corpus_scores(metric, stats_a, counts) - corpus_scores(metric, stats_b, counts)
    return _interval(point / len(groups), draws / len(groups))


# 문장 점수 평균으로 시스템 점수를 내는 지표(COMET-22, MetricX)의 차이 구간. 재표집은 corpus_difference와 같은
# 방식이고, 재표집마다 (뽑힌 횟수 @ 점수) / 문장 수가 그 표본의 평균이다.
# 호출 위치: eval/reanalysis.py(COMET-22), eval/mt_decide.py(T83의 COMET-22·MetricX 차이).
def mean_difference(
    groups: Sequence[tuple[np.ndarray, np.ndarray]], rounds: int = ROUNDS, seed: int = SEED
) -> Interval:
    """Mean segment score (COMET) of a minus b, averaged over groups, resampled like `corpus_difference`."""
    count = _same_length([len(part) for group in groups for part in group])
    counts = resample_counts(count, rounds, seed)
    point, draws = 0.0, np.zeros(rounds)
    for scores_a, scores_b in groups:
        a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
        point += a.mean() - b.mean()
        draws += (counts @ a - counts @ b) / count
    return _interval(point / len(groups), draws / len(groups))


# 음성 인식 오류율(CER·WER)의 차이 구간. 입력 strata의 각 원소는 (a의 편집 수, b의 편집 수, 참조 길이)
# 배열이다.
# 오류율은 '전체 편집 수 / 전체 참조 길이'라 재표집마다 뽑힌 항목의 편집 수와 길이를 따로 더해 나눈다(1절
# '전체 합산'과 같음). 층(예: 한국어 CER, 영어 WER)은 항목 수가 달라 따로 뽑고, 층별 차이를 평균해 '(한국어
# CER + 영어 WER) / 2' 차이를 만든다. 층들은 생성기 하나에서 차례로 뽑아 결과가 재현된다.
# 참조 길이가 0인 항목은 오류율을 정의할 수 없어 ValueError를 낸다.
def rate_difference(
    strata: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], rounds: int = ROUNDS, seed: int = SEED
) -> Interval:
    """Corpus error rate of a minus b, averaged over strata (e.g. Korean CER and English WER).

    Each stratum is (edits_a, edits_b, reference_lengths) over the same items. Strata are resampled apart,
    in order, from one generator.
    """
    rng = np.random.default_rng(seed)
    point, draws = 0.0, np.zeros(rounds)
    for edits_a, edits_b, lengths in strata:
        a, b, n = (np.asarray(x, dtype=np.int64) for x in (edits_a, edits_b, lengths))
        _same_length([len(a), len(b), len(n)])
        if (n <= 0).any():
            raise ValueError("every item needs a non-empty reference")
        point += (a.sum() - b.sum()) / n.sum()
        counts = _counts(rng, len(n), rounds)
        draws += (counts @ a - counts @ b) / (counts @ n)
    return _interval(point / len(strata), draws / len(strata))


# 같은 id(문장 ID)를 가진 항목의 값을 합쳐 id마다 한 행으로 만든다(처음 나온 순서). np.bincount의 weights로
# id별 합을 구한다.
# 왜 필요한가: FLEURS는 한 문장을 여러 사람이 읽어 같은 문장의 녹음끼리 독립이 아니다. 발화 단위로 뽑으면 이를
# 독립으로 세어 구간이 좁게 나오므로, 문장 단위로 묶어 한 문장의 녹음을 모두 함께 뽑는다(13절 '계획 변경 (측정
# 전)'). 결과로는 분류가 하나도 바뀌지 않았다.
def by_group(ids: Sequence, *columns: np.ndarray) -> tuple[np.ndarray, ...]:
    """Columns summed over items that share an id, one row per id in first-seen order.

    Resampling these rows draws whole sentences with all their recordings (FLEURS has several readers per
    sentence, so recordings of one sentence are not independent).
    """
    order = {key: index for index, key in enumerate(dict.fromkeys(ids))}
    rows = np.array([order[key] for key in ids])
    return tuple(
        np.bincount(rows, weights=column, minlength=len(order)).astype(np.int64) for column in columns
    )


# 구간을 판단 문턱에 견줘 분류한다. 문턱이 없는 비교(참고 행)는 None.
# - straddles(문턱 걸침): 문턱이 구간 안. 경계에 닿기만 해도 걸침이다. 다른 문장 표본이었다면 결정이 달라질 수
#   있었다는 뜻이다.
# - robust(견고): 구간 전체가 점 추정과 같은 쪽.
# - opposite: 구간 전체가 점 추정의 반대쪽. 백분위 구간은 보통 점 추정을 품으므로 예상하지 않지만 생기면
#   보고한다.
def classify(interval: Interval, threshold: float | None) -> str | None:
    """Where the interval lies against a decision threshold.

    "robust": the whole interval is strictly on the point estimate's side, so the recorded decision would
    not change within the interval. "straddles": the threshold is inside the interval (touching counts).
    "opposite": the whole interval is on the other side of the point (not expected; reported if it happens).
    """
    if threshold is None:
        return None
    if interval.contains(threshold):
        return "straddles"
    # 여기까지 왔으면 문턱은 구간 밖이다. 구간 하한과 점 추정이 문턱의 같은 쪽에 있으면 구간 전체가 점 추정
    # 쪽이다.
    if (interval.low > threshold) == (interval.point > threshold):
        return "robust"
    return "opposite"


# sacrebleu의 PairedTest로 짝지은 붓스트랩(bs)과 approximate randomization(ar) 검정을 돌려 지표마다 두
# 시스템의 점수, 붓스트랩 평균과 95% 반폭, 두 p값을 돌려준다. 13절 번역 표의 'p (bs / ar)' 열이 이 값이다.
# sacrebleu는 시스템 목록의 첫 번째를 기준선으로 삼으므로 b를 먼저 넣는다. p값은 a가 기준선 b와 다른지에 대한
# 것이다.
def paired_test(
    metrics: dict[str, Metric],
    hypotheses_a: Sequence[str],
    hypotheses_b: Sequence[str],
    references: Sequence[str],
    rounds: int = ROUNDS,
    trials: int = TRIALS,
) -> dict[str, dict]:
    """sacrebleu's paired bootstrap and approximate randomization tests of a against b (the baseline).

    Returns per metric both corpus scores, sacrebleu's 95% half-widths around the bootstrap means, and the
    two p-values.
    """
    from sacrebleu.significance import PairedTest

    # SACREBLEU_SEED가 다른 값으로 설정돼 있으면 sacrebleu가 다른 표본을 뽑아 이 파일의 구간과 맞지 않게
    # 되므로 미리 멈춘다.
    if os.environ.get("SACREBLEU_SEED", str(SEED)) != str(SEED):
        raise RuntimeError(f"SACREBLEU_SEED must be unset or {SEED} so the resamples match")
    systems = [("b", list(hypotheses_b)), ("a", list(hypotheses_a))]
    refs = [list(references)]
    names = list(metrics)
    _, bootstrap = PairedTest(systems, metrics, refs, test_type="bs", n_samples=rounds)()
    _, randomization = PairedTest(systems, metrics, refs, test_type="ar", n_samples=trials)()
    out = {}
    # PairedTest keys its columns by the score name (e.g. chrF2), in the order of `metrics`.
    for name, column in zip(names, [key for key in bootstrap if key != "System"], strict=True):
        result_b, result_a = bootstrap[column]
        out[name] = {
            "score_a": result_a.score,
            "score_b": result_b.score,
            "mean_a": float(result_a.mean),
            "ci_a": float(result_a.ci),
            "mean_b": float(result_b.mean),
            "ci_b": float(result_b.ci),
            "p_bootstrap": float(result_a.p_value),
            "p_randomization": float(randomization[column][1].p_value),
        }
    return out
