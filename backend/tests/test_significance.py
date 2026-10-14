"""오류율과 문장 점수 평균의 짝지은 구간을 검증한다. numpy만 쓰므로 CI에서도 돈다."""

# [파일 역할] eval/significance.py의 짝지은 붓스트랩 구간 함수 중 numpy만 쓰는 부분(오류율 차이, 문장 점수
# 평균 차이, 재표집 횟수 행렬, 문장별 묶기, 문턱 대비 분류)을 검증한다.
# [왜 필요한가] 기록된 모델 결정에 95% 구간을 붙인 재분석(docs/experiments.md 13절)과 Hy-MT2 채택 규칙
# (a)(15절)가 모두 이 함수들로 판정됐다. 구간이 짝지어 계산되는지, 말뭉치 단위인지, 결과가 재현되는지가 틀리면
# '견고/문턱 걸침' 판정이 달라진다.
# [방법] 답을 손으로 알 수 있는 입력(모든 항목의 차이가 같으면 구간이 한 점으로 모임)을 주로 쓴다. sacrebleu가
# 필요한 chrF·BLEU 쪽은 test_significance_corpus.py에서 따로 본다.

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# SEED는 sacrebleu 기본 시드(12345)다. 같은 시드·크기면 sacrebleu의 짝지은 붓스트랩과 같은 표본을 뽑게 하려는
# 값이다(모듈 주석).
from eval.significance import (
    SEED,
    Interval,
    by_group,
    classify,
    mean_difference,
    rate_difference,
    resample_counts,
)


# 재표집 횟수 행렬이 sacrebleu와 같은 방식(같은 시드의 default_rng로 choice 복원 추출)으로 뽑은 결과를 항목별
# 횟수로 센 것과 정확히 같은지, 각 행의 합이 항목 수(6)인지 확인한다. 이것이 맞아야 시스템별 구간이
# sacrebleu와 일치한다.
def test_resample_counts_are_sacrebleus_draws():
    counts = resample_counts(6, rounds=50, seed=SEED)
    picks = np.random.default_rng(SEED).choice(6, size=(50, 6), replace=True)
    expected = np.stack([np.bincount(row, minlength=6) for row in picks])
    assert counts.shape == (50, 6)
    assert (counts == expected).all()
    assert (counts.sum(axis=1) == 6).all()


# 짝지음 확인: 모든 발화에서 a가 b보다 10글자당 편집 1개 많으면, 어떤 문장을 뽑든 차이가 정확히 0.1이므로
# 구간도 [0.1, 0.1]이어야 한다. 두 시스템을 따로 뽑았다면 구간이 넓어진다.
def test_rate_difference_is_paired():
    lengths = np.full(100, 10)
    interval = rate_difference([(np.full(100, 2), np.full(100, 1), lengths)], rounds=500)
    assert [interval.point, interval.low, interval.high] == pytest.approx([0.1] * 3, abs=1e-12)


# 오류율은 문장별 비율의 평균이 아니라 '전체 편집 수 ÷ 전체 정답 길이'(말뭉치 단위)다. 2글자 중 1개 틀림과
# 8글자 중 0개 틀림은 0.5와 0의 평균 0.25가 아니라 1/10 = 0.1이다.
def test_rate_difference_is_corpus_level():
    interval = rate_difference([(np.array([1, 0]), np.array([0, 0]), np.array([2, 8]))], rounds=10)
    assert interval.point == pytest.approx(0.1)


# 층(예: 한국어 CER과 영어 WER)은 따로 재표집하고 두 층의 차이를 평균한다. 한국어는 늘 +0.1, 영어는 늘 0이라
# 결과는 정확히 0.05다. 두 언어를 한 표본으로 섞어 뽑으면 언어 비율이 재표집마다 달라진다.
def test_rate_difference_averages_strata_resampled_apart():
    korean = (np.full(50, 2), np.full(50, 1), np.full(50, 10))  # 늘 +0.1
    english = (np.full(80, 3), np.full(80, 3), np.full(80, 20))  # 늘 0
    interval = rate_difference([korean, english], rounds=300)
    assert [interval.point, interval.low, interval.high] == pytest.approx([0.05] * 3, abs=1e-12)


# 무작위 데이터에서 점 추정이 구간 안에 있고, 같은 입력으로 다시 계산하면 똑같은 구간이 나오는지(시드 고정
# 재현성) 확인한다.
def test_rate_interval_contains_the_point_and_is_reproducible():
    rng = np.random.default_rng(1)
    lengths = rng.integers(5, 40, size=200)
    worse, better = rng.binomial(lengths, 0.12), rng.binomial(lengths, 0.1)
    interval = rate_difference([(worse, better, lengths)], rounds=1000)
    assert interval.low < interval.point < interval.high
    assert interval == rate_difference([(worse, better, lengths)], rounds=1000)


# 정답 길이가 0인 항목이 있으면 ValueError로 멈춘다. 재표집에서 길이 0만 뽑히면 0으로 나누게 되기 때문이다.
def test_rate_difference_rejects_empty_references():
    with pytest.raises(ValueError):
        rate_difference([(np.array([1, 0]), np.array([0, 0]), np.array([2, 0]))], rounds=10)


# 같은 문장 id의 녹음들을 한 행으로 더하고, 행 순서는 id가 처음 나온 순서(7, 3, 9)다. FLEURS는 한 문장을 여러
# 사람이 읽어 같은 문장의 녹음끼리 독립이 아니므로, 재표집 단위를 녹음이 아니라 문장으로 하려는
# 것이다(by_group docstring).
def test_by_group_sums_the_recordings_of_a_sentence():
    ids = [7, 3, 7, 9, 3]
    edits, lengths = by_group(ids, np.array([1, 2, 3, 4, 5]), np.array([10, 20, 30, 40, 50]))
    assert edits.tolist() == [4, 7, 4]  # 처음 나온 순서대로 id 7, 3, 9
    assert lengths.tolist() == [40, 70, 40]


# COMET-22 같은 문장 점수 평균의 차이도 짝지어 계산되는지(모든 문장 +0.02면 구간이 0.02 한 점), 묶음이 둘이면
# 묶음별 차이(0.02와 0)를 평균하는지(0.01) 확인한다.
def test_mean_difference_is_paired_and_averages_groups():
    a = np.linspace(0.5, 0.9, 40)
    interval = mean_difference([(a + 0.02, a)], rounds=300)
    assert [interval.point, interval.low, interval.high] == pytest.approx([0.02] * 3, abs=1e-12)
    two = mean_difference([(a + 0.02, a), (a, a)], rounds=300)
    assert [two.point, two.low, two.high] == pytest.approx([0.01] * 3, abs=1e-12)


# 평균 차이 구간이 resample_counts와 같은 재표집(같은 시드)을 쓰는지, 구간 끝이 재표집 차이의 2.5·97.5
# 백분위인지 직접 계산해 비교한다. mean_difference docstring의 'corpus_difference(chrF·BLEU)와 같은 방식으로
# 재표집'이 실제로 지켜지는지 보는 것이다.
def test_mean_difference_uses_the_same_resamples_as_corpus_difference():
    rng = np.random.default_rng(3)
    a, b = rng.uniform(0.6, 0.9, 6), rng.uniform(0.6, 0.9, 6)
    counts = resample_counts(6, rounds=200)
    draws = (counts @ a - counts @ b) / 6
    interval = mean_difference([(a, b)], rounds=200)
    assert interval.low == pytest.approx(np.percentile(draws, 2.5))
    assert interval.high == pytest.approx(np.percentile(draws, 97.5))


# 구간을 결정 문턱과 견주는 분류: 구간 전체가 점 추정과 같은 쪽에 있으면 'robust'(구간 안 어디서도 결정이
# 같음), 문턱이 구간 안에 있으면 'straddles'(문턱 걸침). 구간 끝이 문턱에 닿기만 해도 걸침으로 본다. 문턱이
# 없는 비교는 분류하지 않는다(None).
@pytest.mark.parametrize(
    ("interval", "threshold", "label"),
    [
        (Interval(0.5, -0.4, 1.3), -1.0, "robust"),  # 점 추정처럼 구간 전체가 문턱 위
        (Interval(0.5, -1.2, 1.3), -1.0, "straddles"),
        (Interval(-1.2, -1.9, -0.4), -0.5, "straddles"),
        (Interval(-1.2, -1.9, -0.7), -0.5, "robust"),  # 점 추정처럼 구간 전체가 문턱 아래
        (Interval(0.27, -0.1, 0.6), 1.0, "robust"),
        (Interval(-1.0, -1.5, -1.0), -1.0, "straddles"),  # 문턱에 닿기만 해도 벗어난 것이 아니다
        (Interval(0.5, -0.4, 1.3), None, None),
    ],
)
def test_classify_against_the_decision_threshold(interval, threshold, label):
    assert classify(interval, threshold) == label


# 점 추정은 문턱 위인데 구간 전체가 문턱 아래인 비정상적인 경우는 'opposite'로 따로 표시한다(일어나리라 보지
# 않지만 생기면 보고하려는 분류).
def test_classify_flags_an_interval_on_the_other_side_of_the_point():
    assert classify(Interval(0.1, -0.9, -0.2), 0.0) == "opposite"


# CI 기본 환경처럼 sacrebleu·jiwer가 없어도 eval.significance를 import할 수 있는지 새 프로세스에서 확인한다.
# sys.modules에 None을 넣으면 그 모듈 import가 실패하므로, 모듈 맨 위에서 sacrebleu를 불러오는 회귀가 생기면
# 이 테스트가 실패한다(check=True).
def test_the_intervals_import_without_the_eval_group():
    code = (
        "import sys; sys.modules['sacrebleu'] = None; sys.modules['jiwer'] = None; import eval.significance"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])
