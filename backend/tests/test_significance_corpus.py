"""말뭉치 chrF와 BLEU의 짝지은 구간을 검증한다(eval 그룹의 sacrebleu가 필요하다. CI는 sacrebleu를 더한
별도 단계에서 이 모듈을 돌린다)."""

# [파일 역할] eval/significance.py 중 sacrebleu가 필요한 말뭉치 지표(chrF·BLEU) 구간 함수를 검증한다: 문장별
# 통계로 말뭉치 점수 다시 만들기, 차이 구간, sacrebleu의 짝지은 붓스트랩과 같은 결과, 여러 묶음(오타 수준)의
# 한 번 재표집, p값 보고.
# [왜 필요한가] chrF는 문장 점수의 평균이 아니라 문장별 n-gram 통계를 더해 한 번에 계산하는 말뭉치 점수다.
# 재표집도 통계를 더하는 방식이어야 기록된 chrF와 같은 정의의 구간이 나온다(docs/experiments.md 13절).
# [실행 환경] sacrebleu는 eval 의존성 그룹에만 있어, 없으면 이 파일 전체를 건너뛴다. CI는 이 파일을
# sacrebleu를 얹은 별도 단계에서 돌린다(.github/workflows/ci.yml).

import numpy as np
import pytest

# importorskip이 sacrebleu가 없을 때 모듈을 건너뛰게 하므로, 그 뒤의 import는 이 줄 아래에 있어야 한다(E402
# 경고를 끄는 표시는 그래서 붙어 있다). CHRF·BLEU는 지표 객체, PairedTest·estimate_ci는 결과를 견줄 sacrebleu
# 자체 구현이다.
sacrebleu = pytest.importorskip("sacrebleu")
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402
from sacrebleu.significance import PairedTest, estimate_ci  # noqa: E402

from eval.significance import (  # noqa: E402
    SEED,
    corpus_difference,
    corpus_scores,
    paired_test,
    resample_counts,
    segment_statistics,
)

# 참조 6문장과 두 시스템의 번역. A는 참조에 가깝고 B는 단어를 더 많이 바꿔, chrF 차이가 0이 아닌 쌍이 되게
# 만든 작은 말뭉치다.
REFS = [
    "The cat sat on the mat.",
    "It is raining in Seoul today.",
    "We will meet at the station at noon.",
    "The museum opens again next spring.",
    "She translated the letter into Korean.",
    "Prices rose by three percent last year.",
]
SYS_A = [
    "The cat sat on a mat.",
    "Today it rains in Seoul.",
    "We meet at the station at noon.",
    "The museum will open next spring.",
    "She translated the letter to Korean.",
    "Prices went up three percent last year.",
]
SYS_B = [
    "A cat is sitting on the mat.",
    "It rains in Seoul.",
    "We will meet at noon.",
    "The museum opens next year.",
    "She wrote the letter in Korean.",
    "Prices rose last year.",
]


# 모든 문장을 한 번씩 뽑은 행(전부 1)으로 계산한 점수가 sacrebleu.corpus_chrf와 같은지 확인한다. 문장별 통계를
# 더해 점수를 내는 경로(segment_statistics → corpus_scores)가 sacrebleu와 같은 정의라는 기준점이다.
def test_full_counts_reproduce_the_corpus_score():
    metric = CHRF()
    stats = segment_statistics(metric, SYS_A, REFS)
    score = corpus_scores(metric, stats, np.ones((1, len(REFS)), dtype=np.int64))[0]
    assert score == pytest.approx(sacrebleu.corpus_chrf(SYS_A, [REFS]).score, abs=1e-9)


# 구간의 점 추정이 두 시스템 말뭉치 chrF의 차이와 같고, 점 추정이 구간 안에 있는지 확인한다.
def test_corpus_difference_point_is_the_corpus_score_difference():
    interval = corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS)], rounds=200)
    expected = sacrebleu.corpus_chrf(SYS_A, [REFS]).score - sacrebleu.corpus_chrf(SYS_B, [REFS]).score
    assert interval.point == pytest.approx(expected, abs=1e-9)
    assert interval.low <= interval.point <= interval.high


# 재표집마다 점수를 '뽑힌 문장들의 통계 합'으로 다시 계산하는지(expected와 같음), 그리고 그것이 문장 chrF의
# 평균과는 다른지 확인한다. 문장 점수 평균으로 바꾸는 회귀를 잡는다.
def test_corpus_difference_resamples_statistics_not_sentence_scores():
    metric = CHRF()
    stats = segment_statistics(metric, SYS_A, REFS)
    counts = resample_counts(len(REFS), rounds=3)
    sentence = np.array([sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(SYS_A, REFS, strict=True)])
    resampled = corpus_scores(metric, stats, counts)
    expected = [metric._compute_score_from_stats((row[:, None] * stats).sum(axis=0)).score for row in counts]
    assert resampled == pytest.approx(expected, abs=1e-9)
    assert not np.allclose(resampled, counts @ sentence / len(REFS))


# 같은 번역끼리 비교하면 점 추정과 구간이 정확히 0이다. 두 시스템에 같은 재표집을 쓴다는 짝지음의 기본
# 성질이다.
def test_identical_systems_give_a_zero_interval():
    interval = corpus_difference(CHRF(), [(SYS_A, SYS_A, REFS)], rounds=200)
    assert (interval.point, interval.low, interval.high) == (0.0, 0.0, 0.0)


# 같은 시드면 같은 구간, 시드를 바꾸면 다른 구간이 나오는지 확인한다(시드가 실제로 재표집에 쓰인다는 확인).
def test_corpus_difference_is_reproducible():
    first = corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS)], rounds=300)
    assert first == corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS)], rounds=300)
    assert first != corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS)], rounds=300, seed=SEED + 1)


# 같은 시드·같은 크기면 이 모듈의 재표집이 sacrebleu PairedTest(test_type="bs")의 재표집과 같아, 시스템별
# 평균과 95% 반폭이 sacrebleu 결과와 1e-4 안에서 일치하는지 chrF·BLEU 모두 확인한다. 남는 작은 차이는
# sacrebleu가 float32 통계로 계산하기 때문이다(테스트 주석).
# PairedTest에는 기준 시스템(b)을 먼저 넘기는 것이 sacrebleu의 약속이다. 결과 표에서 'System' 열을 뺀 첫 지표
# 열이 (b, a) 결과 쌍이다.
@pytest.mark.parametrize("metric", [CHRF(), BLEU()], ids=["chrF", "BLEU"])
def test_system_interval_equals_sacrebleus_paired_bootstrap(metric):
    rounds = 300
    _, scores = PairedTest(
        [("b", SYS_B), ("a", SYS_A)], {"m": metric}, [REFS], test_type="bs", n_samples=rounds
    )()
    (result_b, result_a) = next(iter(v for k, v in scores.items() if k != "System"))
    counts = resample_counts(len(REFS), rounds=rounds)
    for hypotheses, result in [(SYS_A, result_a), (SYS_B, result_b)]:
        mean, ci = estimate_ci(corpus_scores(metric, segment_statistics(metric, hypotheses, REFS), counts))
        # sacrebleu는 재표집 점수를 float32 통계로 계산해 작은 차이가 남는다. 재표집이 달랐다면 차이가 훨씬
        # 컸을 것이다.
        assert mean == pytest.approx(result.mean, abs=1e-4)
        assert ci == pytest.approx(result.ci, abs=1e-4)


# 여러 묶음(예: 가벼운 오타·심한 오타)은 문장 위치를 한 번만 뽑아 모든 묶음에 같이 쓴다. 두 번째 묶음의 두
# 시스템이 같으면 차이가 늘 0이라, 결과는 첫 묶음 구간을 정확히 반으로 줄인 값이어야 한다. 묶음마다 따로
# 뽑았다면 반이 정확히 나오지 않는다.
def test_groups_share_one_resample_of_sentences():
    one = corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS)], rounds=200)
    two = corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS), (SYS_A, SYS_A, REFS)], rounds=200)
    assert two.point == pytest.approx(one.point / 2, abs=1e-9)
    assert two.low == pytest.approx(one.low / 2, abs=1e-9)
    assert two.high == pytest.approx(one.high / 2, abs=1e-9)


# 묶음끼리 문장 수가 다르면 같은 위치를 뽑을 수 없으므로 ValueError로 멈춘다.
def test_groups_must_cover_the_same_sentences():
    with pytest.raises(ValueError):
        corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS), (SYS_A[:5], SYS_B[:5], REFS[:5])], rounds=10)


# paired_test가 sacrebleu의 짝지은 붓스트랩과 근사 무작위화 검정을 돌려, 두 시스템의 말뭉치 점수(sacrebleu와
# 같음), 두 p값(0 초과 1 이하), 시스템별 구간 반폭(양수)을 보고하는지 확인한다.
def test_paired_test_reports_p_values_and_system_intervals():
    result = paired_test({"chrF": CHRF()}, SYS_A, SYS_B, REFS, rounds=200, trials=500)["chrF"]
    assert result["score_a"] == pytest.approx(sacrebleu.corpus_chrf(SYS_A, [REFS]).score)
    assert result["score_b"] == pytest.approx(sacrebleu.corpus_chrf(SYS_B, [REFS]).score)
    assert 0 < result["p_bootstrap"] <= 1 and 0 < result["p_randomization"] <= 1
    assert result["ci_a"] > 0 and result["ci_b"] > 0
