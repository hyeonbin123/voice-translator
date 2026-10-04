"""Paired intervals for corpus chrF and BLEU (needs sacrebleu, from the eval group, so not run in CI)."""

import numpy as np
import pytest

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


def test_full_counts_reproduce_the_corpus_score():
    metric = CHRF()
    stats = segment_statistics(metric, SYS_A, REFS)
    score = corpus_scores(metric, stats, np.ones((1, len(REFS)), dtype=np.int64))[0]
    assert score == pytest.approx(sacrebleu.corpus_chrf(SYS_A, [REFS]).score, abs=1e-9)


def test_corpus_difference_point_is_the_corpus_score_difference():
    interval = corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS)], rounds=200)
    expected = sacrebleu.corpus_chrf(SYS_A, [REFS]).score - sacrebleu.corpus_chrf(SYS_B, [REFS]).score
    assert interval.point == pytest.approx(expected, abs=1e-9)
    assert interval.low <= interval.point <= interval.high


def test_corpus_difference_resamples_statistics_not_sentence_scores():
    # Resampling sums the n-gram statistics, so a resample's score is not the mean of sentence chrF.
    metric = CHRF()
    stats = segment_statistics(metric, SYS_A, REFS)
    counts = resample_counts(len(REFS), rounds=3)
    sentence = np.array([sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(SYS_A, REFS, strict=True)])
    resampled = corpus_scores(metric, stats, counts)
    expected = [metric._compute_score_from_stats((row[:, None] * stats).sum(axis=0)).score for row in counts]
    assert resampled == pytest.approx(expected, abs=1e-9)
    assert not np.allclose(resampled, counts @ sentence / len(REFS))


def test_identical_systems_give_a_zero_interval():
    interval = corpus_difference(CHRF(), [(SYS_A, SYS_A, REFS)], rounds=200)
    assert (interval.point, interval.low, interval.high) == (0.0, 0.0, 0.0)


def test_corpus_difference_is_reproducible():
    first = corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS)], rounds=300)
    assert first == corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS)], rounds=300)
    assert first != corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS)], rounds=300, seed=SEED + 1)


@pytest.mark.parametrize("metric", [CHRF(), BLEU()], ids=["chrF", "BLEU"])
def test_system_interval_equals_sacrebleus_paired_bootstrap(metric):
    # Same seed and size: the resamples here are the ones sacrebleu's paired bootstrap draws.
    rounds = 300
    _, scores = PairedTest(
        [("b", SYS_B), ("a", SYS_A)], {"m": metric}, [REFS], test_type="bs", n_samples=rounds
    )()
    (result_b, result_a) = next(iter(v for k, v in scores.items() if k != "System"))
    counts = resample_counts(len(REFS), rounds=rounds)
    for hypotheses, result in [(SYS_A, result_a), (SYS_B, result_b)]:
        mean, ci = estimate_ci(corpus_scores(metric, segment_statistics(metric, hypotheses, REFS), counts))
        # sacrebleu scores its resamples from float32 statistics; other resamples would differ far more.
        assert mean == pytest.approx(result.mean, abs=1e-4)
        assert ci == pytest.approx(result.ci, abs=1e-4)


def test_groups_share_one_resample_of_sentences():
    # Two groups (e.g. light and heavy typos) of the same sentences: the point is the mean of the group
    # differences, and a group whose systems agree adds nothing to the spread.
    one = corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS)], rounds=200)
    two = corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS), (SYS_A, SYS_A, REFS)], rounds=200)
    assert two.point == pytest.approx(one.point / 2, abs=1e-9)
    assert two.low == pytest.approx(one.low / 2, abs=1e-9)
    assert two.high == pytest.approx(one.high / 2, abs=1e-9)


def test_groups_must_cover_the_same_sentences():
    with pytest.raises(ValueError):
        corpus_difference(CHRF(), [(SYS_A, SYS_B, REFS), (SYS_A[:5], SYS_B[:5], REFS[:5])], rounds=10)


def test_paired_test_reports_p_values_and_system_intervals():
    result = paired_test({"chrF": CHRF()}, SYS_A, SYS_B, REFS, rounds=200, trials=500)["chrF"]
    assert result["score_a"] == pytest.approx(sacrebleu.corpus_chrf(SYS_A, [REFS]).score)
    assert result["score_b"] == pytest.approx(sacrebleu.corpus_chrf(SYS_B, [REFS]).score)
    assert 0 < result["p_bootstrap"] <= 1 and 0 < result["p_randomization"] <= 1
    assert result["ci_a"] > 0 and result["ci_b"] > 0
