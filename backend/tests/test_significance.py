"""Paired intervals for error rates and segment-score means; numpy only, so these run in CI as well."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eval.significance import (
    SEED,
    Interval,
    by_group,
    classify,
    mean_difference,
    rate_difference,
    resample_counts,
)


def test_resample_counts_are_sacrebleus_draws():
    counts = resample_counts(6, rounds=50, seed=SEED)
    picks = np.random.default_rng(SEED).choice(6, size=(50, 6), replace=True)
    expected = np.stack([np.bincount(row, minlength=6) for row in picks])
    assert counts.shape == (50, 6)
    assert (counts == expected).all()
    assert (counts.sum(axis=1) == 6).all()


def test_rate_difference_is_paired():
    # whisper-ko-ft tests/test_metrics.py: 2 vs 1 edit in every 10-character utterance -> exactly 0.1.
    lengths = np.full(100, 10)
    interval = rate_difference([(np.full(100, 2), np.full(100, 1), lengths)], rounds=500)
    assert [interval.point, interval.low, interval.high] == pytest.approx([0.1] * 3, abs=1e-12)


def test_rate_difference_is_corpus_level():
    # 1 edit in 2 characters and 0 in 8 against 0 everywhere: 1/10, not the mean of 0.5 and 0.
    interval = rate_difference([(np.array([1, 0]), np.array([0, 0]), np.array([2, 8]))], rounds=10)
    assert interval.point == pytest.approx(0.1)


def test_rate_difference_averages_strata_resampled_apart():
    korean = (np.full(50, 2), np.full(50, 1), np.full(50, 10))  # always +0.1
    english = (np.full(80, 3), np.full(80, 3), np.full(80, 20))  # always 0
    interval = rate_difference([korean, english], rounds=300)
    assert [interval.point, interval.low, interval.high] == pytest.approx([0.05] * 3, abs=1e-12)


def test_rate_interval_contains_the_point_and_is_reproducible():
    rng = np.random.default_rng(1)
    lengths = rng.integers(5, 40, size=200)
    worse, better = rng.binomial(lengths, 0.12), rng.binomial(lengths, 0.1)
    interval = rate_difference([(worse, better, lengths)], rounds=1000)
    assert interval.low < interval.point < interval.high
    assert interval == rate_difference([(worse, better, lengths)], rounds=1000)


def test_rate_difference_rejects_empty_references():
    with pytest.raises(ValueError):
        rate_difference([(np.array([1, 0]), np.array([0, 0]), np.array([2, 0]))], rounds=10)


def test_by_group_sums_the_recordings_of_a_sentence():
    ids = [7, 3, 7, 9, 3]
    edits, lengths = by_group(ids, np.array([1, 2, 3, 4, 5]), np.array([10, 20, 30, 40, 50]))
    assert edits.tolist() == [4, 7, 4]  # ids 7, 3, 9 in first-seen order
    assert lengths.tolist() == [40, 70, 40]


def test_mean_difference_is_paired_and_averages_groups():
    a = np.linspace(0.5, 0.9, 40)
    interval = mean_difference([(a + 0.02, a)], rounds=300)
    assert [interval.point, interval.low, interval.high] == pytest.approx([0.02] * 3, abs=1e-12)
    two = mean_difference([(a + 0.02, a), (a, a)], rounds=300)
    assert [two.point, two.low, two.high] == pytest.approx([0.01] * 3, abs=1e-12)


def test_mean_difference_uses_the_same_resamples_as_corpus_difference():
    rng = np.random.default_rng(3)
    a, b = rng.uniform(0.6, 0.9, 6), rng.uniform(0.6, 0.9, 6)
    counts = resample_counts(6, rounds=200)
    draws = (counts @ a - counts @ b) / 6
    interval = mean_difference([(a, b)], rounds=200)
    assert interval.low == pytest.approx(np.percentile(draws, 2.5))
    assert interval.high == pytest.approx(np.percentile(draws, 97.5))


@pytest.mark.parametrize(
    ("interval", "threshold", "label"),
    [
        (Interval(0.5, -0.4, 1.3), -1.0, "robust"),  # whole interval above the threshold, like the point
        (Interval(0.5, -1.2, 1.3), -1.0, "straddles"),
        (Interval(-1.2, -1.9, -0.4), -0.5, "straddles"),
        (Interval(-1.2, -1.9, -0.7), -0.5, "robust"),  # whole interval below, like the point
        (Interval(0.27, -0.1, 0.6), 1.0, "robust"),
        (Interval(-1.0, -1.5, -1.0), -1.0, "straddles"),  # touching the threshold is not clear of it
        (Interval(0.5, -0.4, 1.3), None, None),
    ],
)
def test_classify_against_the_decision_threshold(interval, threshold, label):
    assert classify(interval, threshold) == label


def test_classify_flags_an_interval_on_the_other_side_of_the_point():
    assert classify(Interval(0.1, -0.9, -0.2), 0.0) == "opposite"


def test_the_intervals_import_without_the_eval_group():
    # CI has no sacrebleu or jiwer (eval group): the rate and mean intervals must still import there.
    code = (
        "import sys; sys.modules['sacrebleu'] = None; sys.modules['jiwer'] = None; import eval.significance"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])
