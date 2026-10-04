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

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # sacrebleu is in the eval group; the rate and mean intervals work without it (CI)
    from sacrebleu.metrics.base import Metric

ROUNDS = 10_000
TRIALS = 10_000  # approximate randomization
SEED = 12345  # sacrebleu's default SACREBLEU_SEED, so PairedTest draws the same resamples


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def _counts(rng: np.random.Generator, count: int, rounds: int) -> np.ndarray:
    picks = rng.choice(count, size=(rounds, count), replace=True)
    counts = np.zeros((rounds, count), dtype=np.int64)
    np.add.at(counts, (np.arange(rounds)[:, None], picks), 1)
    return counts


def resample_counts(count: int, rounds: int = ROUNDS, seed: int = SEED) -> np.ndarray:
    """How often each of `count` items is drawn in each resample, shape (rounds, count)."""
    return _counts(np.random.default_rng(seed), count, rounds)


def _interval(point: float, draws: np.ndarray) -> Interval:
    low, high = np.percentile(draws, [2.5, 97.5])
    return Interval(float(point), float(low), float(high))


def segment_statistics(metric: Metric, hypotheses: Sequence[str], references: Sequence[str]) -> np.ndarray:
    """sacrebleu's sufficient statistics, one row per sentence (one reference each)."""
    return np.asarray(metric._extract_corpus_statistics(list(hypotheses), [list(references)]), dtype=np.int64)


def corpus_scores(metric: Metric, stats: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The corpus score of every resample: the drawn sentences' statistics summed, then scored."""
    return np.array([metric._compute_score_from_stats(row).score for row in counts @ stats])


def _same_length(lengths: Sequence[int]) -> int:
    if len(set(lengths)) != 1 or lengths[0] == 0:
        raise ValueError(f"paired inputs must be non-empty and of equal length, got {sorted(set(lengths))}")
    return lengths[0]


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
    full = np.ones((1, count), dtype=np.int64)
    point, draws = 0.0, np.zeros(rounds)
    for hypotheses_a, hypotheses_b, references in groups:
        stats_a = segment_statistics(metric, hypotheses_a, references)
        stats_b = segment_statistics(metric, hypotheses_b, references)
        point += corpus_scores(metric, stats_a, full)[0] - corpus_scores(metric, stats_b, full)[0]
        draws += corpus_scores(metric, stats_a, counts) - corpus_scores(metric, stats_b, counts)
    return _interval(point / len(groups), draws / len(groups))


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
    if (interval.low > threshold) == (interval.point > threshold):
        return "robust"
    return "opposite"


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
