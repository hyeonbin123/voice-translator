"""The re-analysis reads the stored reports right (no intervals are computed on them here).

It scores with sacrebleu and jiwer from the eval group, which CI does not install, so CI skips this module.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

jiwer = pytest.importorskip("jiwer")
pytest.importorskip("sacrebleu")
from sacrebleu.metrics import BLEU  # noqa: E402

from eval import reanalysis  # noqa: E402
from eval.reanalysis import (  # noqa: E402
    MT_COMPARISONS,
    STT_COMPARISONS,
    MtComparison,
    Segments,
    SttComparison,
    Utterances,
    analyse_mt,
    analyse_stt,
    check,
    comet_agreement,
    comet_key,
    comet_segments,
    markdown,
    mt,
    stored,
    typo,
    utterance_edits,
)
from eval.significance import rate_difference  # noqa: E402


def test_stored_items_reproduce_every_recorded_score():
    # The gate in docs/experiments.md 13: chrF, CER and WER recomputed from the items equal the recorded
    # values (T32 chrF with the references from the T3 reports), and each comparison pairs the same sentences.
    rows = check()
    assert [row["name"] for row in rows if not row["ok"]] == []
    # T3 7 + 2, T17 4 + 1, T32 18 + 6 + 6, T65 4, T2 6 + 2, T14 8 + 2
    assert len(rows) == 66


def test_typo_sentences_take_source_and_reference_from_t3():
    noisy = typo(reanalysis.T32_DEV, "en-ko", "C", "heavy")(stored)
    clean = mt(reanalysis.T3_DEV, "opus-mt-tc-big-en-ko", "en-ko")(stored)
    assert noisy.ids == clean.ids and len(noisy.ids) == 129
    assert noisy.sources == clean.sources and noisy.references == clean.references
    assert noisy.hypotheses != clean.hypotheses


def test_comparison_keys_are_unique_and_thresholds_registered():
    keys = [c.key for c in MT_COMPARISONS] + [c.key for c in STT_COMPARISONS]
    assert len(keys) == len(set(keys)) == 27
    thresholds = {c.key: c.threshold for c in MT_COMPARISONS + STT_COMPARISONS}
    assert thresholds["t3-ko-en-nllb"] == -1.0
    assert thresholds["t17-en-ko"] == -0.3
    assert thresholds["t32-en-ko-C-clean"] == -0.5 and thresholds["t32-en-ko-C-typo"] == 1.0
    assert thresholds["t2-turbo-large"] == 1.0 and thresholds["t14-b-a"] == 0.2
    assert thresholds["t65-ko-en"] is None


@pytest.mark.parametrize(
    ("language", "reference", "hypothesis"),
    [
        ("ko", "이백만년동안", "200만년동안"),
        ("en", "the cat sat on the mat", "the cat sat on mat"),
        ("ko", "말", ""),
    ],
)
def test_utterance_edits_match_jiwer_rates(language, reference, hypothesis):
    edits, length = utterance_edits(language, reference, hypothesis)
    rate = jiwer.cer if language == "ko" else jiwer.wer
    assert edits / length == pytest.approx(rate(reference, hypothesis))


def test_comet_segments_are_unique_and_cover_every_side():
    rows = comet_segments()
    keys = [row["key"] for row in rows]
    assert len(keys) == len(set(keys))
    known = set(keys)
    for comparison in MT_COMPARISONS:
        for load in comparison.a + comparison.b:
            segments = load(stored)
            triples = zip(segments.sources, segments.hypotheses, segments.references, strict=True)
            assert all(comet_key(*triple) in known for triple in triples)


def test_precision_subset_is_t3_tc_big_in_both_directions():
    rows = comet_segments(subset="precision")
    assert len(rows) == 258
    ko_en = mt(reanalysis.T3_DEV, "opus-mt-tc-big-ko-en", "ko-en")(stored)
    assert {row["mt"] for row in rows} >= set(ko_en.hypotheses)


def test_comet_agreement_applies_the_registered_limits():
    keys = [row["key"] for row in comet_segments(subset="precision")]
    first = {key: 0.8 for key in keys}
    assert comet_agreement(first, {key: 0.8 + 0.0005 for key in keys})["passes"]
    shifted = dict(first)
    shifted[keys[0]] = 0.8 + 0.02  # one segment too far
    assert not comet_agreement(first, shifted)["passes"]
    assert not comet_agreement(first, {key: 0.8 + 0.002 for key in keys})["passes"]  # system means too far
    with pytest.raises(ValueError):
        comet_agreement(first, {key: 0.8 for key in keys[1:]})


REFS = ("The cat sat on the mat.", "It is raining today.", "We meet at noon.", "Prices rose last year.")


def _segments(hypotheses):
    return lambda reports: Segments((1, 2, 3, 4), ("s1", "s2", "s3", "s4"), REFS, tuple(hypotheses))


def test_analyse_mt_reports_chrf_bleu_and_comet_on_the_same_resamples():
    a = _segments(
        ["The cat sat on the mat.", "It rains today.", "We meet at noon.", "Prices rose last year."]
    )
    b = _segments(["A cat sits on a mat.", "Raining.", "We meet.", "Prices went up."])
    comparison = MtComparison("x", "T3", "validation", "ko-en", "a − b", (a,), (b,), -1.0, "rule")
    comet = {}
    for load, value in [(a, 0.9), (b, 0.7)]:
        segments = load(None)
        for triple in zip(segments.sources, segments.hypotheses, segments.references, strict=True):
            comet[comet_key(*triple)] = value
    row = analyse_mt(comparison, None, comet, rounds=200, trials=200)
    assert row["chrf"]["point"] > 0 and row["chrf"]["classification"] == "robust"
    assert 0 < row["chrf"]["p_bootstrap"] <= 1
    assert row["comet"]["point"] == pytest.approx(20.0)  # ×100
    assert row["comet"]["same_sign_as_chrf"]
    text = markdown(
        {
            "created": "now",
            "rounds": 200,
            "trials": 200,
            "seed": 1,
            "sacrebleu": "2.6.0",
            "signatures": {"chrf": "c", "bleu": "b"},
            "comet": None,
            "mt": [row],
            "stt": [],
        }
    )
    assert "| T3 | validation | ko-en | a − b | 4 |" in text and "견고" in text


def test_analyse_mt_rejects_sides_on_different_sentences():
    a = _segments(["x"] * 4)
    other = lambda reports: Segments((1, 2, 3, 5), ("s1", "s2", "s3", "s4"), REFS, ("y",) * 4)  # noqa: E731
    comparison = MtComparison("x", "T3", "validation", "ko-en", "a − b", (a,), (other,), None, "rule")
    with pytest.raises(ValueError):
        analyse_mt(comparison, None, None, rounds=10, trials=10)


def test_analyse_stt_groups_recordings_by_sentence():
    ids = (1, 1, 2, 3)  # sentence 1 read twice
    lengths = np.array([10, 10, 10, 10])
    worse = lambda reports: Utterances(ids, ("r",) * 4, np.array([2, 2, 1, 1]), lengths)  # noqa: E731
    better = lambda reports: Utterances(ids, ("r",) * 4, np.array([1, 1, 1, 1]), lengths)  # noqa: E731
    comparison = SttComparison("y", "T2", "validation", "a − b", {"ko": worse}, {"ko": better}, 1.0, "rule")
    row = analyse_stt(comparison, None, rounds=300)
    assert row["utterance"]["point"] == pytest.approx(5.0)  # 2 more edits in 40 characters, in %p
    assert row["sentences"] == {"ko": 3} and row["languages"] == {"ko": 4}
    assert row["classification"] == row["sentence"]["classification"]
    # The sentence interval resamples the three sentences (sentence 1 with both its recordings) ...
    grouped = rate_difference(
        [(np.array([400, 100, 100]), np.array([200, 100, 100]), np.array([20, 10, 10]))], 300
    )
    assert (row["sentence"]["low"], row["sentence"]["high"]) == pytest.approx((grouped.low, grouped.high))
    # ... and the utterance interval the four recordings.
    single = rate_difference([(np.array([200, 200, 100, 100]), np.array([100] * 4), lengths)], 300)
    assert (row["utterance"]["low"], row["utterance"]["high"]) == pytest.approx((single.low, single.high))


def test_run_writes_the_report_and_the_comet_scores_it_used(tmp_path, monkeypatch):
    a = _segments(
        ["The cat sat on the mat.", "It rains today.", "We meet at noon.", "Prices rose last year."]
    )
    b = _segments(["A cat sits on a mat.", "Raining.", "We meet.", "Prices went up."])
    ids, lengths = (1, 1, 2, 3), np.array([10, 10, 10, 10])
    worse = lambda reports: Utterances(ids, ("r",) * 4, np.array([2, 2, 1, 1]), lengths)  # noqa: E731
    better = lambda reports: Utterances(ids, ("r",) * 4, np.array([1, 1, 1, 1]), lengths)  # noqa: E731
    monkeypatch.setattr(
        reanalysis,
        "MT_COMPARISONS",
        (MtComparison("x", "T3", "validation", "ko-en", "a − b", (a,), (b,), -1.0, "rule"),),
    )
    monkeypatch.setattr(
        reanalysis,
        "STT_COMPARISONS",
        (SttComparison("y", "T2", "validation", "a − b", {"ko": worse}, {"ko": better}, 1.0, "rule"),),
    )
    monkeypatch.setattr(reanalysis, "check", lambda reports: [])
    monkeypatch.setattr(reanalysis, "bleu", BLEU)  # 13a here: flores200 would download its tokenizer
    monkeypatch.setattr(reanalysis, "REPORTS", tmp_path)
    scores = {row["key"]: 0.5 for row in comet_segments(None)}
    scores["unused"] = 0.1
    comet_file = tmp_path / "scores.json"
    comet_file.write_text(
        json.dumps(
            {
                "meta": {
                    "model": "m",
                    "revision": "r" * 40,
                    "device": "cuda",
                    "precision": "fp16",
                    "segments": 9,
                },
                "scores": scores,
            }
        )
    )
    args = argparse.Namespace(comet=str(comet_file), tag="unit", rounds=100, trials=100)
    base = reanalysis.run(args, reports=None)
    result = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    assert [row["key"] for row in result["mt"]] == ["x"] and [row["key"] for row in result["stt"]] == ["y"]
    assert "nrefs:1" in result["signatures"]["chrf"] and "nrefs:1" in result["signatures"]["bleu"]
    kept = json.loads(Path(f"{base}_comet22.json").read_text(encoding="utf-8"))
    assert "unused" not in kept["scores"] and len(kept["scores"]) == 8
    assert "| T2 | validation | a − b | ko 4 (3) |" in base.with_suffix(".md").read_text(encoding="utf-8")


def test_run_refuses_missing_comet_scores(tmp_path, monkeypatch):
    a = _segments(["x"] * 4)
    comparison = MtComparison("x", "T3", "validation", "ko-en", "a − b", (a,), (a,), None, "rule")
    monkeypatch.setattr(reanalysis, "MT_COMPARISONS", (comparison,))
    monkeypatch.setattr(reanalysis, "STT_COMPARISONS", ())
    monkeypatch.setattr(reanalysis, "check", lambda reports: [])
    monkeypatch.setattr(reanalysis, "REPORTS", tmp_path)
    comet_file = tmp_path / "scores.json"
    comet_file.write_text(json.dumps({"meta": {}, "scores": {}}))
    with pytest.raises(SystemExit):
        reanalysis.run(argparse.Namespace(comet=str(comet_file), tag="u", rounds=10, trials=10), reports=None)
