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

TC_BIG = {"ko-en": "opus-mt-tc-big-ko-en", "en-ko": "opus-mt-tc-big-en-ko"}
LABELS = {"robust": "견고", "straddles": "문턱 걸침", "opposite": "반대쪽", None: "참고"}
# COMET fp16 against fp32 on the precision subset (docs/experiments.md 13): within these, fp16 scores stand.
COMET_MAX_SEGMENT_GAP = 0.01
COMET_MAX_MEAN_GAP = 0.001

Reports = Callable[[str], object]


@cache
def stored(name: str) -> object:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Segments:
    """One system's translations of a sentence list, with the clean source and the reference."""

    ids: tuple[int, ...]
    sources: tuple[str, ...]
    references: tuple[str, ...]
    hypotheses: tuple[str, ...]


def _mt_entry(reports: Reports, name: str, model: str, direction: str) -> dict:
    for entry in reports(name):
        if entry["model"] == model and direction in entry["directions"]:
            return entry["directions"][direction]
    raise KeyError(f"{name}: no {model} {direction}")


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


def _typo_entry(reports: Reports, name: str, direction: str, candidate: str) -> dict:
    for entry in reports(name):
        if entry["direction"] == direction and entry["candidate"] == candidate:
            return entry
    raise KeyError(f"{name}: no {direction} {candidate}")


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


def _levels(name: str, direction: str, candidate: str, levels: tuple[str, ...]) -> tuple:
    return tuple(typo(name, direction, candidate, level) for level in levels)


CLEAN = ("clean",)
TYPOS = ("light", "heavy")
T3_TIE = "1점 이내면 빠른 쪽: 상대가 1점 넘게 높지 않아야 결정 유지"
T17_RULE = "B 채택 조건 chrF가 A보다 0.3점 넘게 떨어지지 않음"
T32_CLEAN = "깨끗한 입력 chrF가 A보다 0.5점 넘게 떨어지지 않아야 함"
T32_TYPO = "오타 평균이 A보다 1.0점 넘게 높을 때만 채택"

MT_COMPARISONS: tuple[MtComparison, ...] = (
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


@dataclass(frozen=True)
class Utterances:
    """One recognizer's results on a list of recordings, with per-utterance edits and reference lengths."""

    ids: tuple[int, ...]
    references: tuple[str, ...]
    edits: np.ndarray
    lengths: np.ndarray


def utterance_edits(language: str, reference: str, hypothesis: str) -> tuple[int, int]:
    """Edits and reference length of one utterance, as jiwer.cer (ko) or jiwer.wer (en) count them."""
    process = jiwer.process_characters if language == "ko" else jiwer.process_words
    out = process(reference, hypothesis)
    return out.substitutions + out.deletions + out.insertions, out.hits + out.substitutions + out.deletions


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


def _stt_pair(
    name_a, model_a, variant_a, name_b, model_b, variant_b, languages=("ko", "en")
) -> tuple[dict, dict]:
    return (
        {language: stt(name_a, model_a, variant_a, language) for language in languages},
        {language: stt(name_b, model_b, variant_b, language) for language in languages},
    )


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


def _paired_stt(comparison: SttComparison, reports: Reports) -> list[tuple[str, Utterances, Utterances]]:
    strata = []
    for language in comparison.a:
        a, b = comparison.a[language](reports), comparison.b[language](reports)
        if (a.ids, a.references) != (b.ids, b.references):
            raise ValueError(f"{comparison.key}: the two sides are not the same recordings")
        strata.append((language, a, b))
    return strata


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


def comet_key(source: str, translation: str, reference: str) -> str:
    return hashlib.sha1(json.dumps([source, translation, reference], ensure_ascii=False).encode()).hexdigest()


PRECISION_SUBSET = (
    "t3-ko-en-nllb",
    "t3-en-ko-nllb",
)  # their `a` sides: T3 validation tc-big, both directions


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


def _interval_dict(interval: Interval, threshold: float | None = None) -> dict:
    return {
        "point": interval.point,
        "low": interval.low,
        "high": interval.high,
        "contains_zero": interval.contains(0.0),
        "classification": classify(interval, threshold),
    }


def signature(metric) -> str:
    """sacrebleu's signature string; it knows the reference count only after scoring something."""
    metric.corpus_score(["a"], [["a"]])
    return str(metric.get_signature())


def bleu() -> BLEU:
    return BLEU(tokenize="flores200")  # spBLEU, the FLORES-200 convention, for both target languages


def analyse_mt(
    comparison: MtComparison, reports: Reports, comet: dict[str, float] | None, rounds: int, trials: int
) -> dict:
    groups = _paired_mt(comparison, reports)
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
    if len(groups) == 1:
        a, b = groups[0]
        tests = paired_test(
            {"chrf": CHRF(), "bleu": bleu()}, a.hypotheses, b.hypotheses, a.references, rounds, trials
        )
        for metric in ("chrf", "bleu"):
            result[metric].update(tests[metric])
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


def run(args: argparse.Namespace, reports: Reports = stored) -> Path:
    failed = [row for row in check(reports) if not row["ok"]]
    if failed:
        raise SystemExit(f"stored items do not reproduce the recorded scores: {failed[:3]}")
    comet = meta = None
    if args.comet:
        data = json.loads(Path(args.comet).read_text(encoding="utf-8"))
        comet, meta = data["scores"], data["meta"]
        if not all(math.isfinite(value) for value in comet.values()):
            raise SystemExit("a COMET score is not a finite number")
        missing = [s["key"] for s in comet_segments(reports) if s["key"] not in comet]
        if missing:
            raise SystemExit(f"{len(missing)} segments have no COMET score")
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
    if comet is not None:
        used = {s["key"] for s in comet_segments(reports)}
        kept = {"meta": meta, "scores": {key: comet[key] for key in sorted(used)}}
        Path(f"{base}_comet22.json").write_text(json.dumps(kept, indent=0), encoding="utf-8")
    base.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    return base


def _fmt(interval: dict, digits: int = 2) -> str:
    return f"{interval['point']:+.{digits}f} [{interval['low']:+.{digits}f}, {interval['high']:+.{digits}f}]"


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
