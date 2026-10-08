"""The T83 decision rules (docs/experiments.md 15) on made-up reports.

Needs sacrebleu (eval group), so CI skips this module.
"""

import json

import numpy as np
import pytest

pytest.importorskip("sacrebleu")

from eval import mt_decide  # noqa: E402
from eval.mt_checks import check_items  # noqa: E402
from eval.mt_decide import (  # noqa: E402
    comet_input,
    comet_key,
    compare,
    contam,
    metricx_input,
    pick,
    typo,
    validation,
    vram,
)

ROUNDS = 300
WORDS = "the river runs past the old mill and the children wave at the boats every summer morning".split()


def sentences(count: int) -> list[str]:
    return [" ".join(WORDS[i % 7 : i % 7 + 6 + i % 4]) + f" {i}." for i in range(count)]


def direction_values(hypotheses, references, p50=0.2, p95=0.5, sources=None):
    sources = sources or [f"src {i}" for i in range(len(references))]
    items = [
        {"id": i, "source": s, "reference": r, "hypothesis": h}
        for i, (s, r, h) in enumerate(zip(sources, references, hypotheses, strict=True))
    ]
    import sacrebleu

    return {
        "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score,
        "latency_p50": p50,
        "latency_p95": p95,
        "checks": check_items(items, "en-ko"),
        "items": items,
    }


def comet_for(values: dict, score: float) -> dict[str, float]:
    return {comet_key(i["source"], i["hypothesis"], i["reference"]): score for i in values["items"]}


def degraded(references):
    return [" ".join(r.split()[:-2]) for r in references]  # drops words: lower chrF


def test_an_arm_better_on_both_metrics_and_inside_the_budgets_passes():
    refs = sentences(40)
    arm, base = direction_values(refs, refs), direction_values(degraded(refs), refs)
    comet = {**comet_for(arm, 0.9), **comet_for(base, 0.7)}
    result = compare(arm, base, comet, ROUNDS)
    assert result["chrf"]["low"] > 0 and result["comet"]["low"] > 0
    assert result["rules"] == {"a": True, "b": True, "c": True} and result["passes"]


def test_rule_a_needs_both_intervals_above_zero():
    refs = sentences(40)
    arm, base = direction_values(refs, refs), direction_values(degraded(refs), refs)
    comet = {**comet_for(arm, 0.7), **comet_for(base, 0.7)}  # COMET says no difference
    assert compare(arm, base, comet, ROUNDS)["rules"]["a"] is False


def test_rule_b_allows_one_flagged_output_and_no_empty_one():
    refs = sentences(40)
    good = list(refs)
    one = good[:1] + ["東京 " + good[1]] + good[2:]
    two = one[:2] + ["東京 " + good[2]] + one[3:]
    empty = [""] + good[1:]
    base = direction_values(degraded(refs), refs)
    outcomes = []
    for hyps in (one, two, empty):
        arm = direction_values(hyps, refs)
        comet = {**comet_for(arm, 0.9), **comet_for(base, 0.7)}
        outcomes.append(compare(arm, base, comet, ROUNDS)["rules"]["b"])
    assert outcomes == [True, False, False]


@pytest.mark.parametrize(("p50", "p95", "ok"), [(0.35, 0.8, True), (0.36, 0.5, False), (0.2, 0.81, False)])
def test_rule_c_latency_limits(p50, p95, ok):
    refs = sentences(30)
    arm, base = direction_values(refs, refs, p50, p95), direction_values(degraded(refs), refs)
    comet = {**comet_for(arm, 0.9), **comet_for(base, 0.7)}
    result = compare(arm, base, comet, ROUNDS)
    assert result["rules"]["c"] is ok
    assert result["latency_only_miss"] is (not ok)


def report(name, direction, values, vram_mb=2400):
    return {"model": name, "vram_mb": vram_mb, "directions": {direction: values}}


def test_validation_picks_per_direction_and_asks_for_q4_only_after_a_latency_only_miss():
    refs = sentences(40)
    base_en = direction_values(degraded(refs), refs)
    base_ko = direction_values(degraded(refs), refs)
    whole = direction_values(refs, refs, p50=0.5)  # quality passes, too slow
    split = direction_values(refs, refs, p50=0.3)
    ko = direction_values(degraded(refs), refs)  # no better than opus
    results = [
        report("opus-mt-tc-big-en-ko/split", "en-ko", base_en, 512),
        report("opus-mt-tc-big-ko-en", "ko-en", base_ko, 514),
        {"model": "hy-mt2-q8", "vram_mb": 2400, "directions": {"en-ko": whole, "ko-en": ko}},
        report("hy-mt2-q8/split", "en-ko", split),
    ]
    comet = {}
    for values, score in ((base_en, 0.7), (base_ko, 0.7), (whole, 0.9), (split, 0.9), (ko, 0.7)):
        comet.update(comet_for(values, score))
    result = validation(results, comet, ROUNDS)
    assert result["directions"]["en-ko"]["pick"] == "hy-mt2-q8/split"
    assert result["directions"]["en-ko"]["fallback_needed"] == ["hy-mt2-q4"]
    assert result["directions"]["ko-en"]["pick"] is None
    assert result["directions"]["ko-en"]["fallback_needed"] == []


def test_pick_prefers_higher_chrf_then_the_faster_arm():
    def arm(chrf, p50, passes=True):
        return {"passes": passes, "chrf": {"arm": chrf}, "latency_p50": p50}

    assert pick({"a": arm(40.0, 0.3), "b": arm(41.0, 0.3)}) == "b"
    assert pick({"a": arm(40.0, 0.3), "b": arm(40.05, 0.32)}) == "a"
    assert pick({"a": arm(40.0, 0.3, passes=False)}) is None


def test_vram_counts_what_the_pick_replaces():
    decision = {
        "directions": {
            "en-ko": {"pick": "hy-mt2-q8/split", "arms": {"hy-mt2-q8/split": {"vram_mb": 2200}}},
            "ko-en": {"pick": None, "arms": {}},
        }
    }
    kept = vram(decision, {"retire_corrector": False, "corrector_vram_mb": 1300})
    assert kept["net_mb"] == 2200 - 512 and kept["passes"] is False
    retired = vram(decision, {"retire_corrector": True, "corrector_vram_mb": 1300})
    assert retired["net_mb"] == 2200 - 512 - 1300 and retired["passes"] is True
    edge = vram(decision, {"retire_corrector": True, "corrector_vram_mb": 1176})
    assert edge["net_mb"] == 512 and edge["passes"] is True  # +0.5 GB itself still passes
    both = {
        "directions": {
            "en-ko": {"pick": "hy-mt2-q8", "arms": {"hy-mt2-q8": {"vram_mb": 2400}}},
            "ko-en": {"pick": "hy-mt2-q8", "arms": {"hy-mt2-q8": {"vram_mb": 2400}}},
        }
    }
    # One Ollama model serves both directions: its memory counts once.
    assert vram(both, None)["net_mb"] == 2400 - 512 - 514


def typo_result(candidate, translator, hypotheses, refs, vram_mb=0):
    import sacrebleu

    levels = {}
    for level, hyps in hypotheses.items():
        items = [{"id": i, "input": "x", "corrected": "x", "hypothesis": h} for i, h in enumerate(hyps)]
        levels[level] = {"chrf": sacrebleu.corpus_chrf(hyps, [refs]).score, "items": items}
    return {
        "direction": "en-ko",
        "candidate": candidate,
        "corrector": "ollama/qwen2.5:1.5b-instruct" if candidate == "C" else None,
        "translator": translator,
        "vram_mb": vram_mb,
        "levels": levels,
    }


def test_the_corrector_retires_when_hy_mt2_alone_is_within_half_a_point():
    refs = sentences(40)
    rows = {i: {"id": i, "en": f"src {i}", "ko": r} for i, r in enumerate(refs)}
    corrected = typo_result("C", "opus", {"clean": refs, "light": refs, "heavy": refs}, refs, 1300)
    same = typo_result("A", "hy-mt2", {"clean": refs, "light": refs, "heavy": refs}, refs)
    worse = typo_result(
        "A", "hy-mt2", {"clean": refs, "light": degraded(refs), "heavy": degraded(refs)}, refs
    )
    assert typo(corrected, same, rows, None, ROUNDS)["retire_corrector"] is True
    result = typo(corrected, worse, rows, None, ROUNDS)
    assert result["retire_corrector"] is False
    assert result["typo_average"]["point"] < -0.5
    assert result["corrector_vram_mb"] == 1300


def test_contamination_vetoes_only_when_opus_is_clearly_better():
    frozen = [{"id": f"en-{i:03d}", "lang": "en", "text": f"s{i}"} for i in range(50)]
    outputs = {"hymt": {"en-ko": {}}, "opus": {"en-ko": {}}}
    rng = np.random.default_rng(1)
    worse = {f"hymt/en-ko/{r['id']}": 5 + rng.random() for r in frozen}
    worse |= {f"opus/en-ko/{r['id']}": 2 + rng.random() for r in frozen}
    assert contam(frozen, outputs, worse, ROUNDS)["en-ko"]["veto"] is True
    better = {f"hymt/en-ko/{r['id']}": 2 + rng.random() for r in frozen}
    better |= {f"opus/en-ko/{r['id']}": 2 + rng.random() for r in frozen}
    assert contam(frozen, outputs, better, ROUNDS)["en-ko"]["veto"] is False


def test_metricx_input_keys_each_system_direction_and_sentence():
    frozen = [{"id": "en-001", "lang": "en", "text": "Hi."}, {"id": "ko-001", "lang": "ko", "text": "안녕."}]
    outputs = {"hymt": {"en-ko": {"en-001": "안녕."}}, "opus": {"en-ko": {"en-001": "안녕하세요."}}}
    assert metricx_input(frozen, outputs) == [
        {"key": "hymt/en-ko/en-001", "src": "Hi.", "mt": "안녕."},
        {"key": "opus/en-ko/en-001", "src": "Hi.", "mt": "안녕하세요."},
    ]


def test_comet_input_uses_the_clean_source_for_typo_reports(tmp_path, monkeypatch):
    rows = {0: {"id": 0, "en": "Clean source.", "ko": "참조."}}
    monkeypatch.setattr(mt_decide, "load_typo_rows", lambda split: rows)
    typo_report = tmp_path / "typo.json"
    levels = {
        level: {"chrf": 0, "items": [{"id": 0, "input": "Clena sourse.", "hypothesis": "번역."}]}
        for level in ("clean", "light")
    }
    typo_report.write_text(json.dumps([{"direction": "en-ko", "levels": levels}]), encoding="utf-8")
    mt_report = tmp_path / "mt.json"
    item = {"id": 0, "source": "Clean source.", "reference": "참조.", "hypothesis": "번역."}
    mt_report.write_text(json.dumps([{"directions": {"en-ko": {"items": [item]}}}]), encoding="utf-8")
    segments = comet_input([typo_report, mt_report], "validation")
    assert segments == [
        {
            "key": comet_key("Clean source.", "번역.", "참조."),
            "src": "Clean source.",
            "mt": "번역.",
            "ref": "참조.",
        }
    ]
