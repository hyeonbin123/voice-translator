"""Decision rules of the Hy-MT2 trial (task T83, docs/experiments.md 15), applied to stored reports.

Nothing runs a translation model here: the inputs are eval.mt_eval and eval.typo_eval reports, COMET-22 scores
from eval/comet_score.py (its own environment) and MetricX-24 QE scores from eval/metricx_score.py. The
intervals are the paired bootstrap of eval/significance.py (10,000 resamples, seed 12345), as in section 13.

From backend/ with the eval group:
    uv run --no-sync python -m eval.mt_decide comet-input --reports R1.json R2.json \
        [--typo-split validation] --output ../work/vt-n1/comet_input.jsonl
    uv run --no-sync python -m eval.mt_decide validation --report eval/reports/mt_t83_dev_<stamp>.json \
        --comet ../work/vt-n1/comet_dev.json --tag t83_dev
    uv run --no-sync python -m eval.mt_decide typo --opus eval/reports/typo_t83_typo_c_<stamp>.json \
        --hymt eval/reports/typo_t83_typo_h_<stamp>.json --comet ../work/vt-n1/comet_typo.json --tag t83_typo
    uv run --no-sync python -m eval.mt_decide vram --validation eval/reports/mt_decide_t83_dev_<stamp>.json \
        --typo eval/reports/mt_decide_t83_typo_<stamp>.json
    uv run --no-sync python -m eval.mt_decide contam --set ../data/contam/v1.jsonl \
        --outputs ../work/vt-n1/contam_outputs.json --metricx ../work/vt-n1/contam_metricx.json \
        --tag t83_contam
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval.common import DATA, REPORTS
from eval.significance import ROUNDS, SEED, Interval, mean_difference

# The registered comparisons (docs/experiments.md 15). The baseline is the server's setting per direction.
BASELINE = {"ko-en": "opus-mt-tc-big-ko-en", "en-ko": "opus-mt-tc-big-en-ko/split"}
ARMS = {"ko-en": ["hy-mt2-q8"], "en-ko": ["hy-mt2-q8", "hy-mt2-q8/split"]}
FALLBACK = {"hy-mt2-q8": "hy-mt2-q4", "hy-mt2-q8/split": "hy-mt2-q4/split"}  # only after a latency-only miss
MAX_FLAGGED = 1  # (b) other scripts or explanatory outputs, of 129
MAX_EMPTY = 0
MAX_P50_S = 0.35  # (c)
MAX_P95_S = 0.8
MAX_NET_VRAM_MB = 512  # (d) +0.5 GB against what the pick replaces (MB = MiB, as elsewhere in the document)
# What opus takes on the GPU, from docs/experiments.md 2 (T3 validation; both loaded after the CUDA context
# existed, so the numbers are the models alone). The api process keeps its CUDA context for Whisper
# either way.
OPUS_VRAM_MB = {"en-ko": 512, "ko-en": 514}
TYPO_MARGIN = (
    0.5  # the corrector retires if Hy-MT2 alone reaches (corrector + opus) - 0.5 on the typo average
)


def comet_key(source: str, translation: str, reference: str) -> str:
    from eval.reanalysis import comet_key as key

    return key(source, translation, reference)


def _interval(interval: Interval) -> dict:
    return {
        "point": interval.point,
        "low": interval.low,
        "high": interval.high,
        "contains_zero": interval.contains(0.0),
    }


def load_typo_rows(split: str) -> dict[int, dict]:
    path = DATA / "typo" / f"{split}.jsonl"
    return {
        row["id"]: row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }


def _typo_triples(result: dict, rows: dict[int, dict]) -> list[tuple[str, str, str]]:
    """COMET's source is the clean original (its reference translates that), as in docs/experiments.md 13."""
    source, target = result["direction"].split("-")
    return [
        (rows[item["id"]][source], item["hypothesis"], rows[item["id"]][target])
        for level in result["levels"].values()
        for item in level["items"]
    ]


def comet_input(reports: list[Path], typo_split: str | None) -> list[dict]:
    """Unique (source, translation, reference) triples of mt_eval and typo_eval reports, first seen first."""
    rows = load_typo_rows(typo_split) if typo_split else None
    seen: dict[str, dict] = {}
    for path in reports:
        for result in json.loads(Path(path).read_text(encoding="utf-8")):
            if "levels" in result:
                if rows is None:
                    raise SystemExit(f"{path} is a typo_eval report: give --typo-split")
                triples = _typo_triples(result, rows)
            else:
                triples = [
                    (item["source"], item["hypothesis"], item["reference"])
                    for values in result["directions"].values()
                    for item in values["items"]
                ]
            for source, translation, reference in triples:
                key = comet_key(source, translation, reference)
                seen.setdefault(key, {"key": key, "src": source, "mt": translation, "ref": reference})
    return list(seen.values())


def _comet(comet: dict[str, float], items: list[dict]) -> np.ndarray:
    keys = [comet_key(item["source"], item["hypothesis"], item["reference"]) for item in items]
    missing = [key for key in keys if key not in comet]
    if missing:
        raise SystemExit(f"{len(missing)} segments have no COMET score")
    values = np.array([comet[key] for key in keys], dtype=float)
    if not np.isfinite(values).all():
        raise SystemExit("a COMET score is not a finite number")
    return values * 100


def compare(arm: dict, base: dict, comet: dict[str, float], rounds: int = ROUNDS) -> dict:
    """One arm against the baseline in one direction: rules (a), (b) and (c)."""
    from sacrebleu.metrics import CHRF

    from eval.significance import corpus_difference

    arm_items, base_items = arm["items"], base["items"]
    if [i["id"] for i in arm_items] != [i["id"] for i in base_items]:
        raise SystemExit("the two runs must hold the same sentences in the same order")
    references = [item["reference"] for item in base_items]
    chrf = corpus_difference(
        CHRF(),
        [([i["hypothesis"] for i in arm_items], [i["hypothesis"] for i in base_items], references)],
        rounds,
    )
    comet_diff = mean_difference([(_comet(comet, arm_items), _comet(comet, base_items))], rounds)
    checks = arm["checks"]
    rule_a = chrf.low > 0 and comet_diff.low > 0
    rule_b = checks["flagged"] <= MAX_FLAGGED and checks["empty"] <= MAX_EMPTY
    rule_c = arm["latency_p50"] <= MAX_P50_S and arm["latency_p95"] <= MAX_P95_S
    return {
        "chrf": {**_interval(chrf), "arm": arm["chrf"], "baseline": base["chrf"]},
        "comet": {
            **_interval(comet_diff),
            "arm": float(_comet(comet, arm_items).mean()),
            "baseline": float(_comet(comet, base_items).mean()),
        },
        "flagged": checks["flagged"],
        "empty": checks["empty"],
        "cut_at_cap": arm.get("cut_at_cap"),
        "latency_p50": arm["latency_p50"],
        "latency_p95": arm["latency_p95"],
        "baseline_latency_p50": base["latency_p50"],
        "rules": {"a": rule_a, "b": rule_b, "c": rule_c},
        "passes": rule_a and rule_b and rule_c,
        "latency_only_miss": rule_a and rule_b and not rule_c,
    }


def pick(arms: dict[str, dict]) -> str | None:
    """The passing arm with the higher chrF; on equal chrF (within 0.1) the faster one."""
    passing = [(name, r) for name, r in arms.items() if r["passes"]]
    if not passing:
        return None
    best = max(r["chrf"]["arm"] for _, r in passing)
    close = [(name, r) for name, r in passing if best - r["chrf"]["arm"] <= 0.1]
    return min(close, key=lambda pair: pair[1]["latency_p50"])[0]


def validation(results: list[dict], comet: dict[str, float], rounds: int = ROUNDS) -> dict:
    by_name = {result["model"]: result for result in results}
    out: dict = {"directions": {}}
    for direction, base_name in BASELINE.items():
        base = by_name[base_name]["directions"][direction]

        def measured(name: str, direction: str = direction) -> bool:
            # A Q4_K_M run may hold only the direction whose Q8_0 arm needed it.
            return name in by_name and direction in by_name[name]["directions"]

        arm_names = [n for n in ARMS[direction] + [FALLBACK[a] for a in ARMS[direction]] if measured(n)]
        arms = {
            name: compare(by_name[name]["directions"][direction], base, comet, rounds) for name in arm_names
        }
        for name, arm in arms.items():
            arm["vram_mb"] = by_name[name]["vram_mb"]
            arm["ollama"] = by_name[name].get("ollama")
        q8 = {name: arm for name, arm in arms.items() if "-q8" in name}
        fallback = sorted(FALLBACK[name] for name, arm in q8.items() if arm["latency_only_miss"])
        q4 = {name: arm for name, arm in arms.items() if name in fallback}
        # One pick over the Q8_0 arms and the Q4_K_M arms that stand in for a Q8_0 arm that missed only on
        # latency (section 15: the passing arm with the higher chrF; T85). Other Q4_K_M runs are not arms.
        chosen = pick({**q8, **q4})
        out["directions"][direction] = {
            "baseline": base_name,
            "arms": arms,
            "pick": chosen,
            # Q4_K_M is measured only for Q8_0 arms that passed (a) and (b) and missed only (c).
            "fallback_needed": [name for name in fallback if not measured(name)],
        }
    return out


def typo(opus: dict, hymt: dict, rows: dict[int, dict], comet: dict[str, float] | None, rounds: int = ROUNDS):
    """Hy-MT2 alone (candidate A with the Hy-MT2 translator) against the corrector with opus (candidate C)."""
    from sacrebleu.metrics import CHRF

    from eval.significance import corpus_difference

    source, target = opus["direction"].split("-")
    references = [rows[item["id"]][target] for item in opus["levels"]["light"]["items"]]

    def hyps(result: dict, level: str) -> list[str]:
        return [item["hypothesis"] for item in result["levels"][level]["items"]]

    for result in (opus, hymt):
        for level in ("light", "heavy"):
            if [i["id"] for i in result["levels"][level]["items"]] != [
                i["id"] for i in opus["levels"]["light"]["items"]
            ]:
                raise SystemExit("the two runs must hold the same sentences in the same order")
    groups = [(hyps(hymt, level), hyps(opus, level), references) for level in ("light", "heavy")]
    noisy = corpus_difference(CHRF(), groups, rounds)
    clean = corpus_difference(CHRF(), [(hyps(hymt, "clean"), hyps(opus, "clean"), references)], rounds)

    def average(result: dict) -> float:
        return (result["levels"]["light"]["chrf"] + result["levels"]["heavy"]["chrf"]) / 2

    out = {
        "direction": opus["direction"],
        "opus": {
            "candidate": opus["candidate"],
            "corrector": opus["corrector"],
            "translator": opus["translator"],
        },
        "hymt": {
            "candidate": hymt["candidate"],
            "corrector": hymt["corrector"],
            "translator": hymt["translator"],
        },
        "typo_average": {"hymt": average(hymt), "opus_corrected": average(opus), **_interval(noisy)},
        "clean": {
            "hymt": hymt["levels"]["clean"]["chrf"],
            "opus_corrected": opus["levels"]["clean"]["chrf"],
            **_interval(clean),
        },
        "corrector_vram_mb": opus["vram_mb"],
        "corrector_ollama": opus.get("ollama"),
        "retire_corrector": average(hymt) >= average(opus) - TYPO_MARGIN,
    }
    if comet is not None:

        def scores(result: dict, level: str) -> np.ndarray:
            items = [
                {
                    "source": rows[i["id"]][source],
                    "hypothesis": i["hypothesis"],
                    "reference": rows[i["id"]][target],
                }
                for i in result["levels"][level]["items"]
            ]
            return _comet(comet, items)

        out["comet_typo_average"] = _interval(
            mean_difference(
                [(scores(hymt, level), scores(opus, level)) for level in ("light", "heavy")], rounds
            )
        )
    return out


def vram(decision: dict, typo_decision: dict | None) -> dict:
    """(d): what the picked Hy-MT2 builds take against what they replace. A build (hy-mt2-q8, hy-mt2-q4; the
    whole and sentence-by-sentence arms are one Ollama model) counts once even when it serves both
    directions, at the larger of its readings; two different builds are both loaded and both count (T85)."""
    picks = {d: v["pick"] for d, v in decision["directions"].items() if v["pick"]}
    if not picks:
        return {"picks": {}, "passes": None}
    builds: dict[str, int] = {}
    for d, name in picks.items():
        build = name.split("/", 1)[0]
        builds[build] = max(builds.get(build, 0), decision["directions"][d]["arms"][name]["vram_mb"])
    hymt_mb = sum(builds.values())
    replaced = {f"opus {d}": OPUS_VRAM_MB[d] for d in picks}
    retire = bool(typo_decision and typo_decision["retire_corrector"] and "en-ko" in picks)
    if retire:
        replaced["corrector"] = typo_decision["corrector_vram_mb"]
    net = hymt_mb - sum(replaced.values())
    return {
        "picks": picks,
        "builds_mb": builds,
        "hymt_vram_mb": hymt_mb,
        "replaced_mb": replaced,
        "corrector_retired": retire,
        "net_mb": net,
        "passes": net <= MAX_NET_VRAM_MB,
    }


def contam(frozen: list[dict], outputs: dict, metricx: dict[str, float], rounds: int = ROUNDS) -> dict:
    """MetricX-24 QE (an error score, lower is better) of Hy-MT2 minus opus on the frozen set, per direction.

    A direction is vetoed when the whole interval is above 0 (opus better) there; this check can only veto.
    """
    out = {}
    for direction, language in (("en-ko", "en"), ("ko-en", "ko")):
        ids = [row["id"] for row in frozen if row["lang"] == language]
        if not ids or direction not in outputs.get("hymt", {}):
            continue
        hy = np.array([metricx[f"hymt/{direction}/{i}"] for i in ids], dtype=float)
        op = np.array([metricx[f"opus/{direction}/{i}"] for i in ids], dtype=float)
        if not (np.isfinite(hy).all() and np.isfinite(op).all()):
            raise SystemExit("a MetricX score is not a finite number")
        interval = mean_difference([(hy, op)], rounds)
        out[direction] = {
            "sentences": len(ids),
            "hymt": float(hy.mean()),
            "opus": float(op.mean()),
            **_interval(interval),
            "veto": interval.low > 0,
        }
    return out


def metricx_input(frozen: list[dict], outputs: dict) -> list[dict]:
    """One QE segment per system, direction and sentence, keyed system/direction/id."""
    rows = []
    for system, by_direction in outputs.items():
        for direction, translations in by_direction.items():
            language = direction.split("-")[0]
            for row in frozen:
                if row["lang"] == language:
                    rows.append(
                        {
                            "key": f"{system}/{direction}/{row['id']}",
                            "src": row["text"],
                            "mt": translations[row["id"]],
                        }
                    )
    return rows


def _write(name: str, tag: str, result: dict, lines: list[str]) -> Path:
    stamp = datetime.now(UTC)
    result = {"created": stamp.isoformat(timespec="seconds"), "rounds": ROUNDS, "seed": SEED, **result}
    base = REPORTS / f"mt_decide_{tag}_{stamp:%Y%m%d_%H%M%S}"
    base.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    base.with_suffix(".md").write_text(
        "\n".join([f"# T83 {name} ({tag})", "", *lines]) + "\n", encoding="utf-8"
    )
    print(f"written to {base.with_suffix('.md')}")
    return base


def _fmt(interval: dict) -> str:
    return f"{interval['point']:+.2f} [{interval['low']:+.2f}, {interval['high']:+.2f}]"


def validation_lines(result: dict) -> list[str]:
    lines = [
        "| 방향 | 팔 | chrF (기준) | chrF 차이 [95%] | COMET-22 차이 [95%] | 다른 문자·설명형 / 빈 | "
        "p50 / p95 (s) | VRAM (MB) | (a) | (b) | (c) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for direction, values in result["directions"].items():
        for name, arm in values["arms"].items():
            rules = arm["rules"]
            lines.append(
                f"| {direction} | {name} | {arm['chrf']['arm']:.1f} ({arm['chrf']['baseline']:.1f}) "
                f"| {_fmt(arm['chrf'])} | {_fmt(arm['comet'])} | {arm['flagged']} / {arm['empty']} "
                f"| {arm['latency_p50']:.3f} / {arm['latency_p95']:.3f} | {arm['vram_mb']} | "
                + " | ".join("O" if rules[r] else "X" for r in "abc")
                + " |"
            )
    lines.append("")
    for direction, values in result["directions"].items():
        lines.append(
            f"- {direction}: 고른 팔 {values['pick'] or '없음'} (기준 {values['baseline']}), "
            f"Q4 대체 팔이 필요한 것: {', '.join(values['fallback_needed']) or '없음'}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("comet-input")
    c.add_argument("--reports", nargs="+", type=Path, required=True)
    c.add_argument("--typo-split", choices=("validation", "test"))
    c.add_argument("--output", type=Path, required=True)
    v = commands.add_parser("validation")
    v.add_argument("--report", type=Path, nargs="+", required=True, help="mt_eval reports (Q8 run, Q4 run)")
    v.add_argument("--comet", type=Path, required=True)
    v.add_argument("--tag", default="t83_dev")
    t = commands.add_parser("typo")
    t.add_argument("--opus", type=Path, required=True, help="typo_eval report with candidate C on opus")
    t.add_argument("--hymt", type=Path, required=True, help="typo_eval report with candidate A on Hy-MT2")
    t.add_argument("--comet", type=Path)
    t.add_argument("--split", choices=("validation", "test"), default="validation")
    t.add_argument("--tag", default="t83_typo")
    m = commands.add_parser("vram")
    m.add_argument("--validation", type=Path, required=True)
    m.add_argument("--typo", type=Path)
    x = commands.add_parser("metricx-input")
    x.add_argument("--set", type=Path, required=True)
    x.add_argument("--outputs", type=Path, required=True)
    x.add_argument("--output", type=Path, required=True)
    k = commands.add_parser("contam")
    k.add_argument("--set", type=Path, required=True)
    k.add_argument("--outputs", type=Path, required=True)
    k.add_argument("--metricx", type=Path, required=True)
    k.add_argument("--tag", default="t83_contam")
    args = parser.parse_args()

    def read(path: Path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def frozen(path: Path) -> list[dict]:
        return [
            json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    if args.command == "comet-input":
        rows = comet_input(args.reports, args.typo_split)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        print(f"{len(rows)} segments -> {args.output}")
    elif args.command == "validation":
        results = [result for path in args.report for result in read(path)]
        comet = read(args.comet)
        result = validation(results, comet["scores"])
        result["reports"] = [str(path) for path in args.report]
        result["comet_meta"] = comet["meta"]
        _write("validation", args.tag, result, validation_lines(result))
    elif args.command == "typo":
        rows = load_typo_rows(args.split)
        opus = next(r for r in read(args.opus) if r["direction"] == "en-ko" and r["candidate"] == "C")
        hymt = next(r for r in read(args.hymt) if r["direction"] == "en-ko" and r["candidate"] == "A")
        comet = read(args.comet)["scores"] if args.comet else None
        result = typo(opus, hymt, rows, comet)
        lines = [
            f"- 오타 평균 chrF: Hy-MT2 {result['typo_average']['hymt']:.2f}, 교정기+opus "
            f"{result['typo_average']['opus_corrected']:.2f}, 차이 {_fmt(result['typo_average'])}",
            f"- 깨끗한 입력 chrF: Hy-MT2 {result['clean']['hymt']:.2f}, 교정기+opus "
            f"{result['clean']['opus_corrected']:.2f}, 차이 {_fmt(result['clean'])}",
            f"- 교정기 은퇴(차이 ≥ −{TYPO_MARGIN}): {result['retire_corrector']}",
        ]
        _write("typo", args.tag, result, lines)
    elif args.command == "vram":
        typo_decision = read(args.typo) if args.typo else None
        print(json.dumps(vram(read(args.validation), typo_decision), ensure_ascii=False, indent=1))
    elif args.command == "metricx-input":
        rows = metricx_input(frozen(args.set), read(args.outputs))
        args.output.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        print(f"{len(rows)} segments -> {args.output}")
    else:
        metricx = read(args.metricx)
        result = {"contamination": contam(frozen(args.set), read(args.outputs), metricx["scores"])}
        result["metricx_meta"] = metricx["meta"]
        lines = [
            f"- {d}: MetricX QE Hy-MT2 {r['hymt']:.2f}, opus {r['opus']:.2f}, 차이 {_fmt(r)}, "
            f"거부권 {r['veto']}"
            for d, r in result["contamination"].items()
        ]
        _write("contamination", args.tag, result, lines)


if __name__ == "__main__":
    main()
