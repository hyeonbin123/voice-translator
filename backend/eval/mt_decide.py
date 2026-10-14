"""저장된 리포트에 Hy-MT2 시험의 판단 규칙(docs/experiments.md 15절)을 적용한다.

여기서는 번역 모델을 돌리지 않는다: 입력은 eval.mt_eval과 eval.typo_eval의 리포트, eval/comet_score.py(별도
환경)의 COMET-22 점수, eval/metricx_score.py의 MetricX-24 QE 점수다. 구간은 13절과 같이
eval/significance.py의 짝지은 붓스트랩(10,000번 재표본, 시드 12345)이다.

backend/에서 eval 그룹으로:
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

# [파일 역할] Hy-MT2 번역 시험의 판단 규칙(docs/experiments.md 15절)을 저장된 측정 결과에 그대로 적용하는
# 도구. 번역 모델은 돌리지 않는다.
# [설계 이유] 측정 전에 적은 규칙을 코드로 고정해, 같은 입력이면 언제든 같은 판정을 다시 낼 수 있게 한다.
# 구간은 재분석(13절)과 같은 짝지은 붓스트랩(eval/significance.py, 10,000번, 시드 12345)을 쓴다.
# [주요 구성] comet-input(COMET 채점할 문장 모으기) → validation(규칙 (a) 품질·(b) 출력·(c) 지연과 팔 고르기)
# → typo(영어 오타 교정기를 은퇴시킬지) → vram(규칙 (d) GPU 메모리) → metricx-input·contam(오염 거부권).
# [연결] 입력은 eval/mt_eval.py·eval/typo_eval.py 리포트, eval/comet_score.py의 COMET-22 점수,
# eval/metricx_score.py의 MetricX 점수, eval/contam_set.py의 번역 결과. 결과는
# eval/reports/mt_decide_<태그>_<시각>.json·.md로 남는다. 테스트: tests/test_mt_decide.py.

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

# numpy: 문장 점수 배열. eval.significance: 붓스트랩 반복 수·시드, 구간 타입, 평균 차이의 짝지은 구간.
# sacrebleu(chrF)는 쓰는 함수 안에서 import한다.
import numpy as np

from eval.common import DATA, REPORTS
from eval.significance import ROUNDS, SEED, Interval, mean_difference

# 미리 등록한 비교(docs/experiments.md 15절).
# BASELINE: 방향별 기준(서버의 현재 설정: 한→영 통째로, 영→한 문장 단위). ARMS: 시험 팔(Q8_0; 영→한은 통째로와
# 문장 단위 둘). FALLBACK: Q8_0 팔이 지연에서만 떨어졌을 때 대신 재는 Q4_K_M 팔.
BASELINE = {"ko-en": "opus-mt-tc-big-ko-en", "en-ko": "opus-mt-tc-big-en-ko/split"}
ARMS = {"ko-en": ["hy-mt2-q8"], "en-ko": ["hy-mt2-q8", "hy-mt2-q8/split"]}
FALLBACK = {"hy-mt2-q8": "hy-mt2-q4", "hy-mt2-q8/split": "hy-mt2-q4/split"}
# 규칙의 문턱: (b) 다른 문자·설명형 출력 129개 중 1개 이하·빈 출력 0개, (c) 지연 p50 ≤ 0.35초·p95 ≤ 0.8초, (d)
# 고른 빌드의 메모리 − 대신하는 것 ≤ +512MB.
MAX_FLAGGED = 1  # (b) 129개 중 다른 문자·설명형 출력
MAX_EMPTY = 0
MAX_P50_S = 0.35  # (c)
MAX_P95_S = 0.8
MAX_NET_VRAM_MB = 512  # (d) 고른 것이 대신하는 것보다 +0.5GB까지(문서의 다른 곳처럼 MB = MiB)
# OPUS_VRAM_MB: Hy-MT2가 대신하는 opus 모델이 GPU에서 차지하는 메모리(MB, docs/experiments.md 2절 validation
# 표의 값). 둘 다 CUDA 컨텍스트가 이미 있을 때 올려 잰 값이라 모델만의 크기다. api 프로세스는 어느 쪽이든
# Whisper를 위해 CUDA 컨텍스트를 그대로 가진다.
OPUS_VRAM_MB = {"en-ko": 512, "ko-en": 514}
TYPO_MARGIN = (
    0.5  # Hy-MT2 혼자의 오타 평균 chrF가 (교정기 + opus) − 0.5 이상이면 교정기를 은퇴시킨다(15절 판단 규칙)
)


# COMET 점수를 찾는 키. 재분석(eval/reanalysis.py)과 같은 함수를 써서 두 도구의 키가 같게 한다.
def comet_key(source: str, translation: str, reference: str) -> str:
    from eval.reanalysis import comet_key as key

    return key(source, translation, reference)


# 구간 객체를 리포트용 dict(점 추정, 하한, 상한, 0을 포함하는지)로 바꾼다.
def _interval(interval: Interval) -> dict:
    return {
        "point": interval.point,
        "low": interval.low,
        "high": interval.high,
        "contains_zero": interval.contains(0.0),
    }


# 오타 세트(data/typo/<분할>.jsonl)를 id → 행(깨끗한 원문, 참조 등)으로 읽는다. 오타 리포트에는 오타 입력만
# 있어 깨끗한 원문·참조를 여기서 가져온다.
def load_typo_rows(split: str) -> dict[int, dict]:
    path = DATA / "typo" / f"{split}.jsonl"
    return {
        row["id"]: row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }


def _typo_triples(result: dict, rows: dict[int, dict]) -> list[tuple[str, str, str]]:
    """오타 리포트의 (원문, 번역, 참조) 세 쌍. COMET의 원문은 오타 입력이 아니라 깨끗한 원문이다.

    참조가 깨끗한 원문의 번역이기 때문이다(docs/experiments.md 13절과 같게).
    """
    source, target = result["direction"].split("-")
    return [
        (rows[item["id"]][source], item["hypothesis"], rows[item["id"]][target])
        for level in result["levels"].values()
        for item in level["items"]
    ]


def comet_input(reports: list[Path], typo_split: str | None) -> list[dict]:
    """mt_eval·typo_eval 리포트에서 겹치지 않는 (원문, 번역, 참조) 세 쌍을 처음 나온 순서대로 모은다.

    COMET으로 채점할 문장이다. 같은 세 쌍은 키가 같으므로 한 번만 넣어 채점 시간을 아낀다. 오타
    리포트(levels가 있음)는 --typo-split으로 원문 세트를 알아야 한다.
    """
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


# 항목들의 COMET 점수 배열(0~100 눈금으로 ×100). 점수가 빠졌거나 유한하지 않으면 판정을 내지 않고 멈춘다.
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
    """한 방향에서 한 팔을 기준과 견준다: 규칙 (a), (b), (c).

    두 실행이 같은 문장을 같은 순서로 가졌는지 먼저 확인한다(짝지은 비교의 전제). chrF 차이는 문장 통계를
    다시 더하는 corpus 점수의 짝지은 구간, COMET 차이는 문장 점수 평균의 짝지은 구간이다. (a) 두 구간의
    하한이 모두 0보다 큼(두 지표 모두 유의하게 나음), (b) 출력 점검, (c) 지연. latency_only_miss는 (a)·(b)는
    통과하고 (c)에서만 떨어진 경우로, Q4_K_M 대체 팔을 잴지 정한다.
    """
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
    """모든 규칙을 통과한 팔 중 chrF가 더 높은 팔. chrF가 같으면(0.1점 안) 더 빠른 팔.

    가장 높은 값과 0.1점 안인 팔들 중에서는 지연 p50이 짧은 팔을 고른다. 통과한 팔이 없으면 None(opus
    유지).
    """
    passing = [(name, r) for name, r in arms.items() if r["passes"]]
    if not passing:
        return None
    best = max(r["chrf"]["arm"] for _, r in passing)
    close = [(name, r) for name, r in passing if best - r["chrf"]["arm"] <= 0.1]
    return min(close, key=lambda pair: pair[1]["latency_p50"])[0]


# 방향마다: 기준 결과를 찾고, 측정된 팔(Q8_0 팔과, 리포트에 있으면 Q4_K_M 팔)을 모두 비교한 뒤 팔을 고른다.
# 고르는 대상은 Q8_0 팔과, 지연에서만 떨어진 Q8_0 팔을 대신하는 Q4_K_M 팔뿐이다(그 밖의 Q4_K_M 결과는 팔이
# 아님). 아직 재지 않은 필요한 대체 팔은 fallback_needed로 알려 준다.
# measured의 기본 인수(direction=direction)는 반복문 변수의 현재 값을 붙잡는다.
def validation(results: list[dict], comet: dict[str, float], rounds: int = ROUNDS) -> dict:
    by_name = {result["model"]: result for result in results}
    out: dict = {"directions": {}}
    for direction, base_name in BASELINE.items():
        base = by_name[base_name]["directions"][direction]

        def measured(name: str, direction: str = direction) -> bool:
            # Q4_K_M 실행에는 Q8_0 팔이 대체 팔을 필요로 한 방향만 들어 있을 수 있다.
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
        # Q8_0 팔들과, 지연에서만 떨어진 Q8_0 팔을 대신하는 Q4_K_M 팔들 가운데서 한 번에 고른다(15절: 통과한
        # 팔 중 chrF가 높은 팔). 그 밖의 Q4_K_M 실행은 팔이 아니다.
        chosen = pick({**q8, **q4})
        out["directions"][direction] = {
            "baseline": base_name,
            "arms": arms,
            "pick": chosen,
            # Q4_K_M은 (a)·(b)를 통과하고 (c)에서만 떨어진 Q8_0 팔에 대해서만 잰다.
            "fallback_needed": [name for name in fallback if not measured(name)],
        }
    return out


def typo(opus: dict, hymt: dict, rows: dict[int, dict], comet: dict[str, float] | None, rounds: int = ROUNDS):
    """Hy-MT2 혼자(Hy-MT2 번역기를 쓴 후보 A)를 지금 설정인 교정기 + opus(후보 C)와 견준다.

    같은 오타 세트에서 영→한을 비교한다. 두 리포트의 모든 수준(가벼움·심함·깨끗)이 같은 문장 순서인지 먼저
    확인한다. 판정(retire_corrector)은 점 추정 규칙(오타 평균 chrF 차이 ≥ −0.5)이고, 오타 평균·깨끗한
    입력의 chrF 차이 구간과 COMET 차이 구간은 함께 적기만 한다(15절).
    """
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
    """(d): 고른 Hy-MT2 빌드가 차지하는 GPU 메모리를 그것이 대신하는 것과 견준다.

    빌드(hy-mt2-q8, hy-mt2-q4. 통째로 팔과 문장 단위 팔은 같은 Ollama 모델이다)는 두 방향을 모두 맡아도
    한 번만, 읽은 값 중 큰 값으로 센다. 서로 다른 두 빌드는 둘 다 올라가므로 둘 다 센다. 대신하는 것(바꾼
    방향의 opus, 교정기를 은퇴시키면 교정기)을 빼서 +512MB 이하인지 본다. 고른 팔이 없으면 판정하지
    않는다(passes None). 교정기는 영→한을 바꿀 때만 은퇴할 수 있다.
    """
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
    """오염 거부권: 고정 세트에서 방향마다 Hy-MT2 − opus의 MetricX-24 QE 차이(오류 점수, 낮을수록 좋음).

    방향마다 평균 차이와 짝지은 구간을 낸다(영어 문장 → 영→한, 한국어 문장 → 한→영). 구간 전체가 0보다
    크면(opus가 뚜렷이 나음) 그 방향을 거부한다. 이 점검은 거부만 할 수 있다. Hy-MT2가 번역하지 않은 방향은
    건너뛴다.
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
    """시스템·방향·문장마다 QE 문장 하나. 키는 "시스템/방향/id"다.

    eval/contam_set.py가 쓴 번역 결과에서 MetricX 입력 행을 만든다. contam이 같은 키로 점수를 찾는다.
    """
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


# 판정 결과를 만든 시각과 붓스트랩 설정(반복 수·시드)과 함께 JSON으로, 사람이 읽을 요약을 마크다운으로 남긴다.
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


# 구간을 "+점 [하한, 상한]" 형식의 문자열로.
def _fmt(interval: dict) -> str:
    return f"{interval['point']:+.2f} [{interval['low']:+.2f}, {interval['high']:+.2f}]"


# validation 판정의 마크다운 표: 팔마다 chrF·차이 구간·COMET 차이 구간·출력 점검·지연·VRAM과 규칙 (a)(b)(c)
# 통과 여부(O/X), 끝에 방향별로 고른 팔과 필요한 대체 팔.
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


# 하위 명령: comet-input / validation / typo / vram / metricx-input / contam. 각 명령은 위 함수에 저장된
# 리포트를 넣고 결과를 쓰거나(validation·typo·contam) 출력한다(vram·입력 만들기).
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
    v.add_argument("--report", type=Path, nargs="+", required=True, help="mt_eval 리포트(Q8 실행, Q4 실행)")
    v.add_argument("--comet", type=Path, required=True)
    v.add_argument("--tag", default="t83_dev")
    t = commands.add_parser("typo")
    t.add_argument("--opus", type=Path, required=True, help="opus로 후보 C를 잰 typo_eval 리포트")
    t.add_argument("--hymt", type=Path, required=True, help="Hy-MT2로 후보 A를 잰 typo_eval 리포트")
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
        # 오타 리포트에서 영→한의 후보 C(교정기 + opus)와 후보 A(Hy-MT2 혼자) 결과만 꺼내 비교한다.
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
