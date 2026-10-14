"""T83 판단 규칙(docs/experiments.md 15절)을 지어낸 보고서로 검증한다.

sacrebleu(eval 그룹)가 필요하다. CI의 일반 테스트 단계는 이 모듈을 건너뛰고, sacrebleu를 더한 별도 단계가
이 모듈을 돌린다.
"""

# [파일 역할] Hy-MT2 번역기 시험의 판단 규칙을 저장된 결과에 적용하는 eval/mt_decide.py를 지어낸 보고서로
# 검증한다(docs/experiments.md 15절). 번역 모델은 돌리지 않는다.
# [규칙 요약] 방향마다 (a) 품질: chrF와 COMET-22 차이의 95% 짝지은 구간 하한이 모두 0 초과, (b) 출력: 다른
# 문자·설명형 출력 1개 이하·빈 출력 0, (c) 지연: p50 ≤ 0.35초·p95 ≤ 0.8초. 통과한 팔 중 chrF가 높은 팔(0.1점
# 안이면 빠른 팔)을 고르고, Q8 팔이 (c)에서만 떨어졌으면 그 Q4_K_M을 대체 팔로 잰다. 그 밖에 오타 교정기 은퇴,
# (d) GPU 메모리, 오염 거부권 판정이 있다.
# [왜 중요한가] 측정 전에 적은 규칙을 코드로 고정해 같은 입력이면 늘 같은 판정이 나오게 한 도구라, 경계값과 팔
# 고르기 순서가 문서와 같아야 한다.

import json

import numpy as np
import pytest

# chrF 계산에 sacrebleu가 필요해 없으면 파일 전체를 건너뛴다. CI는 sacrebleu를 얹은 별도 단계에서 이 파일을
# 돌린다(.github/workflows/ci.yml). 그래서 아래 import들은 이 줄 뒤에 E402 경고를 끄는 표시와 함께 있다.
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

# 테스트 속도를 위해 붓스트랩 반복을 실제(10,000번)보다 줄인다. WORDS는 서로 다른 시험 문장을 만들 단어
# 목록이다.
ROUNDS = 300
WORDS = "the river runs past the old mill and the children wave at the boats every summer morning".split()


# 단어 목록에서 길이가 다른 문장 count개를 만든다(끝의 번호로 모두 다른 문장이 된다).
def sentences(count: int) -> list[str]:
    return [" ".join(WORDS[i % 7 : i % 7 + 6 + i % 4]) + f" {i}." for i in range(count)]


# eval.mt_eval 보고서의 한 방향 결과 모양(chrF, 지연 p50·p95, 출력 점검 결과, 문장별 항목)을 만든다. 출력
# 점검은 실제 check_items로 계산해 규칙 (b)가 실제 점검 결과를 쓰게 한다.
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


# 한 결과의 모든 문장에 같은 COMET 점수를 주는 점수표(키는 원문·번역문·참조로 만든 comet_key).
def comet_for(values: dict, score: float) -> dict[str, float]:
    return {comet_key(i["source"], i["hypothesis"], i["reference"]): score for i in values["items"]}


# 참조에서 끝 두 단어를 뺀 번역. chrF가 낮은 '기준(opus) 쪽'을 흉내 낸다.
def degraded(references):
    return [" ".join(r.split()[:-2]) for r in references]  # 단어를 빼서 chrF를 낮춘다


# 참조와 똑같은 후보가 chrF·COMET 모두 기준보다 높고(구간 하한 > 0) 출력·지연도 기준 안이면 (a)(b)(c) 모두
# 통과하는지 확인한다.
def test_an_arm_better_on_both_metrics_and_inside_the_budgets_passes():
    refs = sentences(40)
    arm, base = direction_values(refs, refs), direction_values(degraded(refs), refs)
    comet = {**comet_for(arm, 0.9), **comet_for(base, 0.7)}
    result = compare(arm, base, comet, ROUNDS)
    assert result["chrf"]["low"] > 0 and result["comet"]["low"] > 0
    assert result["rules"] == {"a": True, "b": True, "c": True} and result["passes"]


# chrF는 높아도 COMET이 같으면(차이 0) (a)에서 떨어진다. 두 지표가 모두 유의하게 나아야 한다는 규칙이다.
def test_rule_a_needs_both_intervals_above_zero():
    refs = sentences(40)
    arm, base = direction_values(refs, refs), direction_values(degraded(refs), refs)
    comet = {**comet_for(arm, 0.7), **comet_for(base, 0.7)}  # COMET으로는 차이가 없다
    assert compare(arm, base, comet, ROUNDS)["rules"]["a"] is False


# 다른 문자(한자 '東京')가 섞인 출력 1개는 허용, 2개는 실패, 빈 출력은 1개만 있어도 실패하는지
# 확인한다(MAX_FLAGGED 1, MAX_EMPTY 0).
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


# 지연 한도의 경계: p50 0.35초·p95 0.8초는 통과(이하), p50 0.36초나 p95 0.81초는 실패. (a)(b)는 통과하고
# (c)에서만 떨어졌을 때 latency_only_miss가 켜져 Q4 대체 팔을 잴 근거가 된다.
@pytest.mark.parametrize(("p50", "p95", "ok"), [(0.35, 0.8, True), (0.36, 0.5, False), (0.2, 0.81, False)])
def test_rule_c_latency_limits(p50, p95, ok):
    refs = sentences(30)
    arm, base = direction_values(refs, refs, p50, p95), direction_values(degraded(refs), refs)
    comet = {**comet_for(arm, 0.9), **comet_for(base, 0.7)}
    result = compare(arm, base, comet, ROUNDS)
    assert result["rules"]["c"] is ok
    assert result["latency_only_miss"] is (not ok)


# eval.mt_eval 보고서의 모델 하나(이름, GPU 메모리, 방향별 결과) 모양을 만든다.
def report(name, direction, values, vram_mb=2400):
    return {"model": name, "vram_mb": vram_mb, "directions": {direction: values}}


# 방향마다 따로 고르는지 확인한다. 영→한: 통째로 번역하는 Q8 팔은 품질은 통과하지만 느려(p50 0.5초) 떨어지고,
# 문장 단위 Q8 팔이 뽑힌다. 통째로 팔이 지연에서만 떨어졌으므로 그 Q4(hy-mt2-q4)를 재라고 알린다. 한→영:
# 후보가 opus보다 낫지 않아 고른 팔이 없고, 지연 탓이 아니므로 Q4도 요청하지 않는다.
def test_validation_picks_per_direction_and_asks_for_q4_only_after_a_latency_only_miss():
    refs = sentences(40)
    base_en = direction_values(degraded(refs), refs)
    base_ko = direction_values(degraded(refs), refs)
    whole = direction_values(refs, refs, p50=0.5)  # 품질은 통과, 너무 느림
    split = direction_values(refs, refs, p50=0.3)
    ko = direction_values(degraded(refs), refs)  # opus보다 낫지 않음
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


# 회귀 테스트(커밋 22fe56c): 예전에는 Q8 팔 하나가 통과하면 Q4 대체 팔을 보지 않았다. 규칙대로 Q4 팔도 같은
# 규칙으로 함께 겨뤄, chrF가 더 높은 Q4(느렸던 통째로 팔과 같은 품질, 충분히 빠름)가 통과한 문장 단위 Q8(한
# 단어 짧아 chrF 낮음)을 이기는지 확인한다. 한→영은 Q8이 지연 때문에 떨어진 것이 아니라 Q4 결과가 있어도
# 후보가 아니다.
def test_a_q4_arm_measured_after_a_latency_only_miss_competes_with_the_passing_q8_arms():
    refs = sentences(40)
    base = direction_values(degraded(refs), refs)
    one_word_short = [" ".join(r.split()[:-1]) for r in refs]
    q8_whole = direction_values(refs, refs, p50=0.5)  # 품질 최고, 너무 느림
    q8_split = direction_values(one_word_short, refs, p50=0.3)  # 통과, chrF 낮음
    q4_whole = direction_values(refs, refs, p50=0.3)  # 느렸던 팔의 Q4: 같은 품질, 충분히 빠름
    base_ko = direction_values(degraded(refs), refs)
    ko = direction_values(degraded(refs), refs)  # opus보다 낫지 않음
    results = [
        report("opus-mt-tc-big-en-ko/split", "en-ko", base, 512),
        report("opus-mt-tc-big-ko-en", "ko-en", base_ko, 514),
        {"model": "hy-mt2-q8", "vram_mb": 2400, "directions": {"en-ko": q8_whole, "ko-en": ko}},
        report("hy-mt2-q8/split", "en-ko", q8_split),
        {"model": "hy-mt2-q4", "vram_mb": 1400, "directions": {"en-ko": q4_whole, "ko-en": ko}},
    ]
    comet = {}
    for values, score in ((base, 0.7), (base_ko, 0.7), (ko, 0.7), (q8_whole, 0.9), (q8_split, 0.9)):
        comet.update(comet_for(values, score))
    comet.update(comet_for(q4_whole, 0.9))
    decision = validation(results, comet, ROUNDS)["directions"]
    result = decision["en-ko"]
    assert result["arms"]["hy-mt2-q8/split"]["passes"] and result["arms"]["hy-mt2-q4"]["passes"]
    assert result["arms"]["hy-mt2-q8/split"]["chrf"]["arm"] < result["arms"]["hy-mt2-q4"]["chrf"]["arm"]
    assert result["pick"] == "hy-mt2-q4"
    assert decision["ko-en"]["pick"] is None


# Q4는 지연에서만 떨어진 방향에서만 재므로 보고서에 한 방향만 있을 수 있다. 그 방향에서는 Q4가 뽑히고 더
# 요청할 대체 팔이 없으며, 다른 방향의 팔 목록에는 Q4가 끼지 않는지 확인한다(한 방향만 있는 Q4 보고서에서
# KeyError가 났던 경우).
def test_a_q4_arm_measured_for_one_direction_leaves_the_other_alone():
    refs = sentences(40)
    base, base_ko = direction_values(degraded(refs), refs), direction_values(degraded(refs), refs)
    slow, ko = direction_values(refs, refs, p50=0.5), direction_values(degraded(refs), refs)
    q4 = direction_values(refs, refs, p50=0.3)
    results = [
        report("opus-mt-tc-big-en-ko/split", "en-ko", base, 512),
        report("opus-mt-tc-big-ko-en", "ko-en", base_ko, 514),
        {"model": "hy-mt2-q8", "vram_mb": 2400, "directions": {"en-ko": slow, "ko-en": ko}},
        report("hy-mt2-q4", "en-ko", q4, 1400),
    ]
    comet = {}
    for values, score in ((base, 0.7), (base_ko, 0.7), (ko, 0.7), (slow, 0.9), (q4, 0.9)):
        comet.update(comet_for(values, score))
    decision = validation(results, comet, ROUNDS)["directions"]
    assert decision["en-ko"]["pick"] == "hy-mt2-q4" and decision["en-ko"]["fallback_needed"] == []
    assert decision["ko-en"]["pick"] is None and "hy-mt2-q4" not in decision["ko-en"]["arms"]


# 팔 고르기: chrF가 1점 높으면 높은 팔, 0.1점 안(0.05 차이)이면 p50이 짧은 팔, 통과한 팔이 없으면 None.
def test_pick_prefers_higher_chrf_then_the_faster_arm():
    def arm(chrf, p50, passes=True):
        return {"passes": passes, "chrf": {"arm": chrf}, "latency_p50": p50}

    assert pick({"a": arm(40.0, 0.3), "b": arm(41.0, 0.3)}) == "b"
    assert pick({"a": arm(40.0, 0.3), "b": arm(40.05, 0.32)}) == "a"
    assert pick({"a": arm(40.0, 0.3, passes=False)}) is None


# 규칙 (d): 고른 Hy-MT2의 메모리 − 그것이 대신하는 것(opus 영→한 512MB, 한→영 514MB, 은퇴시킨 교정기) ≤
# +512MB인지 계산한다. 교정기를 유지하면 2200 − 512로 넘치고, 은퇴시키면 통과하며, 정확히 +512MB도
# 통과(이하)다.
# 두 방향이 같은 Ollama 모델(통째로·문장 단위 팔은 같은 빌드)이면 메모리를 한 번만 세고, 측정값이 다르면 큰
# 쪽을 쓴다.
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
    assert edge["net_mb"] == 512 and edge["passes"] is True  # 정확히 +0.5 GB는 아직 통과다
    both = {
        "directions": {
            "en-ko": {"pick": "hy-mt2-q8", "arms": {"hy-mt2-q8": {"vram_mb": 2400}}},
            "ko-en": {"pick": "hy-mt2-q8", "arms": {"hy-mt2-q8": {"vram_mb": 2400}}},
        }
    }
    # Ollama 모델 하나가 두 방향을 맡으면 그 메모리는 한 번만 센다.
    assert vram(both, None)["net_mb"] == 2400 - 512 - 514
    whole_and_split = {
        "directions": {
            "en-ko": {"pick": "hy-mt2-q8/split", "arms": {"hy-mt2-q8/split": {"vram_mb": 2400}}},
            "ko-en": {"pick": "hy-mt2-q8", "arms": {"hy-mt2-q8": {"vram_mb": 2380}}},
        }
    }
    # 통째로 팔과 문장 단위 팔은 같은 빌드라 역시 한 번만 센다(둘 중 큰 값).
    assert vram(whole_and_split, None)["net_mb"] == 2400 - 512 - 514


# 회귀 테스트(커밋 22fe56c): 한 방향은 Q4_K_M, 다른 방향은 Q8_0이면 Ollama 모델 두 개가 함께 올라가므로 둘을
# 더해야 한다. 예전에는 하나만 셌다.
def test_vram_adds_two_different_builds():
    decision = {
        "directions": {
            "en-ko": {"pick": "hy-mt2-q4", "arms": {"hy-mt2-q4": {"vram_mb": 1200}}},
            "ko-en": {"pick": "hy-mt2-q8", "arms": {"hy-mt2-q8": {"vram_mb": 2000}}},
        }
    }
    result = vram(decision, {"retire_corrector": True, "corrector_vram_mb": 1000})
    assert result["hymt_vram_mb"] == 1200 + 2000
    assert result["net_mb"] == 1200 + 2000 - 512 - 514 - 1000 and result["passes"] is False


# eval.typo_eval 보고서의 후보 하나 모양을 만든다. 후보 C는 교정기(qwen2.5 1.5B)를 앞에 둔 opus, 후보 A는
# 교정기 없는 번역기다. 수준(clean·light·heavy)별 chrF와 문장 항목을 담는다.
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


# 오타 판정: 교정기 없는 Hy-MT2의 오타 평균 chrF(가벼움·심함 평균)가 교정기 + opus의 오타 평균 − 0.5 이상이면
# 영어 오타 교정기를 은퇴시킨다. 같으면 은퇴, 오타 수준에서 크게 낮으면 유지(차이 < −0.5)이고, 은퇴 시 빠질
# 교정기 메모리(1300MB)를 (d) 계산용으로 넘기는지 확인한다.
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


# 오염 거부권: MetricX QE는 오류 점수(낮을수록 좋음)라, Hy-MT2 − opus 차이의 구간 하한이 0보다 크면(opus가
# 뚜렷이 나음) 그 방향을 거부한다. Hy-MT2 점수가 약 3점 높으면 거부, 같은 분포면 거부하지 않는다. 이 점검은
# 거부만 하고 채택 근거가 되지 않는다(15절).
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


# MetricX 채점 입력은 '시스템/방향/문장 id' 키로 시스템마다 한 줄씩이며, 방향의 원문 언어와 같은 고정 세트
# 문장만 들어간다(출력이 영→한뿐이라 한국어 문장 ko-001은 빠짐).
def test_metricx_input_keys_each_system_direction_and_sentence():
    frozen = [{"id": "en-001", "lang": "en", "text": "Hi."}, {"id": "ko-001", "lang": "ko", "text": "안녕."}]
    outputs = {"hymt": {"en-ko": {"en-001": "안녕."}}, "opus": {"en-ko": {"en-001": "안녕하세요."}}}
    assert metricx_input(frozen, outputs) == [
        {"key": "hymt/en-ko/en-001", "src": "Hi.", "mt": "안녕."},
        {"key": "opus/en-ko/en-001", "src": "Hi.", "mt": "안녕하세요."},
    ]


# 오타 보고서의 COMET 입력은 오타 섞인 입력("Clena sourse.")이 아니라 같은 문장 id의 깨끗한 원문과 참조를
# 쓴다(15절 지표: 6절 오타 세트도 깨끗한 원문과 참조로). 그래서 오타 보고서의 두 수준과 번역 보고서의 같은
# (원문, 번역, 참조)가 중복 없이 하나로 모인다. 오타 세트 파일 읽기는 monkeypatch로 바꾼다.
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
