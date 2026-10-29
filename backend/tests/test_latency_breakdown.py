"""말 끝 → 번역 음성 단계별 진단(eval/latency_breakdown.py)의 나누기와 17절 판단 규칙을 가짜 보고서로
검증한다."""

# [파일 역할] docs/experiments.md 17절의 진단 계산을 모델·GPU·서버 없이 확인한다. 마디 하나를 클라이언트·서버
# 시각으로 나누는 방법(components), 모드·언어별 중앙값(summarize), 측정 전에 정한 판단 규칙(decide)의 경계,
# 보고서 쓰기를 본다.
# [왜 중요한가] 이 계산의 결과가 다음 작업(A3·A9)을 열지 닫을지를 정한다. 규칙의 문턱이나 시각 맞추기가 틀리면
# 측정이 맞아도 결론이 달라진다.

import argparse
import json

import pytest

from eval import latency_breakdown as lb


# 서버가 LIVE_TIMINGS로 보낸 timings 모양(routers/live.Timing.report)과 stream_eval service의 마디 결과
# 모양으로 마디 하나를 만든다. 기본값: pause를 말 끝 0.25초 뒤에 보내고, 처리 1100ms(1초 확정 end 800ms보다
# 늦음), 저장 30ms, final 1135ms, 전달 5ms, 음성 받기 16ms.
def item(
    *,
    prepared=1100.0,
    end=800.0,
    first_wait=40.0,
    tts=600.0,
    anchor="pause",
    busy=True,
    outcome="final",
    audio=True,
):
    pause_sent = 0.25
    save_started = max(prepared, end) + 2.0
    final_ms = save_started + 30.0 + 3.0
    final_s = pause_sent + (final_ms + 5.0) / 1000
    stages = {
        "validate": {"queued_ms": 0.1, "started_ms": first_wait, "wait_ms": first_wait, "run_ms": 2.0},
        "stt": {
            "queued_ms": first_wait + 2.5,
            "started_ms": first_wait + 3.0,
            "wait_ms": 0.5,
            "run_ms": 300.0,
        },
        "mt": {
            "queued_ms": first_wait + 303.5,
            "started_ms": first_wait + 304.0,
            "wait_ms": 0.5,
            "run_ms": 70.0,
        },
        "tts": {
            "queued_ms": first_wait + 374.5,
            "started_ms": first_wait + 375.0,
            "wait_ms": 0.5,
            "run_ms": tts,
        },
    }
    return {
        "key": "1-1",
        "outcome": outcome,
        "final_s": final_s,
        "audio_s": final_s + 0.016 if audio else None,
        "pause_sent_s": pause_sent if anchor == "pause" else None,
        "end_sent_s": pause_sent + end / 1000,
        "timings": {
            "from": anchor,
            "update_busy": busy,
            "stages": stages,
            "prepared_ms": prepared,
            "end_ms": end if anchor == "pause" else 0.0,
            "save_started_ms": save_started,
            "save_ms": 30.0,
            "final_ms": final_ms,
            "redone": False,
        },
    }


# 한 마디의 나누기: 말 끝 → 음성이 (기준점까지 + final을 보낼 때까지 + 전달 + 음성 받기)로 맞아떨어지고, 첫
# 단계의 대기와 단계 사이 대기(0.5ms × 3)를 따로 센다. 처리(1100ms)가 end(800ms)보다 늦으면 '처리가 확정보다
# 늦음'이다.
def test_components_split_the_end_to_audio_time():
    part = lb.components(item())
    assert part["audio"] == pytest.approx(
        part["anchor"] + part["final_server"] + part["delivery"] + part["audio_get"]
    )
    assert part["anchor"] == pytest.approx(250.0)
    assert part["delivery"] == pytest.approx(5.0)
    assert part["audio_get"] == pytest.approx(16.0)
    assert (part["first_wait"], part["between_wait"]) == (40.0, 1.5)
    assert (part["stt"], part["mt"], part["tts"]) == (300.0, 70.0, 600.0)
    assert part["save_wait"] == pytest.approx(2.0)
    assert part["processing_bound"] is True and part["update_busy"] is True
    assert lb.components(item(prepared=700.0))["processing_bound"] is False


# pause 없이 end에서 처리한 마디는 end를 보낸 때를 기준점으로 쓴다. 실패한 마디, 음성이 없는 마디, 서버 시각이
# 없는 마디(LIVE_TIMINGS를 끈 서버)는 규칙에서 뺀다.
def test_components_use_the_end_as_anchor_and_skip_unusable_utterances():
    from_end = lb.components(item(anchor="end"))
    assert from_end["anchor"] == pytest.approx(1050.0)
    assert lb.components(item(outcome="error")) is None
    assert lb.components(item(audio=False)) is None
    assert lb.components({**item(), "timings": None}) is None


def report(mode, en_items, ko_items):
    return {"mode": mode, "details": {"en": en_items, "ko": ko_items}}


# 모드·언어별 요약은 쓸 수 있는 마디만 세고, 처리가 확정보다 늦은 마디와 갱신 중 pause의 비율을 낸다.
def test_summary_counts_only_usable_utterances():
    rows = lb.summarize(report("live", [item(), item(prepared=700.0, busy=False), item(outcome="error")], []))
    en = next(row for row in rows if row["lang"] == "en")
    assert (en["utterances"], en["used"]) == (3, 2)
    assert en["processing_bound"] == 0.5 and en["update_busy"] == 0.5
    ko = next(row for row in rows if row["lang"] == "ko")
    assert ko["used"] == 0 and ko["audio_p50"] is None


def stages_summary(multi=0.4, cache=0.12):
    return {"multi_piece_ratio": multi, "stages": {"empty_cache": {"share": cache}}}


# 규칙 i·ii·iii·iv가 각 문턱에서 바뀌는지(이상이면 참). iii은 합성 시간·조각 비율·처리가 확정보다 늦은 비율이
# 모두 문턱 이상일 때만 참이다.
def test_the_rules_switch_at_their_thresholds():
    slow = report("live", [item(first_wait=120.0)] * 3, [item(first_wait=20.0)] * 3)
    conversation = report("conversation", [item()] * 3, [item()] * 3)
    rows = lb.summarize(slow) + lb.summarize(conversation)
    decision = lb.decide(rows, stages_summary())
    assert decision["i_live_en_within_goal"] is True  # 말 끝 → 음성이 약 1.4초
    # 대기 조건은 맞지만 동시통역이 이미 목표 안이라 A9는 열지 않는다(17절 측정 전 정정).
    assert decision["ii_queue_condition"] is True and decision["ii_open_a9"] is False
    assert decision["iii_open_a3"] is True
    assert decision["iv_cache_condition"] is True and decision["iv_a3_skip_cache_variant"] is True
    assert lb.decide(rows, stages_summary(multi=0.29, cache=0.09))["iii_open_a3"] is False
    assert lb.decide(rows, stages_summary(multi=0.29, cache=0.09))["iv_a3_skip_cache_variant"] is False

    late = report("live", [item(prepared=1600.0, first_wait=99.0)] * 3, [item(first_wait=10.0)] * 3)
    fast_tts = report("conversation", [item(tts=399.0, prepared=700.0)] * 3, [])
    decision = lb.decide(lb.summarize(late) + lb.summarize(fast_tts), stages_summary())
    assert decision["i_live_en_within_goal"] is False
    assert decision["ii_queue_condition"] is False and decision["ii_open_a9"] is False
    # 합성 중앙값은 두 모드의 영어 원문 행을 합친 중앙값(600과 399의 중앙값 499.5)이라 아직 문턱 이상이다.
    assert decision["values"]["en_source_tts_p50_ms"] == pytest.approx(499.5)


# 경계(이상·이하)와 선행 조건: 말 끝 → 음성 p50이 정확히 1500ms면 목표 안(i 참)이라 첫 대기가 정확히 100ms여도
# A9를 열지 않고, 1500ms를 넘으면 같은 대기에서 연다. iii이 거짓이거나 판단 못 함이면 캐시 몫이 정확히 10%여도
# 생략 변형을 넣지 않는다(각각 거짓, None).
def test_rule_boundaries_and_preconditions():
    def rows_with(audio_ms: float, wait_ms: float) -> list[dict]:
        row = {
            "mode": "live",
            "lang": "en",
            "audio_p50": audio_ms,
            "first_wait_p50": wait_ms,
            "tts_p50": 600.0,
            "processing_bound": 0.6,
        }
        return [row]

    at_goal = lb.decide(rows_with(1500.0, 100.0), stages_summary(cache=0.10))
    assert at_goal["i_live_en_within_goal"] is True
    assert at_goal["ii_queue_condition"] is True and at_goal["ii_open_a9"] is False
    over = lb.decide(rows_with(1500.1, 100.0), stages_summary(cache=0.10))
    assert over["i_live_en_within_goal"] is False and over["ii_open_a9"] is True
    assert over["iii_open_a3"] is True and over["iv_a3_skip_cache_variant"] is True  # 캐시 몫 10%는 '이상'
    closed = lb.decide(rows_with(1500.1, 99.9), stages_summary(multi=0.29, cache=0.10))
    assert closed["ii_open_a9"] is False
    assert closed["iii_open_a3"] is False
    assert closed["iv_cache_condition"] is True and closed["iv_a3_skip_cache_variant"] is False
    unknown = lb.decide([{**rows_with(1500.1, 100.0)[0], "tts_p50": None}], stages_summary(cache=0.10))
    assert unknown["iii_open_a3"] is None and unknown["iv_a3_skip_cache_variant"] is None
    no_live = lb.decide([], None)
    assert no_live["i_live_en_within_goal"] is None and no_live["ii_open_a9"] is None


# 처리가 1초 확정보다 늦은 마디가 어느 모드에서도 절반 미만이면 조각을 먼저 보내도 말 끝 → 음성이 줄지
# 않으므로 A3를 열지 않는다. 합성 보고서가 없으면 iii·iv는 판단 못 함(None)이다.
def test_a3_stays_closed_when_processing_ends_before_the_commit():
    early = report("live", [item(prepared=700.0)] * 3, [item(prepared=700.0)] * 3)
    rows = lb.summarize(early)
    assert lb.decide(rows, stages_summary())["iii_open_a3"] is False
    unknown = lb.decide(rows, None)
    assert unknown["iii_open_a3"] is None and unknown["iv_a3_skip_cache_variant"] is None


# 보고서는 JSON(행·판단·입력 경로)과 사람이 읽을 표(MD)로 남고, 표의 열 수가 제목과 같다.
def test_the_report_keeps_rows_and_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "REPORTS", tmp_path)
    rows = lb.summarize(report("live", [item()] * 2, [item()] * 2))
    decision = lb.decide(rows, stages_summary())
    args = argparse.Namespace(tag="t", stream=[tmp_path / "s.json"], stages=None)
    md = lb.write_report(rows, decision, stages_summary(), args)
    record = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["decision"] == decision and len(record["rows"]) == 2
    table = [line for line in md.read_text(encoding="utf-8").splitlines() if line.startswith("| live")]
    assert len(table) == 2 and all(line.count("|") == len(lb.COLUMNS) + 1 for line in table)


# 여기부터 18절(g2pkk 정규식 캐시 끔 A / 켬 B)의 짝 계산. 블록 하나는 서버를 한 조건으로 띄우고 잰 stream_eval
# service 보고서 하나다. 같은 클립(key)끼리 서버 timings의 합성 실행 시간을 짝짓고, 같은 조건의 블록이 둘이면
# (동시통역 A1·A2, B1·B2) 클립마다 두 블록을 평균한다.


def clip(key, tts, *, source="hello", translated="안녕", **kwargs):
    made = item(tts=tts, **kwargs)
    made["key"] = key
    made["result"] = {"source_text": source, "translated_text": translated}
    return made


def block(mode, en=(), ko=(), sm=1800.0):
    details = {}
    if en:
        details["en"] = list(en)
    if ko:
        details["ko"] = list(ko)
    gpu = {"samples": 10, "mean_util": 20.0, "sm_clock_p50_mhz": sm, "busy_sm_clock_p50_mhz": sm + 100}
    return {"mode": mode, "details": details, "gpu_use": {**gpu, "pstates": {"P2": 10}, "limited_ratio": 0.1}}


# 같은 클립끼리 짝짓는다: 동시통역은 A 두 블록 평균 − B 두 블록 평균, 대화 모드는 블록 하나씩. 한 블록에서라도
# 원문·번역문이 다른 클립은 짝에서 빼고 수를 센다. 주 지표는 두 모드에 모두 남은 클립마다 두 모드의 차이를
# 평균한 값의 평균이고, 구간은 클립을 한 번 뽑아 두 모드에 함께 쓰는 짝지은
# 재표본(significance.mean_difference)이다.
def test_ab_pairs_the_same_clips_and_averages_the_two_live_blocks():
    live_a = [
        ("a1", block("live", en=[clip("c1", 600), clip("c2", 700), clip("c3", 800)])),
        ("a2", block("live", en=[clip("c1", 620), clip("c2", 680), clip("c3", 820)])),
    ]
    live_b = [
        ("b1", block("live", en=[clip("c1", 300), clip("c2", 400), clip("c3", 500, translated="다름")])),
        ("b2", block("live", en=[clip("c1", 280), clip("c2", 420), clip("c3", 520)])),
    ]
    conv_a = [("ca", block("conversation", en=[clip("c1", 610), clip("c2", 690), clip("c3", 790)]))]
    conv_b = [("cb", block("conversation", en=[clip("c1", 310), clip("c2", 390), clip("c3", 495)]))]
    live = lb.pair_mode([r for _, r in live_a], [r for _, r in live_b], "live")
    assert live["keys"] == ["c1", "c2"] and live["text_differs"] == ["c3"]
    assert live["a"] == [610.0, 690.0] and live["b"] == [290.0, 410.0]
    result = lb.compare(live_a + conv_a, live_b + conv_b, None)
    primary = result["primary"]
    assert primary["modes"] == ["live", "conversation"] and primary["clips"] == 2
    assert primary["point_ms"] == pytest.approx(((320 + 300) / 2 + (280 + 300) / 2) / 2)
    assert primary["low_ms"] <= primary["point_ms"] <= primary["high_ms"]
    assert result["modes"]["conversation"]["clips"] == 3  # 대화 모드만 보면 c3도 짝이 된다
    assert result["modes"]["live"]["median_ms"] == pytest.approx(300.0)
    assert result["excluded"]["text_differs"] == ["c3"] and result["errors"] == 0
    assert result["decision"]["text_differs_warning"] is False


# 주 지표의 구간은 같은 클립 수·시드에서 늘 같다(13절과 같은 10,000번·시드 12345). 모든 클립의 차이가 같으면
# 구간이 그 값 한 점이다.
def test_the_primary_interval_is_seeded_and_collapses_for_constant_differences():
    a = [("a", block("conversation", en=[clip(f"c{i}", 500.0 + i) for i in range(5)]))]
    b = [("b", block("conversation", en=[clip(f"c{i}", 380.0 + i) for i in range(5)]))]
    first, again = lb.compare(a, b, None)["primary"], lb.compare(a, b, None)["primary"]
    assert first == again
    assert first["point_ms"] == first["low_ms"] == first["high_ms"] == pytest.approx(120.0)


# 채택 규칙(모두 참이어야 채택): (a) 주 지표 구간 하한이 100ms보다 큼(같으면 거짓) (b) 한국어 원문 말 끝 →
# 음성 p50 회귀(B − A)가 +50ms 이하(같으면 참) (c) 오류 0. 짝에서 원문·번역문이 달라 뺀 클립이 2개를 넘으면
# 정상성 경고만 켜고 판정은 바꾸지 않는다. 값을 모르면(None) 채택도 None이다.
def test_the_adoption_rule_boundaries():
    assert lb.decide_ab(100.01, 50.0, 0, 0)["adopt"] is True
    assert lb.decide_ab(100.0, 0.0, 0, 0)["a_primary_low_over_100"] is False
    assert lb.decide_ab(100.0, 0.0, 0, 0)["adopt"] is False
    assert lb.decide_ab(150.0, 50.01, 0, 0)["b_ko_regression_within_50"] is False
    assert lb.decide_ab(150.0, -20.0, 1, 0)["c_no_errors"] is False
    assert lb.decide_ab(150.0, -20.0, 1, 0)["adopt"] is False
    warned = lb.decide_ab(150.0, -20.0, 0, 3)
    assert warned["text_differs_warning"] is True and warned["adopt"] is True
    assert lb.decide_ab(150.0, -20.0, 0, 2)["text_differs_warning"] is False
    assert lb.decide_ab(None, -20.0, 0, 0)["adopt"] is None
    assert lb.decide_ab(150.0, None, 0, 0)["adopt"] is None
    assert lb.decide_ab(50.0, None, 0, 0)["adopt"] is False  # 하나라도 거짓이면 모르는 칸이 있어도 거짓


# 실패한 마디(최종 결과 없음)와 음성이 없는 마디는 오류로 세고 짝에서 뺀다. 어느 블록에 없는 클립(고르기가
# 달라짐)도 짝에서 빼고 따로 센다. 한국어 원문 회귀는 조건별 동시통역 한국어 블록의 말 끝 → 음성 p50 차이(B −
# A)다.
def test_ab_counts_errors_and_missing_clips_and_the_korean_regression():
    a = [
        ("a_en", block("conversation", en=[clip("c1", 600), clip("c2", 600), clip("c3", 600)])),
        ("a_ko", block("live", ko=[clip("k1", 150, prepared=1000.0), clip("k2", 150, prepared=1000.0)])),
    ]
    b = [
        ("b_en", block("conversation", en=[clip("c1", 300), clip("c2", 300, outcome="error")])),
        ("b_ko", block("live", ko=[clip("k1", 150, prepared=1030.0), clip("k2", 150, prepared=1030.0)])),
    ]
    result = lb.compare(a, b, None)
    assert result["modes"]["conversation"]["clips"] == 1
    assert result["excluded"]["unusable"] == ["c2"] and result["excluded"]["missing"] == ["c3"]
    assert result["errors"] == 1 and result["decision"]["c_no_errors"] is False
    assert result["ko_regression_ms"] == pytest.approx(30.0)
    assert result["decision"]["b_ko_regression_within_50"] is True
    assert result["decision"]["adopt"] is False


# 17절 규칙 iii(A3를 열 근거)를 조건마다 같은 문턱으로 다시 계산한다. 같은 조건·모드의 블록은 마디를 합쳐
# 요약하고, 조각 비율과 캐시 비우기 몫은 melo-stages alternate 보고서의 그 조건 요약에서 읽는다. 조건이
# 하나뿐인 옛 보고서(17절)는 끔(A) 요약으로만 쓴다.
def test_ab_recomputes_rule_iii_of_section_17_for_each_condition():
    a = [
        ("a1", block("live", en=[clip("c1", 600, prepared=1100.0), clip("c2", 600, prepared=1100.0)])),
        ("a2", block("live", en=[clip("c1", 600, prepared=1100.0), clip("c2", 600, prepared=1100.0)])),
    ]
    b = [("b1", block("live", en=[clip("c1", 300, prepared=700.0), clip("c2", 300, prepared=700.0)]))]
    alternate = {
        "regex_cache": "alternate",
        "summary": {
            "off": stages_summary(multi=0.45, cache=0.01),
            "on": stages_summary(multi=0.45, cache=0.02),
        },
    }
    result = lb.compare(a, b, alternate)
    assert result["rule_iii"]["A"]["iii_open_a3"] is True
    assert result["rule_iii"]["B"]["iii_open_a3"] is False
    assert result["rule_iii"]["B"]["values"]["en_source_tts_p50_ms"] == 300.0
    live_a = next(row for row in result["rows"]["A"] if row["mode"] == "live" and row["lang"] == "en")
    assert live_a["utterances"] == 4  # 두 블록의 마디를 합침
    old = {"summary": stages_summary(multi=0.45, cache=0.01)}
    assert lb.stage_summary(old, "off") == old["summary"] and lb.stage_summary(old, "on") is None
    assert lb.stage_summary(alternate, "on") == alternate["summary"]["on"]
    assert lb.stage_summary(None, "off") is None


# 명령줄: --a·--b로 18절 계산을 하고 latency_ab_<tag>_<시각>.json·.md를 쓴다(블록별 GPU 클럭 표 포함). 17절의
# --stream 사용은 그대로이고, 둘 다 없거나 한쪽만 주면 멈춘다.
def test_the_ab_command_writes_its_report_and_keeps_the_stream_form(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(lb, "REPORTS", tmp_path)

    def save(name, report_):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report_, ensure_ascii=False), encoding="utf-8")
        return str(path)

    a = save("stream_t_live_en_a1", block("live", en=[clip("c1", 600), clip("c2", 640)], sm=1700.0))
    b = save("stream_t_live_en_b1", block("live", en=[clip("c1", 300), clip("c2", 330)], sm=1750.0))
    monkeypatch.setattr(sys, "argv", ["latency_breakdown", "--a", a, "--b", b, "--tag", "t_ab"])
    lb.main()
    [md] = tmp_path.glob("latency_ab_t_ab_*.md")
    record = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["result"]["modes"]["live"]["clips"] == 2 and record["a"] == [a]
    text = md.read_text(encoding="utf-8")
    assert "stream_t_live_en_a1" in text and "1700" in text and "1750" in text
    monkeypatch.setattr(sys, "argv", ["latency_breakdown", "--stream", a, "--tag", "t_stream"])
    lb.main()
    assert list(tmp_path.glob("latency_t_stream_*.md"))
    for argv in (["--tag", "x"], ["--a", a, "--tag", "x"], ["--stream", a, "--a", a, "--b", b, "--tag", "x"]):
        monkeypatch.setattr(sys, "argv", ["latency_breakdown", *argv])
        with pytest.raises(SystemExit):
            lb.main()
