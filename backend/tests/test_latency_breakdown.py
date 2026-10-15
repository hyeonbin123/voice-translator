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
    assert decision["ii_open_a9"] is True
    assert decision["iii_open_a3"] is True
    assert decision["iv_a3_skip_cache_variant"] is True
    assert lb.decide(rows, stages_summary(multi=0.29, cache=0.09))["iii_open_a3"] is False
    assert lb.decide(rows, stages_summary(multi=0.29, cache=0.09))["iv_a3_skip_cache_variant"] is False

    late = report("live", [item(prepared=1600.0, first_wait=99.0)] * 3, [item(first_wait=10.0)] * 3)
    fast_tts = report("conversation", [item(tts=399.0, prepared=700.0)] * 3, [])
    decision = lb.decide(lb.summarize(late) + lb.summarize(fast_tts), stages_summary())
    assert decision["i_live_en_within_goal"] is False
    assert decision["ii_open_a9"] is False
    # 합성 중앙값은 두 모드의 영어 원문 행을 합친 중앙값(600과 399의 중앙값 499.5)이라 아직 문턱 이상이다.
    assert decision["values"]["en_source_tts_p50_ms"] == pytest.approx(499.5)


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
