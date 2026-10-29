"""말을 멈춘 뒤 번역 음성까지의 시간을 단계별로 나누고, 측정 전에 정한 규칙으로 다음 작업을 고른다.

  uv run python -m eval.latency_breakdown --stream <live 보고서> <conversation 보고서> \\
      --stages <tts melo-stages 보고서> --tag t95_dev
  uv run python -m eval.latency_breakdown --a <끔 블록 보고서들> --b <켬 블록 보고서들> \\
      --stages <tts melo-stages alternate 보고서> --tag t105_dev

첫 형태는 docs/experiments.md 17절(측정 전에 커밋)의 진단이다. 입력은 서버를 LIVE_TIMINGS=true로 띄우고
`eval.stream_eval service`로 잰 보고서(final마다 서버의 단계별 시각이 들어 있음)와, 같은 번역문을
`eval.tts_eval melo-stages`로 합성한 MeloTTS 단계별 시간이다. 둘째 형태는 18절(g2pkk 정규식 캐시
MELO_REGEX_CACHE 끔 A / 켬 B)의 짝 계산이다: 조건마다 서버를 다시 띄워 잰 블록 보고서들을 같은 클립끼리
짝짓고 18절 채택 규칙을 적용한다. 둘 다 측정이 아니라 저장된 보고서를 읽어 계산만 한다.
"""

# [파일 역할] 17절의 진단 계산. 마디마다 '말 끝 → 번역 음성'을 클라이언트 쪽(pause를 보낸 때, final을 받은 때,
# 음성을 다 받은 때)과 서버 쪽(LIVE_TIMINGS의 단계별 시각)으로 나누고, 모드·원문 언어별 중앙값을 낸 뒤 17절
# 판단 규칙(A9·A3를 열지, A3에 캐시 비우기 생략 변형을 넣을지, 동시통역 영어 원문이 이미 목표 안인지)을
# 적용한다. [왜 따로 두나] 잴 때(GPU)와 판단할 때(CPU)를 나눠, 규칙 계산을 가짜 보고서로 테스트하고 같은
# 보고서로 몇 번이고 다시 낼 수 있게 한다(tests/test_latency_breakdown.py). [시각 맞추기] 서버 시각은 서버가
# pause(또는 pause 없이 end)를 받은 때를 0으로 잰 밀리초다. 측정 도구와 서버가 같은 PC라 pause를 보낸 때와
# 받은 때는 거의 같다(차이는 '전달'로 남겨 따로 보인다).

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval.common import REPORTS

# significance는 numpy만 쓰는 모듈이다(sacrebleu는 타입 표기용). 18절 주 지표의 짝지은 재표본에 13절과 같은
# 함수(mean_difference, 10,000번·시드 12345)를 쓴다.
from eval.significance import mean_difference

# 17절 판단 규칙의 문턱(측정 전에 정함). GOAL_MS: 처음 목표(말 끝 → 번역 음성 1.5초). QUEUE_MS: 최종 처리가
# 모델 스레드를 기다린 시간(첫 단계의 대기) 중앙값이 이 이상이면 A9(자막 갱신이 최종 처리를 막지 않게)를 연다.
# KO_TTS_MS·MULTI_PIECE·PROCESSING_BOUND: 한국어 합성 중앙값, 조각 2개 이상인 번역문 비율, 처리가 1초 확정보다
# 늦게 끝난 마디 비율이 모두 이 이상이면 A3(다 만든 조각부터 보내기)를 연다. CACHE_SHARE: MeloTTS 합성에서
# 캐시 비우기가 차지하는 몫이 이 이상이면 A3에 '캐시 비우기 생략' 변형을 넣는다.
GOAL_MS = 1500
QUEUE_MS = 100
KO_TTS_MS = 400
MULTI_PIECE = 0.30
PROCESSING_BOUND = 0.50
CACHE_SHARE = 0.10


def components(item: dict) -> dict | None:
    """마디 하나를 단계별 밀리초로 나눈다. 최종 결과·음성·서버 시각이 없는 마디는 None(규칙에서 뺀다).

    돌려주는 값: audio(말 끝 → 음성 받기 끝), anchor(말 끝 → 서버 기준점을 보낸 때: pause 또는 end),
    first_wait(최종 처리의 첫 단계가 모델 스레드를 기다린 시간), stt·mt·tts(각 실행), between_wait(단계 사이에
    다시 줄 선 시간의 합), prepared(기준점 → 처리 끝), end(기준점 → end 받음), save_wait(처리·end 중 늦은 쪽
    뒤 저장 시작까지), save, final_server(기준점 → final 보냄), delivery(final을 보낸 때 → 받은 때, 전달),
    audio_get(final 받음 → 음성 받기 끝), processing_bound(처리가 end보다 늦게 끝났나), update_busy.
    """
    timings = item.get("timings")
    if item.get("outcome") != "final" or item.get("audio_s") is None or not timings:
        return None
    anchor_s = item["pause_sent_s"] if timings["from"] == "pause" else item["end_sent_s"]
    if anchor_s is None:
        return None
    stages = timings["stages"]
    order = sorted(stages, key=lambda name: stages[name]["started_ms"])
    first = order[0]
    run = {name: stages[name]["run_ms"] if name in stages else None for name in ("stt", "mt", "tts")}
    return {
        "audio": item["audio_s"] * 1000,
        "anchor": anchor_s * 1000,
        "first_wait": stages[first]["wait_ms"],
        **run,
        "between_wait": sum(stages[name]["wait_ms"] for name in order[1:]),
        "prepared": timings["prepared_ms"],
        "end": timings["end_ms"],
        "save_wait": timings["save_started_ms"] - max(timings["prepared_ms"], timings["end_ms"]),
        "save": timings["save_ms"],
        "final_server": timings["final_ms"],
        "delivery": (item["final_s"] - anchor_s) * 1000 - timings["final_ms"],
        "audio_get": (item["audio_s"] - item["final_s"]) * 1000,
        "processing_bound": timings["prepared_ms"] > timings["end_ms"],
        "update_busy": bool(timings["update_busy"]),
    }


def median(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def summarize(report: dict) -> list[dict]:
    """보고서 하나(한 모드)의 원문 언어별 요약: 마디 수, 쓴 마디 수, 말 끝 → 음성 p50·p90, 단계별 p50, 처리가
    1초 확정보다 늦게 끝난 비율, 자막 갱신이 돌던 중 pause가 온 비율."""
    rows = []
    for lang, items in report["details"].items():
        parts = [part for item in items if (part := components(item)) is not None]
        audio = sorted(part["audio"] for part in parts)
        row = {
            "mode": report.get("mode", "live"),
            "lang": lang,
            "utterances": len(items),
            "used": len(parts),
            "audio_p50": median(audio),
            "audio_p90": audio[min(len(audio) - 1, round(0.9 * (len(audio) - 1)))] if audio else None,
            "processing_bound": sum(part["processing_bound"] for part in parts) / len(parts)
            if parts
            else None,
            "update_busy": sum(part["update_busy"] for part in parts) / len(parts) if parts else None,
        }
        for key in (
            "anchor",
            "first_wait",
            "stt",
            "mt",
            "tts",
            "between_wait",
            "prepared",
            "end",
            "save_wait",
            "save",
            "final_server",
            "delivery",
            "audio_get",
        ):
            row[f"{key}_p50"] = median([part[key] for part in parts])
        rows.append(row)
    return rows


def gate(precondition: bool | None, condition: bool | None) -> bool | None:
    """선행 조건이 참일 때만 조건을 따른다. 선행 조건이 거짓이면 거짓, 둘 중 하나라도 모르면(None) None."""
    if precondition is False:
        return False
    if precondition is None or condition is None:
        return None
    return condition


def decide(rows: list[dict], stages: dict | None) -> dict:
    """17절 판단 규칙. rows는 summarize의 결과(두 모드), stages는 melo-stages 보고서의 summary(없으면 None).

    조건(*_condition)과 실제로 여는지(open_*)를 나눠 둘 다 돌려준다(17절 2026-10-15 측정 전 정정).
    - i 동시통역 영어 원문의 말 끝 → 음성 p50이 GOAL_MS 이하면 '이미 목표 안'이라 동시통역 지연 개선(A9)을
      열지 않는다. README 수치를 test로 다시 잴지는 사용자 결정
    - ii 동시통역에서 최종 처리의 첫 단계 대기 p50이 어느 원문 언어든 QUEUE_MS 이상인 조건. i가 거짓(목표를
      넘음)일 때만 A9를 연다
    - iii 영어 원문(한국어 합성)의 합성 p50이 KO_TTS_MS 이상이고, 조각 2개 이상인 번역문이 MULTI_PIECE
      이상이며, 두 모드 중 어느 하나에서라도 처리가 1초 확정보다 늦게 끝난 마디가 PROCESSING_BOUND 이상이면
      A3를 연다(사용자가 후보 C를 열기로 할 때. 조각을 먼저 보내는 것은 처리가 확정보다 늦을 때만 돕는다).
      A3는 대화 모드·두 사람 대화에도 해당하므로 i와 상관없다
    - iv MeloTTS 합성에서 캐시 비우기가 CACHE_SHARE 이상인 조건. iii이 참(A3를 엶)일 때만 A3에 캐시 비우기
      생략 변형을 넣는다
    규칙에 필요한 값이 없으면(마디가 없음 등) 그 규칙은 None(판단 못 함)으로 둔다.
    """
    by = {(row["mode"], row["lang"]): row for row in rows}
    live_en = by.get(("live", "en"), {}).get("audio_p50")
    waits = [by[("live", lang)]["first_wait_p50"] for lang in ("en", "ko") if ("live", lang) in by]
    waits = [wait for wait in waits if wait is not None]
    tts = median([row["tts_p50"] for row in rows if row["lang"] == "en"])
    bound = [
        row["processing_bound"] for row in rows if row["lang"] == "en" and row["processing_bound"] is not None
    ]
    multi = stages["multi_piece_ratio"] if stages else None
    cache = stages["stages"]["empty_cache"]["share"] if stages else None
    within_goal = None if live_en is None else live_en <= GOAL_MS
    queue_condition = None if not waits else max(waits) >= QUEUE_MS
    open_a3 = None
    if tts is not None and multi is not None and bound:
        open_a3 = tts >= KO_TTS_MS and multi >= MULTI_PIECE and max(bound) >= PROCESSING_BOUND
    cache_condition = None if cache is None else cache >= CACHE_SHARE
    return {
        "i_live_en_within_goal": within_goal,
        "ii_queue_condition": queue_condition,
        "ii_open_a9": gate(None if within_goal is None else not within_goal, queue_condition),
        "iii_open_a3": open_a3,
        "iv_cache_condition": cache_condition,
        "iv_a3_skip_cache_variant": gate(open_a3, cache_condition),
        "values": {
            "live_en_audio_p50_ms": live_en,
            "live_first_wait_p50_ms": max(waits) if waits else None,
            "en_source_tts_p50_ms": tts,
            "multi_piece_ratio": multi,
            "en_source_processing_bound_max": max(bound) if bound else None,
            "empty_cache_share": cache,
        },
    }


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


# 보고서 표의 열: (제목, 행에서 칸 글을 만드는 함수). 시간은 ms 중앙값, 비율은 백분율.
COLUMNS = [
    ("모드", lambda row: row["mode"]),
    ("원문", lambda row: row["lang"]),
    ("마디(씀)", lambda row: f"{row['utterances']}({row['used']})"),
    ("말 끝→음성 p50 / p90", lambda row: f"{_ms(row['audio_p50'])} / {_ms(row['audio_p90'])}"),
    ("말 끝→기준점", lambda row: _ms(row["anchor_p50"])),
    ("첫 대기", lambda row: _ms(row["first_wait_p50"])),
    ("인식", lambda row: _ms(row["stt_p50"])),
    ("번역", lambda row: _ms(row["mt_p50"])),
    ("합성", lambda row: _ms(row["tts_p50"])),
    ("단계 사이 대기", lambda row: _ms(row["between_wait_p50"])),
    ("처리 끝", lambda row: _ms(row["prepared_p50"])),
    ("end 받음", lambda row: _ms(row["end_p50"])),
    ("저장", lambda row: _ms(row["save_p50"])),
    ("final 보냄", lambda row: _ms(row["final_server_p50"])),
    ("전달", lambda row: _ms(row["delivery_p50"])),
    ("음성 받기", lambda row: _ms(row["audio_get_p50"])),
    ("처리가 확정보다 늦음", lambda row: _ratio(row["processing_bound"])),
    ("갱신 중 pause", lambda row: _ratio(row["update_busy"])),
]


def write_report(rows: list[dict], decision: dict, stages: dict | None, args: argparse.Namespace) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = REPORTS / f"latency_{args.tag}_{stamp}"
    record = {
        "tag": args.tag,
        "streams": [str(path) for path in args.stream],
        "stages": str(args.stages) if args.stages else None,
        "rows": rows,
        "melo_stages": stages,
        "decision": decision,
    }
    base.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [
        f"# 말 끝 → 번역 음성 단계별 진단 ({args.tag})",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}, 보고서: {', '.join(Path(p).name for p in args.stream)}",
        "- 모든 값은 마디별 값의 중앙값(ms). 기준점은 서버가 최종 처리를 시작한 때(보통 pause를 받은 때)",
        "",
        "| " + " | ".join(title for title, _ in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row) for _, cell in COLUMNS) + " |")
    values = decision["values"]
    lines += [
        "",
        "## 17절 판단 규칙",
        "",
        f"- i 동시통역 영어 원문 p50 {_ms(values['live_en_audio_p50_ms'])}ms ≤ {GOAL_MS}ms: "
        f"{decision['i_live_en_within_goal']}",
        f"- ii 동시통역 첫 대기 p50(나쁜 쪽) {_ms(values['live_first_wait_p50_ms'])}ms ≥ {QUEUE_MS}ms: "
        f"{decision['ii_queue_condition']} → i가 거짓일 때만 A9 열기: {decision['ii_open_a9']}",
        f"- iii 영어 원문 합성 p50 {_ms(values['en_source_tts_p50_ms'])}ms ≥ {KO_TTS_MS}ms, 조각 2개 이상 "
        f"{_ratio(values['multi_piece_ratio'])} ≥ {MULTI_PIECE:.0%}, 처리가 확정보다 늦음(나쁜 모드) "
        f"{_ratio(values['en_source_processing_bound_max'])} ≥ {PROCESSING_BOUND:.0%} → A3 열기: "
        f"{decision['iii_open_a3']}",
        f"- iv 캐시 비우기 몫 {_ratio(values['empty_cache_share'])} ≥ {CACHE_SHARE:.0%}: "
        f"{decision['iv_cache_condition']} → iii이 참일 때만 A3에 생략 변형: "
        f"{decision['iv_a3_skip_cache_variant']}",
    ]
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".md")


# 여기부터 18절(g2pkk 정규식 캐시 끔 A / 켬 B, 측정 전에 커밋)의 짝 계산. 블록 하나는 서버를 한 조건으로 다시
# 띄우고 잰 stream_eval service 보고서 하나다(동시통역 영어 A·B·B·A, 동시통역 한국어 A·B, 대화 모드 영어 B·A).
# 서버 설정이라 문장마다 조건을 바꿀 수 없어 블록 단위로 재고, 같은 클립(key)끼리 짝지어 클립 난이도 차이를
# 뺀다. 18절 채택 규칙의 문턱(측정 전에 정함). PRIMARY_LOW_MS: 주 지표(영어 원문 마디의 서버 합성 실행 시간을
# 같은 클립끼리 짝지은 A − B) 95% 구간 하한이 이 값보다 커야 한다. KO_REGRESSION_MS: 한국어 원문(영어 Kokoro
# 합성, g2pkk를 쓰지 않음)의 말 끝 → 음성 p50이 B에서 A보다 이만큼까지만 늦어도 된다. TEXT_DIFFER_WARN:
# 원문·번역문이 블록 사이에 달라 짝에서 뺀 영어 클립이 이보다 많으면 정상성 경고(판정은 바꾸지 않음).
# AB_MODES: 주 지표를 내는 두 모드.
PRIMARY_LOW_MS = 100
KO_REGRESSION_MS = 50
TEXT_DIFFER_WARN = 2
AB_MODES = ("live", "conversation")
# 블록별 GPU 표에 옮길 GpuLog 요약 칸(stream_eval GpuLog.summary, 클럭 칸은 18절에 더함).
GPU_KEYS = (
    "samples",
    "mean_util",
    "p90_util",
    "max_memory_mb",
    "mean_cpu",
    "sm_clock_p50_mhz",
    "busy_sm_clock_p50_mhz",
    "memory_clock_p50_mhz",
    "pstates",
    "limited_ratio",
)


def tts_run_ms(item: dict) -> float | None:
    """마디 하나의 서버 합성 실행 시간(timings의 tts run_ms). 최종 결과·음성·서버 시각·합성 단계 중 하나라도
    없으면 None(짝에서 뺀다)."""
    timings = item.get("timings")
    if item.get("outcome") != "final" or item.get("audio_s") is None or not timings:
        return None
    stage = (timings.get("stages") or {}).get("tts")
    return None if stage is None else float(stage["run_ms"])


def _texts(item: dict) -> tuple:
    result = item.get("result") or {}
    return result.get("source_text"), result.get("translated_text")


def _blocks(reports: list[dict], mode: str, lang: str) -> list[list[dict]]:
    return [
        report["details"][lang]
        for report in reports
        if report.get("mode", "live") == mode and lang in report["details"]
    ]


def pair_mode(a_reports: list[dict], b_reports: list[dict], mode: str, lang: str = "en") -> dict:
    """한 모드에서 같은 클립끼리 서버 합성 실행 시간을 짝짓는다.

    입력: 조건 A·B의 블록 보고서들(다른 모드·언어의 블록은 건너뜀). 클립(key)마다 그 모드·언어의 모든
    블록을 보고,
    - 어느 블록에 없으면 missing(고른 클립이 블록마다 달라짐), 어느 블록에서라도 합성 시간을 쓸 수 없으면
      unusable(실패·음성 없음·timings 없음), 블록 사이에 서버의 최종 원문·번역문이 하나라도 다르면
      text_differs로 빼고 key를 적는다(입력 소리가 같아도 인식·번역이 다르면 합성할 글이 달라 짝이 아니다).
    - 남은 클립은 a(A 블록들의 평균), b(B 블록들의 평균)다. 동시통역은 블록이 조건마다 둘이라(A1·A2,
      B1·B2) 두 블록의 평균, 대화 모드는 하나씩이다.
    돌려주는 값: {"blocks": (A 블록 수, B 블록 수), "keys", "a", "b", "missing", "unusable",
    "text_differs"}. key 순서는 A 블록에서 처음 나온 순서다.
    """
    a_blocks, b_blocks = _blocks(a_reports, mode, lang), _blocks(b_reports, mode, lang)
    out: dict = {"blocks": (len(a_blocks), len(b_blocks)), "keys": [], "a": [], "b": []}
    out.update(missing=[], unusable=[], text_differs=[])
    if not a_blocks or not b_blocks:
        return out
    by_key = [{item["key"]: item for item in items} for items in a_blocks + b_blocks]
    for key in dict.fromkeys(item["key"] for items in a_blocks + b_blocks for item in items):
        found = [block.get(key) for block in by_key]
        if any(item is None for item in found):
            out["missing"].append(key)
            continue
        values = [tts_run_ms(item) for item in found]
        if any(value is None for value in values):
            out["unusable"].append(key)
            continue
        if len({_texts(item) for item in found}) > 1:
            out["text_differs"].append(key)
            continue
        out["keys"].append(key)
        out["a"].append(float(np.mean(values[: len(a_blocks)])))
        out["b"].append(float(np.mean(values[len(a_blocks) :])))
    return out


def pooled_rows(reports: list[dict]) -> list[dict]:
    """한 조건의 블록들을 모드마다 마디를 합쳐 17절과 같은 요약 행(summarize)으로 만든다. 동시통역 영어 원문은
    블록 둘(80마디)이 한 행이 된다."""
    merged: dict[str, dict] = {}
    for report in reports:
        mode = report.get("mode", "live")
        target = merged.setdefault(mode, {"mode": mode, "details": {}})
        for lang, items in report["details"].items():
            target["details"].setdefault(lang, []).extend(items)
    return [row for report in merged.values() for row in summarize(report)]


def stage_summary(record: dict | None, condition: str) -> dict | None:
    """melo-stages 보고서에서 한 조건("off"·"on")의 요약(summarize_stages 모양)을 꺼낸다.

    alternate 보고서는 두 조건이 다 있고, 조건 하나로 잰 보고서(17절은 칸이 없어 off)는 그 조건만 돌려주며
    다른 조건은 None이다. 보고서가 없으면 None.
    """
    if record is None:
        return None
    if record.get("regex_cache") == "alternate":
        return record["summary"].get(condition)
    return record["summary"] if record.get("regex_cache", "off") == condition else None


def decide_ab(primary_low: float | None, ko_regression: float | None, errors: int, text_differs: int) -> dict:
    """18절 채택 규칙(측정 전에 정함). 모두 참이어야 채택한다.

    (a) 주 지표 95% 구간 하한 > PRIMARY_LOW_MS(같으면 거짓) (b) 한국어 원문 말 끝 → 음성 p50 회귀(B − A)
    ≤ KO_REGRESSION_MS(같으면 참) (c) 오류 0(모든 블록). (d) GPU 없는 동일성 관문 통과는 측정 전에 확인하는
    전제라 이 계산 밖이다. 하나라도 거짓이면 채택하지 않고(False), 거짓은 없는데 모르는 값(None)이 있으면
    None(판단 못 함)이다. 짝에서 원문·번역문이 달라 뺀 영어 클립이 TEXT_DIFFER_WARN을 넘으면 정상성 경고만
    켠다.
    """
    a_ok = None if primary_low is None else primary_low > PRIMARY_LOW_MS
    b_ok = None if ko_regression is None else ko_regression <= KO_REGRESSION_MS
    c_ok = errors == 0
    conditions = (a_ok, b_ok, c_ok)
    adopt = False if False in conditions else None if None in conditions else True
    return {
        "a_primary_low_over_100": a_ok,
        "b_ko_regression_within_50": b_ok,
        "c_no_errors": c_ok,
        "adopt": adopt,
        "text_differs_warning": text_differs > TEXT_DIFFER_WARN,
    }


def _interval(groups: list[tuple[list[float], list[float]]]) -> dict:
    if not groups or not groups[0][0]:
        return {"point_ms": None, "low_ms": None, "high_ms": None}
    found = mean_difference([(np.asarray(a), np.asarray(b)) for a, b in groups])
    return {"point_ms": found.point, "low_ms": found.low, "high_ms": found.high}


def compare(a: list[tuple[str, dict]], b: list[tuple[str, dict]], stages: dict | None) -> dict:
    """18절 계산 전체.

    a·b는 (블록 이름, stream_eval service 보고서) 목록, stages는 melo-stages 보고서(없으면 None)다.
    - primary(주 지표): 영어 원문 블록이 두 조건에 다 있는 모드(AB_MODES 순서)마다 pair_mode로 짝짓고,
      모든 모드에 남은 클립만 쓴다. 값은 클립마다 모드별 차이(A − B)를 평균한 것의 클립 평균이고, 95%
      구간은 클립을 한 번 뽑아 모든 모드에 함께 쓰는 짝지은 재표본(significance.mean_difference, 13절과 같은
      10,000번·시드 12345)이다. 두 모드가 다 있을 때만 규칙 (a)에 쓴다
    - modes: 모드별 짝(그 모드에 남은 클립 전부)의 평균 차이와 구간, 차이 중앙값(적기만)
    - excluded: 모드를 합친 뺀 클립 key(missing·unusable·text_differs), errors: 모든 블록·언어에서 최종
      결과가 없거나 음성이 없는 마디 수
    - ko_regression_ms: 동시통역 한국어 원문 블록의 말 끝 → 음성 p50(조건마다 마디를 합침) B − A
    - rows·rule_iii: 조건마다 마디를 합친 17절 요약 행과, 그 행으로 다시 계산한 17절 판단(규칙 iii의 같은
      문턱, 조각 비율·캐시 비우기 몫은 stages의 그 조건 요약). melo_text_p50_ms: 조건별 melo-stages 전처리
      단계 p50
    - gpu: 블록마다 GpuLog 요약(사용률·SM 클럭·P-state·제한 이유)
    - decision: decide_ab
    """
    a_reports, b_reports = [report for _, report in a], [report for _, report in b]
    pairs = {mode: pair_mode(a_reports, b_reports, mode) for mode in AB_MODES}
    present = [mode for mode in AB_MODES if all(pairs[mode]["blocks"])]
    modes = {}
    for mode in present:
        pair = pairs[mode]
        differences = [x - y for x, y in zip(pair["a"], pair["b"], strict=True)]
        modes[mode] = {
            "blocks": list(pair["blocks"]),
            "clips": len(differences),
            **_interval([(pair["a"], pair["b"])]),
            "median_ms": median(differences),
        }
    common = (
        [key for key in pairs[present[0]]["keys"] if all(key in pairs[m]["keys"] for m in present)]
        if present
        else []
    )
    groups = []
    for mode in present:
        index = {key: position for position, key in enumerate(pairs[mode]["keys"])}
        groups.append(
            (
                [pairs[mode]["a"][index[key]] for key in common],
                [pairs[mode]["b"][index[key]] for key in common],
            )
        )
    primary = {"modes": present, "clips": len(common), **_interval(groups)}
    excluded = {
        reason: list(dict.fromkeys(key for mode in present for key in pairs[mode][reason]))
        for reason in ("missing", "unusable", "text_differs")
    }
    errors = sum(
        1
        for report in a_reports + b_reports
        for items in report["details"].values()
        for item in items
        if item.get("outcome") != "final" or item.get("audio_s") is None
    )

    def ko_p50(reports: list[dict]) -> float | None:
        parts = [
            part
            for items in _blocks(reports, "live", "ko")
            for item in items
            if (part := components(item)) is not None
        ]
        return median([part["audio"] for part in parts])

    ko_a, ko_b = ko_p50(a_reports), ko_p50(b_reports)
    ko_regression = None if ko_a is None or ko_b is None else ko_b - ko_a
    rows = {"A": pooled_rows(a_reports), "B": pooled_rows(b_reports)}
    summaries = {"A": stage_summary(stages, "off"), "B": stage_summary(stages, "on")}
    primary_low = primary["low_ms"] if set(present) == set(AB_MODES) else None
    return {
        "primary": primary,
        "modes": modes,
        "excluded": excluded,
        "errors": errors,
        "ko_audio_p50_ms": {"A": ko_a, "B": ko_b},
        "ko_regression_ms": ko_regression,
        "rows": rows,
        "rule_iii": {condition: decide(rows[condition], summaries[condition]) for condition in ("A", "B")},
        "melo_text_p50_ms": {
            condition: summary["stages"]["text"]["p50_ms"]
            if summary and "text" in summary["stages"]
            else None
            for condition, summary in summaries.items()
        },
        "gpu": [
            {
                "block": name,
                "condition": condition,
                "mode": report.get("mode", "live"),
                "langs": list(report["details"]),
                **{key: (report.get("gpu_use") or {}).get(key) for key in GPU_KEYS},
            }
            for condition, blocks in (("A", a), ("B", b))
            for name, report in blocks
        ],
        "decision": decide_ab(primary_low, ko_regression, errors, len(excluded["text_differs"])),
    }


def _interval_text(found: dict) -> str:
    if found["point_ms"] is None:
        return "-"
    return f"{found['point_ms']:.0f}ms [{found['low_ms']:.0f}, {found['high_ms']:.0f}]"


def write_ab_report(result: dict, args: argparse.Namespace) -> Path:
    """18절 계산 결과를 latency_ab_<tag>_<UTC 시각>.json(입력 경로와 결과 전체)과 .md(사람이 읽는 요약)로
    쓰고 .md 경로를 돌려준다."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = REPORTS / f"latency_ab_{args.tag}_{stamp}"
    record = {
        "tag": args.tag,
        "a": [str(path) for path in args.a],
        "b": [str(path) for path in args.b],
        "stages": str(args.stages) if args.stages else None,
        "result": result,
    }
    base.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    primary, decision, excluded = result["primary"], result["decision"], result["excluded"]
    ko = result["ko_audio_p50_ms"]
    lines = [
        f"# g2pkk 정규식 캐시 끔(A)·켬(B) 짝 비교 ({args.tag}, docs/experiments.md 18절)",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}",
        f"- A 블록: {', '.join(Path(path).stem for path in args.a)}",
        f"- B 블록: {', '.join(Path(path).stem for path in args.b)}",
        f"- 주 지표(영어 원문 마디의 서버 합성 실행 시간, 같은 클립끼리 A − B, 클립마다 모드 평균, 모드 "
        f"{', '.join(primary['modes']) or '-'}, 클립 {primary['clips']}개): {_interval_text(primary)}",
    ]
    for mode, found in result["modes"].items():
        lines.append(
            f"- {mode}(적기만): 블록 A {found['blocks'][0]}·B {found['blocks'][1]}, 클립 {found['clips']}개, "
            f"평균 차이 {_interval_text(found)}, 차이 중앙값 {_ms(found['median_ms'])}ms"
        )
    lines += [
        f"- 짝에서 뺀 영어 클립: 원문·번역문 다름 {excluded['text_differs']}, "
        f"쓸 수 없음(실패·음성 없음·timings 없음) {excluded['unusable']}, "
        f"어느 블록에 없음 {excluded['missing']}",
        f"- 오류(최종 결과 없음 또는 음성 없음, 모든 블록): {result['errors']}",
        f"- 동시통역 한국어 원문 말 끝 → 음성 p50: A {_ms(ko['A'])}ms, B {_ms(ko['B'])}ms, 회귀(B − A) "
        f"{_ms(result['ko_regression_ms'])}ms",
        f"- MeloTTS 전처리·BERT 단계 p50(melo-stages): A {_ms(result['melo_text_p50_ms']['A'])}ms, "
        f"B {_ms(result['melo_text_p50_ms']['B'])}ms",
        "",
        "## 18절 채택 규칙",
        "",
        f"- (a) 주 지표 구간 하한 {_ms(primary['low_ms'])}ms > {PRIMARY_LOW_MS}ms(두 모드 모두 있을 때만): "
        f"{decision['a_primary_low_over_100']}",
        f"- (b) 한국어 원문 회귀 {_ms(result['ko_regression_ms'])}ms ≤ +{KO_REGRESSION_MS}ms: "
        f"{decision['b_ko_regression_within_50']}",
        f"- (c) 오류 0: {decision['c_no_errors']}",
        "- (d) GPU 없는 동일성 관문: 측정 전에 확인하는 전제(이 계산 밖)",
        f"- 채택: {decision['adopt']}. 정상성 경고(원문·번역문이 달라 뺀 클립 {TEXT_DIFFER_WARN}개 초과): "
        f"{decision['text_differs_warning']}",
        "",
        "## 조건별 요약 (조건마다 블록의 마디를 합침, 값은 마디별 값의 중앙값 ms)",
        "",
        "| 조건 | " + " | ".join(title for title, _ in COLUMNS) + " |",
        "|" + "---|" * (len(COLUMNS) + 1),
    ]
    for condition in ("A", "B"):
        for row in result["rows"][condition]:
            lines.append(f"| {condition} | " + " | ".join(cell(row) for _, cell in COLUMNS) + " |")
    lines += ["", "## 17절 규칙 iii 다시 계산 (같은 문턱)", ""]
    for condition in ("A", "B"):
        found = result["rule_iii"][condition]
        values = found["values"]
        lines.append(
            f"- {condition}: 영어 원문 합성 p50 {_ms(values['en_source_tts_p50_ms'])}ms ≥ {KO_TTS_MS}ms, "
            f"조각 2개 이상 {_ratio(values['multi_piece_ratio'])} ≥ {MULTI_PIECE:.0%}, "
            f"처리가 확정보다 늦음(나쁜 모드) {_ratio(values['en_source_processing_bound_max'])} ≥ "
            f"{PROCESSING_BOUND:.0%} → A3를 열 근거: {found['iii_open_a3']}"
        )
    lines += [
        "",
        "## 블록별 GPU (측정 중 0.5초 샘플)",
        "",
        "| 블록 | 조건 | 모드 | 원문 | 샘플 | 사용률 평균 | SM 클럭 중앙값 | 사용 중 SM 클럭 | 메모리 클럭 | "
        "P-state 샘플 수 | 제한 이유 샘플 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for gpu in result["gpu"]:
        pstates = ", ".join(f"{name} {count}" for name, count in (gpu["pstates"] or {}).items()) or "-"
        util = "-" if gpu["mean_util"] is None else f"{gpu['mean_util']:.0f}%"
        samples = "-" if gpu["samples"] is None else gpu["samples"]
        lines.append(
            f"| {gpu['block']} | {gpu['condition']} | {gpu['mode']} | {', '.join(gpu['langs'])} | "
            f"{samples} | {util} | {_ms(gpu['sm_clock_p50_mhz'])} | {_ms(gpu['busy_sm_clock_p50_mhz'])} | "
            f"{_ms(gpu['memory_clock_p50_mhz'])} | {pstates} | {_ratio(gpu['limited_ratio'])} |"
        )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".md")


# 명령줄. 17절 형태(--stream)와 18절 형태(--a·--b)를 함께 받되 섞지 않는다. --stages는 두 형태 모두에서 쓴다.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="저장된 서비스 측정 보고서로 말 끝 → 번역 음성을 단계별로 나누고 17절 규칙을 적용한다"
        "(--stream). 또는 g2pkk 정규식 캐시를 끈(A)·켠(B) 서버의 블록 보고서를 같은 클립끼리 짝지어"
        " 18절 규칙을 적용한다(--a·--b)",
    )
    parser.add_argument(
        "--stream",
        type=Path,
        nargs="+",
        help="17절: LIVE_TIMINGS를 켠 서버로 잰 stream_eval service 보고서(JSON). 모드마다 하나",
    )
    parser.add_argument(
        "--a",
        type=Path,
        nargs="+",
        help="18절: MELO_REGEX_CACHE=false(A) 서버로 잰 블록 보고서들(JSON). 동시통역 영어 둘, 한국어 하나,"
        " 대화 모드 영어 하나",
    )
    parser.add_argument(
        "--b",
        type=Path,
        nargs="+",
        help="18절: MELO_REGEX_CACHE=true(B) 서버로 잰 블록 보고서들(JSON). --a와 같은 구성",
    )
    parser.add_argument(
        "--stages",
        type=Path,
        help="tts_eval melo-stages 보고서(JSON). 17절: 없으면 iii·iv는 판단 못 함. 18절: --regex-cache"
        " alternate 보고서(조건별 조각 비율·전처리 단계)",
    )
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if bool(args.stream) == bool(args.a or args.b) or bool(args.a) != bool(args.b):
        parser.error("17절은 --stream만, 18절은 --a와 --b를 함께 준다")
    record = json.loads(args.stages.read_text(encoding="utf-8")) if args.stages else None
    if args.stream:
        rows = [
            row for path in args.stream for row in summarize(json.loads(path.read_text(encoding="utf-8")))
        ]
        stages = record["summary"] if record else None
        print(f"written to {write_report(rows, decide(rows, stages), stages, args)}")
        return

    def load(paths: list[Path]) -> list[tuple[str, dict]]:
        return [(path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in paths]

    print(f"written to {write_ab_report(compare(load(args.a), load(args.b), record), args)}")


if __name__ == "__main__":
    main()
