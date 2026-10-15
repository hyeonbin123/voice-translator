"""말을 멈춘 뒤 번역 음성까지의 시간을 단계별로 나누고, 측정 전에 정한 규칙으로 다음 작업을 고른다.

  uv run python -m eval.latency_breakdown --stream <live 보고서> <conversation 보고서> \\
      --stages <tts melo-stages 보고서> --tag t95_dev

docs/experiments.md 17절(측정 전에 커밋)의 진단이다. 입력은 서버를 LIVE_TIMINGS=true로 띄우고
`eval.stream_eval service`로 잰 보고서(final마다 서버의 단계별 시각이 들어 있음)와, 같은 번역문을
`eval.tts_eval melo-stages`로 합성한 MeloTTS 단계별 시간이다. 측정이 아니라 저장된 보고서를 읽어 계산만 한다.
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

from eval.common import REPORTS

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


def decide(rows: list[dict], stages: dict | None) -> dict:
    """17절 판단 규칙. rows는 summarize의 결과(두 모드), stages는 melo-stages 보고서의 summary(없으면 None).

    - i 동시통역 영어 원문의 말 끝 → 음성 p50이 GOAL_MS 이하면 '이미 목표 안'(지연 개선 작업을 열지 않음.
      README 수치를 test로 다시 잴지는 사용자 결정)
    - ii 동시통역에서 최종 처리의 첫 단계 대기 p50이 어느 원문 언어든 QUEUE_MS 이상이면 A9를 연다
    - iii 영어 원문(한국어 합성)의 합성 p50이 KO_TTS_MS 이상이고, 조각 2개 이상인 번역문이 MULTI_PIECE
      이상이며, 두 모드 중 어느 하나에서라도 처리가 1초 확정보다 늦게 끝난 마디가 PROCESSING_BOUND 이상이면
      A3를 연다(사용자가 후보 C를 열기로 할 때. 조각을 먼저 보내는 것은 처리가 확정보다 늦을 때만 돕는다)
    - iv MeloTTS 합성에서 캐시 비우기가 CACHE_SHARE 이상이면 A3에 캐시 비우기 생략 변형을 넣는다
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
    rule_iii = None
    if tts is not None and multi is not None and bound:
        rule_iii = tts >= KO_TTS_MS and multi >= MULTI_PIECE and max(bound) >= PROCESSING_BOUND
    return {
        "i_live_en_within_goal": None if live_en is None else live_en <= GOAL_MS,
        "ii_open_a9": None if not waits else max(waits) >= QUEUE_MS,
        "iii_open_a3": rule_iii,
        "iv_a3_skip_cache_variant": None if cache is None else cache >= CACHE_SHARE,
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
        f"- ii 동시통역 첫 대기 p50(나쁜 쪽) {_ms(values['live_first_wait_p50_ms'])}ms ≥ {QUEUE_MS}ms "
        f"→ A9 열기: {decision['ii_open_a9']}",
        f"- iii 영어 원문 합성 p50 {_ms(values['en_source_tts_p50_ms'])}ms ≥ {KO_TTS_MS}ms, 조각 2개 이상 "
        f"{_ratio(values['multi_piece_ratio'])} ≥ {MULTI_PIECE:.0%}, 처리가 확정보다 늦음(나쁜 모드) "
        f"{_ratio(values['en_source_processing_bound_max'])} ≥ {PROCESSING_BOUND:.0%} → A3 열기: "
        f"{decision['iii_open_a3']}",
        f"- iv 캐시 비우기 몫 {_ratio(values['empty_cache_share'])} ≥ {CACHE_SHARE:.0%} → A3에 생략 변형: "
        f"{decision['iv_a3_skip_cache_variant']}",
    ]
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="저장된 서비스 측정 보고서로 말 끝 → 번역 음성을 단계별로 나누고 17절 규칙을 적용한다",
    )
    parser.add_argument(
        "--stream",
        type=Path,
        nargs="+",
        required=True,
        help="LIVE_TIMINGS를 켠 서버로 잰 stream_eval service 보고서(JSON). 모드마다 하나",
    )
    parser.add_argument(
        "--stages", type=Path, help="tts_eval melo-stages 보고서(JSON). 없으면 iii·iv는 판단 못 함"
    )
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    rows = [row for path in args.stream for row in summarize(json.loads(path.read_text(encoding="utf-8")))]
    stages = json.loads(args.stages.read_text(encoding="utf-8"))["summary"] if args.stages else None
    print(f"written to {write_report(rows, decide(rows, stages), stages, args)}")


if __name__ == "__main__":
    main()
