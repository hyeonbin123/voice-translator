"""실제 모델을 올리고 실행 중인 서버의 전체 흐름 지연을 잰다.

1. prepare, eval 그룹 환경에서(FLEURS parquet 파일을 읽음):
       uv run python -m eval.e2e_eval prepare --split validation
   data/e2e/<split>/<language>_<id>.wav와 manifest.json을 쓴다: 언어마다 ID 순으로, 8~12초 길이의
   녹음을 문장 ID당 하나씩 처음 30개.
2. run, httpx가 있는 아무 환경에서나, LOAD_MODELS=true로 띄운 서버를 대상으로:
       uv run python -m eval.e2e_eval run --split validation --concurrency 1 --tag t11_dev_threads1

비교한 설정과 판단 규칙은 실행 전에 docs/experiments.md에 적었다.
"""

# [파일 역할] 실제 모델을 올린 서버에 FLEURS 음성을 보내 전체 흐름(업로드 → 인식 → 번역 → 합성 → 저장)의 지연,
# 번역 중 다른 요청(/api/health)의 응답 시간, GPU 메모리, 서버 메모리 증가를 재는 측정 도구.
# [설계 이유] 모델 스레드 수(MODEL_THREADS) 결정(docs/experiments.md 4절), 번역 중 다른 요청이 막히던 문제와
# gc.freeze 확인·메모리 측정(5절), 이후 시험들의 '서버 회귀' 확인(15절 등)을 같은 도구로 한다. 측정 데이터는
# 8~12초 음성으로 고정해 언어별 30개를 뽑고(README: 10초 음성 → 번역 음성까지 중앙값 0.89초), 측정용 임시
# 계정의 기록은 끝에 지운다.
# [흐름] prepare(FLEURS에서 녹음을 골라 WAV와 목록 저장) → run(동시 요청 수별 지연·health·VRAM) 또는
# memory(정해진 예열 뒤 묶음마다 서버 메모리).

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
import uuid
from datetime import UTC, datetime

# httpx: 서버에 비동기로 요청을 보낸다(run은 httpx만 있으면 어느 환경에서나 돈다). pyarrow·psutil은 쓰는 명령
# 안에서만 import한다.
import httpx

from eval.common import DATA, FLEURS_CONFIG, REPORTS

# E2E: 고른 녹음을 두는 폴더(data/e2e/<분할>). 언어별 30개, 길이 8~12초, FLEURS 표본 추출률 16kHz.
E2E = DATA / "e2e"
PER_LANGUAGE = 30
MIN_SECONDS, MAX_SECONDS = 8.0, 12.0
SAMPLE_RATE = 16_000


# 언어마다 FLEURS 행을 ID 순으로 보며 길이가 8~12초인 녹음을 문장 ID당 하나씩, 앞에서부터 30개 고른다(같은
# 문장을 여러 사람이 읽으므로 중복 제거). 원본 음성 바이트를 그대로 WAV로 쓰고 목록(manifest.json)을 남겨,
# run이 매번 같은 녹음을 같은 순서로 보낸다.
def prepare(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    out = E2E / args.split
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for language, config in FLEURS_CONFIG.items():
        path = DATA / "fleurs" / config / f"{args.split}.parquet"
        rows = pq.read_table(path, columns=["id", "num_samples", "audio"]).to_pylist()
        chosen, seen = [], set()
        for row in sorted(rows, key=lambda r: r["id"]):  # 안정 정렬: 한 ID 안에서는 파일 순서를 지킨다
            seconds = row["num_samples"] / SAMPLE_RATE
            if row["id"] in seen or not MIN_SECONDS <= seconds <= MAX_SECONDS:
                continue
            seen.add(row["id"])
            chosen.append(row)
            if len(chosen) == PER_LANGUAGE:
                break
        for row in chosen:
            name = f"{language}_{row['id']}.wav"
            (out / name).write_bytes(row["audio"]["bytes"])
            seconds = round(row["num_samples"] / SAMPLE_RATE, 2)
            manifest.append({"file": name, "language": language, "id": row["id"], "seconds": seconds})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(manifest)} recordings written to {out}")


# nvidia-smi로 GPU 전체의 사용 중 메모리(MiB)를 읽는다. 명령이 없거나 실패하면 None.
def gpu_used_mib() -> int | None:
    query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    try:
        return int(subprocess.run(query, capture_output=True, text=True, check=True).stdout.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


# 95번째 백분위: quantiles(n=20)의 19개 경계 중 마지막(인덱스 18). 값이 하나면 그 값.
def percentile_95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20)[18] if len(values) >= 2 else values[0]


# 측정용 임시 계정을 만들어 로그인한 뒤, 언어별 첫 녹음으로 한 번씩 예열(기록하지 않음)하고, 목록 전체를 동시
# 요청 수(1 또는 2)만큼씩 묶어 보낸다. 그동안 watch가 0.2초마다 /api/health 응답 시간과 GPU 메모리를
# 잰다(health가 200이 아니면 무한대로 적어 p95에 반영). 끝나면 만든 기록을 모두 지운다.
async def run(args: argparse.Namespace) -> None:
    folder = E2E / args.split
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    async with httpx.AsyncClient(base_url=args.base_url, timeout=300) as client:
        # 한 번 쓰고 버리는 계정. 그 기록은 끝에 지운다.
        email, password = f"e2e-{uuid.uuid4().hex[:12]}@example.com", uuid.uuid4().hex
        account = {"email": email, "password": password}
        (await client.post("/api/auth/register", json=account)).raise_for_status()
        login = await client.post("/api/auth/login", data={"username": email, "password": password})
        headers = {"Authorization": f"Bearer {login.raise_for_status().json()['access_token']}"}

        # 녹음 하나를 음성 번역 API로 보내고 벽시계 지연과 응답의 단계별 시간(stt/mt/tts_ms)·합성
        # 오류·원문·번역문을 결과에 붙인다. JSON이 아닌 오류 응답도 결과로 남긴다.
        async def translate(item: dict) -> dict:
            target = "en" if item["language"] == "ko" else "ko"
            files = {"audio": (item["file"], (folder / item["file"]).read_bytes(), "audio/wav")}
            data = {"source_lang": item["language"], "target_lang": target}
            start = time.perf_counter()
            reply = await client.post("/api/translate/speech", headers=headers, files=files, data=data)
            elapsed = time.perf_counter() - start
            is_json = reply.headers.get("content-type", "").startswith("application/json")
            body = reply.json() if is_json else {}
            fields = ("id", "stt_ms", "mt_ms", "tts_ms", "tts_error", "source_text", "translated_text")
            return {**item, "status": reply.status_code, "elapsed_s": round(elapsed, 3)} | {
                key: body.get(key) for key in fields
            }

        first_of = {lang: next(i for i in manifest if i["language"] == lang) for lang in ("ko", "en")}
        warmups = [await translate(item) for item in first_of.values()]  # 기록하지 않음
        health: list[float] = []
        vram = [gpu_used_mib()]
        done = asyncio.Event()

        async def watch() -> None:
            while not done.is_set():
                start = time.perf_counter()
                reply = await client.get("/api/health")
                health.append(time.perf_counter() - start if reply.status_code == 200 else float("inf"))
                vram.append(await asyncio.to_thread(gpu_used_mib))
                await asyncio.sleep(0.2)

        watcher = asyncio.create_task(watch())
        results: list[dict] = []
        for start in range(0, len(manifest), args.concurrency):
            batch = manifest[start : start + args.concurrency]
            results += await asyncio.gather(*(translate(item) for item in batch))
        done.set()
        await watcher
        for item in warmups + results:
            if item.get("id"):
                await client.delete(f"/api/history/{item['id']}", headers=headers)

    write_report(args, results, health, [v for v in vram if v is not None])


# 성공(201)한 요청으로 언어별·전체 지연 p50/p95와 단계별 시간 중앙값을 내고, 서버 오류(5xx)·그 밖의 실패·합성
# 실패 수, health p95·최대, VRAM 시작·최고값과 함께 JSON(요청별 원자료 포함)과 마크다운 표로 남긴다.
def write_report(args: argparse.Namespace, results: list[dict], health: list[float], vram: list[int]) -> None:
    ok = [r for r in results if r["status"] == 201]
    summary: dict = {
        "split": args.split,
        "concurrency": args.concurrency,
        "requests": len(results),
        "server_errors": sum(r["status"] >= 500 for r in results),
        "other_failures": sum(r["status"] not in (201,) and r["status"] < 500 for r in results),
        "tts_errors": sum(bool(r.get("tts_error")) for r in ok),
        "health_p95_s": percentile_95(health) if health else None,
        "health_max_s": max(health) if health else None,
        "vram_mib_first": vram[0] if vram else None,
        "vram_mib_peak": max(vram) if vram else None,
        "languages": {},
    }
    for language in ("ko", "en", "all"):
        rows = [r for r in ok if language == "all" or r["language"] == language]
        if not rows:
            continue
        summary["languages"][language] = {
            "count": len(rows),
            "audio_s_median": statistics.median(r["seconds"] for r in rows),
            "elapsed_p50_s": statistics.median(r["elapsed_s"] for r in rows),
            "elapsed_p95_s": percentile_95([r["elapsed_s"] for r in rows]),
            **{
                f"{stage}_p50_ms": statistics.median(r[stage] for r in rows if r[stage] is not None)
                for stage in ("stt_ms", "mt_ms", "tts_ms")
                if any(r[stage] is not None for r in rows)
            },
        }
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = REPORTS / f"e2e_{args.tag}_{stamp}"
    payload = {"summary": summary, "results": results}
    base.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# 전체 흐름 측정 ({args.tag})",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}",
        f"- 데이터: FLEURS {args.split}, 8~12초 음성 언어별 {PER_LANGUAGE}개",
        f"- 동시 요청 {args.concurrency}개씩",
        "- 전체 지연: 클라이언트가 요청을 보낸 뒤 응답을 받을 때까지(업로드·디코딩·인식·번역·합성·저장 포함)",
        f"- 요청 {summary['requests']}개, 서버 오류(5xx) {summary['server_errors']}개, 그 밖의 실패 "
        f"{summary['other_failures']}개, 합성 실패 {summary['tts_errors']}개",
        f"- 번역 중 health 응답: p95 {summary['health_p95_s']:.3f}초, 최대 {summary['health_max_s']:.3f}초"
        if health
        else "- 번역 중 health 응답: 기록 없음",
        f"- VRAM(MiB, GPU 전체): 시작 {summary['vram_mib_first']}, 최고 {summary['vram_mib_peak']}",
        "",
        "| 언어 | 개수 | 음성 길이 중앙값 | 전체 지연 p50 | p95 | 인식 p50 | 번역 p50 | 합성 p50 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for language, values in summary["languages"].items():
        stages = " | ".join(
            f"{values.get(f'{stage}_p50_ms', '-')}ms" for stage in ("stt_ms", "mt_ms", "tts_ms")
        )
        lines.append(
            f"| {language} | {values['count']} | {values['audio_s_median']:.1f}초 "
            f"| {values['elapsed_p50_s']:.2f}초 | {values['elapsed_p95_s']:.2f}초 | {stages} |"
        )
    report = base.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {report}")


# 서버 프로세스(PID로 지정)의 메모리를 psutil로 읽는다. 순서와 기준은 아래 docstring. 예열 뒤 기준점과 마지막
# 묶음 뒤의 전용 메모리 차이, 두 번째 묶음 뒤부터 마지막까지의 차이(초기화 뒤에도 계속 느는지)를 낸다. README
# 알려진 한계의 '메모리 기준 미달'이 이 측정 결과다.
async def memory(args: argparse.Namespace) -> None:
    """정해진 예열 뒤 묶음마다 서버 메모리를 잰다(판단 규칙은 docs/experiments.md 5절).

    예열: 순차 요청 10개(언어별 5개)와 한·영 동시 요청 5쌍. 기준점: 5초 뒤 1초 간격으로 세 번 읽은 값의
    중앙값. 그다음 120개 요청 묶음을 `--batches`번 보낸다(녹음 60개를 하나씩, 이어서 한·영 동시 요청
    30쌍). 묶음마다 같은 방식으로 기다렸다가 읽는다. 처음부터 끝까지 health는 0.2초마다, 메모리는 2초마다
    재고, 표를 다시 계산할 수 있게 원자료를 모두 저장한다.
    """
    import psutil

    folder = E2E / args.split
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    korean = [item for item in manifest if item["language"] == "ko"]
    english = [item for item in manifest if item["language"] == "en"]
    server = psutil.Process(args.server_pid)
    started = time.perf_counter()

    def clock() -> float:
        return round(time.perf_counter() - started, 3)

    # 작업 집합(rss)과 전용 메모리(Windows의 private bytes, 없으면 rss)를 MB로 읽는다.
    def read_memory() -> dict:
        info = server.memory_info()
        private = getattr(info, "private", info.rss)  # Windows는 private bytes, 그 밖은 rss
        return {
            "t": clock(),
            "working_set_mb": round(info.rss / 2**20, 1),
            "private_mb": round(private / 2**20, 1),
        }

    # 5초 기다린 뒤 1초 간격으로 세 번 읽어 중앙값을 낸다(한 번 읽기의 흔들림을 줄임).
    async def settle(label: str) -> dict:
        await asyncio.sleep(5)
        readings = []
        for _ in range(3):
            readings.append(await asyncio.to_thread(read_memory))
            await asyncio.sleep(1)
        return {
            "label": label,
            "readings": readings,
            "working_set_mb": statistics.median(r["working_set_mb"] for r in readings),
            "private_mb": statistics.median(r["private_mb"] for r in readings),
        }

    requests: list[dict] = []
    health: list[dict] = []
    memory_trace: list[dict] = []
    done = asyncio.Event()
    async with httpx.AsyncClient(base_url=args.base_url, timeout=300) as client:
        email, password = f"e2e-{uuid.uuid4().hex[:12]}@example.com", uuid.uuid4().hex
        account = {"email": email, "password": password}
        (await client.post("/api/auth/register", json=account)).raise_for_status()
        login = await client.post("/api/auth/login", data={"username": email, "password": password})
        headers = {"Authorization": f"Bearer {login.raise_for_status().json()['access_token']}"}

        # 요청 하나를 보내고 단계(예열·묶음 번호), 동시 요청 여부, 보낸 시각, 지연, 상태, 단계별 시간을
        # 기록한다.
        async def send(item: dict, phase: str, concurrent: bool) -> None:
            target = "en" if item["language"] == "ko" else "ko"
            files = {"audio": (item["file"], (folder / item["file"]).read_bytes(), "audio/wav")}
            data = {"source_lang": item["language"], "target_lang": target}
            sent = clock()
            reply = await client.post("/api/translate/speech", headers=headers, files=files, data=data)
            is_json = reply.headers.get("content-type", "").startswith("application/json")
            body = reply.json() if is_json else {}
            requests.append(
                {
                    "phase": phase,
                    "concurrent": concurrent,
                    "file": item["file"],
                    "t": sent,
                    "elapsed_s": round(clock() - sent, 3),
                    "status": reply.status_code,
                    "id": body.get("id"),
                    "stt_ms": body.get("stt_ms"),
                    "mt_ms": body.get("mt_ms"),
                    "tts_ms": body.get("tts_ms"),
                }
            )

        async def watch_health() -> None:
            while not done.is_set():
                sent = clock()
                reply = await client.get("/api/health")
                health.append({"t": sent, "seconds": round(clock() - sent, 4), "status": reply.status_code})
                await asyncio.sleep(0.2)

        async def watch_memory() -> None:
            while not done.is_set():
                memory_trace.append(await asyncio.to_thread(read_memory))
                await asyncio.sleep(2)

        # health는 0.2초, 메모리는 2초마다 계속 잰다. 예열: 순차 10개(언어별 5) + 한·영 동시 5쌍 → 기준점.
        # 묶음마다 60개를 하나씩 + 30쌍을 동시에(120개) → 기준점과 같은 방식으로 읽기.
        watchers = [asyncio.create_task(watch_health()), asyncio.create_task(watch_memory())]
        start_reading = await asyncio.to_thread(read_memory)
        for item in korean[:5] + english[:5]:
            await send(item, "warmup", False)
        for pair in zip(korean[5:10], english[5:10], strict=True):
            await asyncio.gather(*(send(item, "warmup", True) for item in pair))
        checkpoints = [await settle("baseline")]
        for batch in range(1, args.batches + 1):
            phase = f"batch{batch}"
            for item in manifest:
                await send(item, phase, False)
            for pair in zip(korean, english, strict=True):
                await asyncio.gather(*(send(item, phase, True) for item in pair))
            checkpoints.append(await settle(f"after {phase}"))
        done.set()
        await asyncio.gather(*watchers)
        for request in requests:
            if request.get("id"):
                await client.delete(f"/api/history/{request['id']}", headers=headers)

    # 요약은 기준점 이후(예열 제외)의 요청과 health만으로 낸다. checkpoints[2]는 두 번째 묶음 뒤의 읽기다.
    measured_from = checkpoints[0]["readings"][-1]["t"]
    measured = [r for r in requests if r["phase"] != "warmup"]
    sequential = [r["elapsed_s"] for r in measured if not r["concurrent"] and r["status"] == 201]
    health_measured = [h["seconds"] for h in health if h["t"] >= measured_from and h["status"] == 200]
    base, last = checkpoints[0], checkpoints[-1]
    second = checkpoints[2] if len(checkpoints) > 2 else checkpoints[-1]
    summary = {
        "server_pid": args.server_pid,
        "split": args.split,
        "batches": args.batches,
        "requests_measured": len(measured),
        "server_errors": sum(r["status"] >= 500 for r in requests),
        "other_failures": sum(r["status"] != 201 and r["status"] < 500 for r in requests),
        "health_failures": sum(h["status"] != 200 for h in health),
        "sequential_elapsed_p50_s": statistics.median(sequential) if sequential else None,
        "health_p95_s": percentile_95(health_measured) if health_measured else None,
        "memory_at_start": start_reading,
        "checkpoints": [{k: v for k, v in c.items() if k != "readings"} for c in checkpoints],
        "private_growth_mb": round(last["private_mb"] - base["private_mb"], 1),
        "private_growth_batch2_to_last_mb": round(last["private_mb"] - second["private_mb"], 1),
        "working_set_growth_mb": round(last["working_set_mb"] - base["working_set_mb"], 1),
    }
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base_path = REPORTS / f"e2e_{args.tag}_{stamp}"
    payload = {
        "summary": summary,
        "checkpoints": checkpoints,
        "requests": requests,
        "health": health,
        "memory_trace": memory_trace,
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=1)
    base_path.with_suffix(".json").write_text(raw, encoding="utf-8")
    lines = [
        f"# 메모리 측정 ({args.tag})",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}, 서버 PID {args.server_pid}",
        f"- 준비 20개 뒤 기준점, 120개 묶음 {args.batches}번. 측정 요청 {len(measured)}개",
        f"- 서버 오류 {summary['server_errors']}, 그 밖의 실패 {summary['other_failures']}, "
        f"health 실패 {summary['health_failures']}",
        f"- 순차 요청 전체 지연 중앙값 {summary['sequential_elapsed_p50_s']:.2f}초, "
        f"측정 구간 health p95 {summary['health_p95_s']:.3f}초",
        f"- 전용 메모리 증가: 기준점→마지막 {summary['private_growth_mb']}MB, "
        f"2번째 묶음 뒤→마지막 {summary['private_growth_batch2_to_last_mb']}MB. "
        f"작업 집합 증가 {summary['working_set_growth_mb']}MB",
        "",
        "| 시점 | 작업 집합(MB) | 전용 메모리(MB) |",
        "|---|---|---|",
        f"| 서버 시작 뒤 첫 요청 전 | {start_reading['working_set_mb']} | {start_reading['private_mb']} |",
    ]
    lines += [f"| {c['label']} | {c['working_set_mb']} | {c['private_mb']} |" for c in checkpoints]
    report = base_path.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {report}")


# 하위 명령 prepare / run / memory. 서버 주소 기본값은 run이 8000, memory가 8010이다.
def main() -> None:
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    prep = phases.add_parser("prepare", help="FLEURS에서 녹음을 뽑아 둔다")
    prep.add_argument("--split", choices=["validation", "test"], default="validation")
    go = phases.add_parser("run", help="뽑은 녹음을 실행 중인 서버에 보낸다")
    go.add_argument("--split", choices=["validation", "test"], default="validation")
    go.add_argument("--base-url", default="http://127.0.0.1:8000")
    go.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    go.add_argument("--tag", default="run")
    mem = phases.add_parser("memory", help="예열 뒤 묶음마다 서버 메모리를 잰다")
    mem.add_argument("--split", choices=["validation", "test"], default="validation")
    mem.add_argument("--base-url", default="http://127.0.0.1:8010")
    mem.add_argument("--server-pid", type=int, required=True)
    mem.add_argument("--batches", type=int, default=4)
    mem.add_argument("--tag", default="memory")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "memory":
        asyncio.run(memory(args))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
