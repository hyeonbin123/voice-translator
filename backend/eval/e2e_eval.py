"""End-to-end latency of the running server with the real models (task T11).

1. prepare, with the eval group (reads the FLEURS parquet files):
       uv run python -m eval.e2e_eval prepare --split validation
   writes data/e2e/<split>/<language>_<id>.wav and manifest.json: per language, in ID order, the first
   30 recordings that last 8 to 12 seconds, one per sentence ID.
2. run, with any environment that has httpx, against a server started with LOAD_MODELS=true:
       uv run python -m eval.e2e_eval run --split validation --concurrency 1 --tag t11_dev_threads1

The settings compared and the rule are written in docs/experiments.md before any run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
import uuid
from datetime import UTC, datetime

import httpx

from eval.common import DATA, FLEURS_CONFIG, REPORTS

E2E = DATA / "e2e"
PER_LANGUAGE = 30
MIN_SECONDS, MAX_SECONDS = 8.0, 12.0
SAMPLE_RATE = 16_000


def prepare(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    out = E2E / args.split
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for language, config in FLEURS_CONFIG.items():
        path = DATA / "fleurs" / config / f"{args.split}.parquet"
        rows = pq.read_table(path, columns=["id", "num_samples", "audio"]).to_pylist()
        chosen, seen = [], set()
        for row in sorted(rows, key=lambda r: r["id"]):  # stable: keeps file order within one ID
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


def gpu_used_mib() -> int | None:
    query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    try:
        return int(subprocess.run(query, capture_output=True, text=True, check=True).stdout.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def percentile_95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20)[18] if len(values) >= 2 else values[0]


async def run(args: argparse.Namespace) -> None:
    folder = E2E / args.split
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    async with httpx.AsyncClient(base_url=args.base_url, timeout=300) as client:
        # A throwaway account; its records are deleted at the end.
        email, password = f"e2e-{uuid.uuid4().hex[:12]}@example.com", uuid.uuid4().hex
        account = {"email": email, "password": password}
        (await client.post("/api/auth/register", json=account)).raise_for_status()
        login = await client.post("/api/auth/login", data={"username": email, "password": password})
        headers = {"Authorization": f"Bearer {login.raise_for_status().json()['access_token']}"}

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
        warmups = [await translate(item) for item in first_of.values()]  # not recorded
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


async def memory(args: argparse.Namespace) -> None:
    """T24: server memory across batches after a fixed warm-up (rule in docs/experiments.md 5절).

    Warm-up: 10 sequential requests (5 per language) and 5 concurrent ko+en pairs. Baseline: 5 s later, the
    median of 3 readings 1 s apart. Then `--batches` batches of 120 requests (the 60 recordings one by one,
    then 30 concurrent ko+en pairs), each followed by the same settle-and-read step. Health is polled every
    0.2 s and memory every 2 s throughout; every raw sample is saved so the tables can be recomputed.
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

    def read_memory() -> dict:
        info = server.memory_info()
        private = getattr(info, "private", info.rss)  # Windows reports private bytes; elsewhere fall back
        return {
            "t": clock(),
            "working_set_mb": round(info.rss / 2**20, 1),
            "private_mb": round(private / 2**20, 1),
        }

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

        async def send(item: dict, phase: str, concurrent: bool) -> None:
            target = "en" if item["language"] == "ko" else "ko"
            files = {"audio": (item["file"], (folder / item["file"]).read_bytes(), "audio/wav")}
            data = {"source_lang": item["language"], "target_lang": target}
            sent = clock()
            reply = await client.post("/api/translate/speech", headers=headers, files=files, data=data)
            is_json = reply.headers.get("content-type", "").startswith("application/json")
            body = reply.json() if is_json else {}
            requests.append(
                {"phase": phase, "concurrent": concurrent, "file": item["file"], "t": sent,
                 "elapsed_s": round(clock() - sent, 3), "status": reply.status_code, "id": body.get("id"),
                 "stt_ms": body.get("stt_ms"), "mt_ms": body.get("mt_ms"), "tts_ms": body.get("tts_ms")}
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
    payload = {"summary": summary, "checkpoints": checkpoints, "requests": requests, "health": health,
               "memory_trace": memory_trace}
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


def main() -> None:
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    prep = phases.add_parser("prepare", help="extract the recordings from FLEURS")
    prep.add_argument("--split", choices=["validation", "test"], default="validation")
    go = phases.add_parser("run", help="send them to a running server")
    go.add_argument("--split", choices=["validation", "test"], default="validation")
    go.add_argument("--base-url", default="http://127.0.0.1:8000")
    go.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    go.add_argument("--tag", default="run")
    mem = phases.add_parser("memory", help="server memory across batches after a warm-up (T24)")
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
