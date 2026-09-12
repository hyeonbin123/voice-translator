"""Recompute the error rates of a saved speech recognition report with the current normalization (task T15).

Usage, from backend/ with the eval group installed:
    uv run python -m eval.stt_rescore eval/reports/stt_t2_dev_20260912_223312.json

Writes <report>_rescored.json and .md next to the original, which stays as it was. Each item keeps the
normalized reference and hypothesis, and normalizing is idempotent apart from the fix being applied
(Korean apostrophes), so normalizing them again gives the corrected text without running the models.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from eval.stt_eval import error_rate
from eval.text_norm import normalize


def rescore(results: list[dict]) -> list[dict]:
    for result in results:
        for language, values in result["languages"].items():
            for item in values["items"]:
                item["reference"] = normalize(item["reference"], language)
                item["hypothesis"] = normalize(item["hypothesis"], language)
                item["error"] = round(error_rate(language, [item["reference"]], [item["hypothesis"]]), 4)
            values["error_before_rescore"] = values["error"]
            values["error"] = error_rate(
                language,
                [item["reference"] for item in values["items"]],
                [item["hypothesis"] for item in values["items"]],
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="a stt_*.json report written by eval.stt_eval")
    args = parser.parse_args()

    results = rescore(json.loads(args.report.read_text(encoding="utf-8")))
    base = args.report.with_name(args.report.stem + "_rescored")
    base.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# 음성 인식 평가 재계산 ({args.report.name})",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}",
        "- 모델을 다시 돌리지 않고, 저장된 결과에 현재 정규화(`eval/text_norm.py`)를 다시 적용해"
        " 오류율만 다시 계산",
        "",
        "| 모델 | 한국어 CER (전 → 후) | 영어 WER (전 → 후) | 평균 (후) |",
        "|---|---|---|---|",
    ]
    for result in results:
        by_language = result["languages"]
        cells = [
            f"{values['error_before_rescore']:.2%} → {values['error']:.2%}"
            if (values := by_language.get(language))
            else "-"
            for language in ("ko", "en")
        ]
        mean = statistics.mean(values["error"] for values in by_language.values())
        lines.append(f"| {result['model']} | {cells[0]} | {cells[1]} | {mean:.2%} |")
    report = base.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {report}")


if __name__ == "__main__":
    main()
