"""Count visible failure types in a saved translation report (task T3), without running any model.

Usage, from backend/:
    uv run python -m eval.mt_failures eval/reports/mt_t3_dev_20260912_231928.json

The checks are rough signals for reading the results, not part of the selection rule.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HAN = re.compile(r"[一-鿿]")
# Five or more Latin words in a row in a Korean translation: part of the input was left in English.
LATIN_RUN = re.compile(r"(?:[A-Za-z]+[ ,.]+){5,}")


def count(items: list[dict], direction: str) -> dict[str, int]:
    ratios = [len(item["hypothesis"]) / max(1, len(item["reference"])) for item in items]
    return {
        "han": sum(bool(HAN.search(item["hypothesis"])) for item in items),
        "left_in_english": sum(bool(LATIN_RUN.search(item["hypothesis"])) for item in items)
        if direction == "en-ko"
        else 0,
        "short": sum(ratio < 0.6 for ratio in ratios),  # likely dropped content
        "long": sum(ratio > 2.0 for ratio in ratios),  # likely rambling or repetition
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="a mt_*.json report written by eval.mt_eval")
    args = parser.parse_args()
    print("| 후보 | 방향 | 한자 섞임 | 영어로 남음 | 참조의 60% 미만 길이 | 참조의 200% 초과 길이 |")
    print("|---|---|---|---|---|---|")
    for result in json.loads(args.report.read_text(encoding="utf-8")):
        for direction, values in result["directions"].items():
            n = len(values["items"])
            c = count(values["items"], direction)
            print(
                f"| {result['model']} | {direction} | {c['han']}/{n} | {c['left_in_english']}/{n} "
                f"| {c['short']}/{n} | {c['long']}/{n} |"
            )


if __name__ == "__main__":
    main()
