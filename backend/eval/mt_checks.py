"""Output checks for an LLM translator (task T83, docs/experiments.md 15, decision rule (b)).

A translation fails the check when it has letters of another script or when it looks explanatory: a label, a
note, more lines than the input, quotes the input did not have, or more than twice the reference's length
(added text or repetition). Both are counted per output, with the rule written before any run. Empty outputs
are counted apart (the translator raised, so the evaluation stored ""). No model runs here.

    uv run --no-sync python -m eval.mt_checks eval/reports/mt_t83_dev_<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

# Scripts a translation may use: Latin in both (names, abbreviations; English left in a Korean output is
# counted by eval/mt_failures.py), Hangul only in Korean. Any other script (Han, Kana, Cyrillic, ...) counts,
# unless that very character is in the input (a Greek letter or a Hanja in the source may be kept).
ALLOWED = {"ko": ("LATIN", "FULLWIDTH LATIN", "HANGUL"), "en": ("LATIN", "FULLWIDTH LATIN")}
LABEL = re.compile(
    r"^\s*(?:translation|translated text|korean|english|번역|번역문|번역 결과|한국어|영어)\s*[:：]"
    r"|^\s*here(?:'s| is) (?:the|your|a) translation",
    re.IGNORECASE,
)
NOTE = re.compile(r"\bnotes?\s*:|\(\s*note\b|translator'?s note|참고\s*:|역주|번역자 주|※", re.IGNORECASE)
QUOTES = "\"'“”‘’「」『』«»"
LONG_RATIO = 2.0


def foreign_letters(output: str, source: str, target: str) -> str:
    """Letters of a script the target language does not use and the input does not have."""
    allowed = ALLOWED[target]
    found = []
    for char in output:
        if not unicodedata.category(char).startswith("L"):
            continue
        name = unicodedata.name(char, "")
        if name.startswith(allowed):
            continue
        if char in source and not name.startswith("HANGUL"):  # Hangul in an English output always counts
            continue
        found.append(char)
    return "".join(found)


def _quoted(text: str) -> bool:
    text = text.strip()
    return len(text) >= 2 and text[0] in QUOTES and text[-1] in QUOTES


def _letters(text: str) -> int:
    return len("".join(text.split()))


def explanation_signs(output: str, source: str, reference: str) -> list[str]:
    """Which explanatory signs an output shows (empty list: none)."""
    signs = []
    if output.count("\n") > source.count("\n"):
        signs.append("lines")
    if LABEL.search(output):
        signs.append("label")
    if NOTE.search(output) and not NOTE.search(source):
        signs.append("note")
    if _quoted(output) and not _quoted(source):
        signs.append("quotes")
    if _letters(output) > LONG_RATIO * max(1, _letters(reference)):
        signs.append("long")
    return signs


def check_items(items: list[dict], direction: str) -> dict:
    """Counts over an evaluation's items (source, reference, hypothesis), with the flagged ids."""
    target = direction.split("-")[1]
    flagged, empty = [], []
    for item in items:
        hypothesis = item["hypothesis"]
        if not hypothesis.strip():
            empty.append(item["id"])
            continue
        foreign = foreign_letters(hypothesis, item["source"], target)
        signs = explanation_signs(hypothesis, item["source"], item["reference"])
        if foreign or signs:
            flagged.append({"id": item["id"], "foreign": foreign, "signs": signs})
    return {
        "count": len(items),
        "flagged": len(flagged),
        "foreign_script": sum(bool(f["foreign"]) for f in flagged),
        "explanatory": sum(bool(f["signs"]) for f in flagged),
        "empty": len(empty),
        "flagged_items": flagged,
        "empty_ids": empty,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="a mt_*.json report written by eval.mt_eval")
    args = parser.parse_args()
    print("| 후보 | 방향 | 다른 문자 | 설명형 | 둘 중 하나 | 빈 출력 |")
    print("|---|---|---|---|---|---|")
    for result in json.loads(args.report.read_text(encoding="utf-8")):
        for direction, values in result["directions"].items():
            c = check_items(values["items"], direction)
            n = c["count"]
            print(
                f"| {result['model']} | {direction} | {c['foreign_script']}/{n} | {c['explanatory']}/{n} "
                f"| {c['flagged']}/{n} | {c['empty']}/{n} |"
            )


if __name__ == "__main__":
    main()
