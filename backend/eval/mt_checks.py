"""Output checks for an LLM translator (task T83, docs/experiments.md 15, decision rule (b)).

A translation fails the check when it has letters of another script or when it looks explanatory: a label, a
note, more lines than the input, quotes the input did not have, or more than twice the reference's length
(added text or repetition). Both are counted per output, with the rule written before any run. Empty outputs
are counted apart (the translator raised, so the evaluation stored ""). No model runs here.

    uv run --no-sync python -m eval.mt_checks eval/reports/mt_t83_dev_<stamp>.json
"""

# [파일 역할] LLM 번역기(Hy-MT2 시험)의 출력이 '번역문만' 내놓았는지 점검하는 규칙 모음. 모델을 돌리지 않고
# 저장된 평가 결과만 읽는다.
# [설계 이유] LLM은 seq2seq 모델과 달리 다른 문자(한자·가나 등)를 섞거나 '번역:' 같은 꼬리표·설명·따옴표를
# 붙일 수 있다. 측정 전에 적은 판단 규칙 (b)(다른 문자·설명형 출력 129개 중 1개 이하, 빈 출력 0개,
# docs/experiments.md 15절)를 숫자로 셀 수 있게 기준을 코드로 고정했다.
# [연결] eval/mt_eval.py가 평가 결과마다 check_items를 불러 리포트에 넣고, 이 파일을 직접 실행하면 리포트를
# 표로 출력한다. 테스트: tests/test_mt_checks.py.

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

# Scripts a translation may use: Latin in both (names, abbreviations; English left in a Korean output is
# counted by eval/mt_failures.py), Hangul only in Korean. Any other script (Han, Kana, Cyrillic, ...) counts,
# unless that very character is in the input (a Greek letter or a Hanja in the source may be kept).
# ALLOWED: 도착 언어별로 허용하는 문자 계열(유니코드 문자 이름의 앞부분). LABEL: 출력 앞에 붙은 '번역:'·'Here
# is the translation' 같은 꼬리표. NOTE: 메모·역주 표시. QUOTES: 출력 전체를 감싼 따옴표로 볼 문자.
# LONG_RATIO: 참조보다 2배 넘게 길면 덧붙임·반복으로 본다.
ALLOWED = {"ko": ("LATIN", "FULLWIDTH LATIN", "HANGUL"), "en": ("LATIN", "FULLWIDTH LATIN")}
LABEL = re.compile(
    r"^\s*(?:translation|translated text|korean|english|번역|번역문|번역 결과|한국어|영어)\s*[:：]"
    r"|^\s*here(?:'s| is) (?:the|your|a) translation",
    re.IGNORECASE,
)
NOTE = re.compile(r"\bnotes?\s*:|\(\s*note\b|translator'?s note|참고\s*:|역주|번역자 주|※", re.IGNORECASE)
QUOTES = "\"'“”‘’「」『』«»"
LONG_RATIO = 2.0


# 출력의 글자(L 범주) 중 도착 언어가 쓰지 않는 문자 계열을 모아 돌려준다. 입력에 있던 바로 그 글자(그리스
# 문자, 한자 등)는 옮겨 적은 것으로 보고 봐준다. 단 영어 출력의 한글은 입력에 있었더라도 늘 센다(번역하지 않고
# 남긴 것이므로).
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


# 글 전체가 따옴표 문자로 시작하고 끝나는지. _letters는 공백을 뺀 글자 수(길이 비교용).
def _quoted(text: str) -> bool:
    text = text.strip()
    return len(text) >= 2 and text[0] in QUOTES and text[-1] in QUOTES


def _letters(text: str) -> int:
    return len("".join(text.split()))


# 설명형 출력의 표시를 모두 모은다: 입력보다 줄이 많음(lines), 꼬리표(label), 입력에 없던 메모(note), 입력에
# 없던 전체 따옴표(quotes), 참조보다 2배 넘게 김(long). 입력에 원래 있던 메모·따옴표는 세지 않는다.
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


# 한 방향 평가 항목 전체를 점검한다. 빈 출력(번역기가 예외를 내 평가가 ""를 저장한 경우)은 따로 센다. 다른
# 문자와 설명형은 한 출력에 둘 다 있을 수 있어, flagged(둘 중 하나라도)와 각각의 수를 함께 낸다. 걸린 항목의
# id와 이유를 남겨 사람이 다시 볼 수 있게 한다.
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


# eval.mt_eval이 쓴 리포트(JSON)를 읽어 후보·방향마다 점검 결과를 마크다운 표로 출력한다(docs/experiments.md에
# 옮겨 적는 형식).
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
