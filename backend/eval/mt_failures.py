"""저장된 번역 보고서에서 눈에 보이는 실패 유형을 센다. 모델은 돌리지 않는다.

backend/에서 쓰는 법:
    uv run python -m eval.mt_failures eval/reports/mt_t3_dev_20260912_231928.json

이 점검은 결과를 읽기 위한 대략의 신호이고, 선택 규칙의 일부가 아니다.
"""

# [파일 역할] eval/mt_eval.py가 저장한 번역 보고서(JSON)를 다시 읽어, 눈에 보이는 실패 유형(한자 섞임, 영어로
# 남음, 너무 짧음, 너무 김)을 후보·방향별로 세어 Markdown 표로 출력한다. 모델은 돌리지 않는다.
# [왜 필요한가] chrF 하나로는 '문장을 빠뜨렸다', '중국어가 섞였다' 같은 실패가 보이지 않는다. T3 결과를 읽을
# 때 이 신호로 Qwen 영→한의 한자 섞임(129개 중 28개)과 tc-big 영→한의 문장 누락을 찾았고(docs/experiments.md
# 2절), T17에서는 '영어로 남음'과 '참조의 60% 미만 길이' 두 신호의 합계(누락 신호)를 문장 단위 번역의 채택
# 조건으로 썼다(2-1절).
# [주의] T3에서는 선택 규칙과 별개인 읽기용 신호였고, T17에서는 그중 두 신호(영어로 남음, 60% 미만 길이)가
# 채택 조건에 들어갔다. 길이 비율은 원문이 아니라 같은 언어의 참조와 비교한다.

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# HAN: CJK 통합 한자 영역(U+4E00~U+9FFF)의 글자. 한국어·영어 번역문에 한자가 나오면 중국어가 섞인 것으로 본다.
HAN = re.compile(r"[一-鿿]")
# LATIN_RUN: 한국어 번역문에 로마자 단어가 5개 이상 이어지면 입력의 일부가 영어로 남은 것으로 본다.
LATIN_RUN = re.compile(r"(?:[A-Za-z]+[ ,.]+){5,}")


# 한 방향의 문장별 결과(items)에서 실패 신호 네 가지를 센다. 입력: mt_eval 보고서의 items, 방향
# 문자열('ko-en'·'en-ko'). 출력: 신호별 개수.
# - han: 번역문에 한자가 있는 문장 수.
# - left_in_english: 로마자 단어 5개 이상이 이어진 문장 수. 영→한 결과에서만 뜻이 있어 한→영은 0으로
#   둔다(2-1절 표의 주석과 같음).
# - short: 번역문 글자 수가 참조의 60% 미만. 내용을 빠뜨렸을 가능성.
# - long: 참조의 2배 초과. 같은 말 반복이나 덧붙인 설명 가능성.
# 참조 길이가 0이어도 나누기 오류가 나지 않게 max(1, ...)로 나눈다.
def count(items: list[dict], direction: str) -> dict[str, int]:
    ratios = [len(item["hypothesis"]) / max(1, len(item["reference"])) for item in items]
    return {
        "han": sum(bool(HAN.search(item["hypothesis"])) for item in items),
        "left_in_english": sum(bool(LATIN_RUN.search(item["hypothesis"])) for item in items)
        if direction == "en-ko"
        else 0,
        "short": sum(ratio < 0.6 for ratio in ratios),
        "long": sum(ratio > 2.0 for ratio in ratios),
    }


# 명령줄 진입점. 보고서 경로 하나를 받아 후보·방향마다 한 행씩 '개수/전체' 형식의 표를 출력한다. 출력 표는
# docs/experiments.md에 옮겨 적기 쉽게 Markdown 형식이다.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="eval.mt_eval이 쓴 mt_*.json 리포트")
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
