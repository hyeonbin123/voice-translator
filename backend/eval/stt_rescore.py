"""저장된 음성 인식 보고서의 오류율을 지금의 정규화 규칙으로 다시 계산한다(작업 T15).

사용법(backend/에서, eval 그룹을 설치한 상태로):
    uv run python -m eval.stt_rescore eval/reports/stt_t2_dev_20260912_223312.json

원본 옆에 <report>_rescored.json과 .md를 쓰고, 원본은 그대로 둔다. 각 항목에는 정규화된 참조와 인식 결과가
남아 있고, 정규화는 이번에 적용하는 수정(한국어 아포스트로피)을 빼면 여러 번 해도 결과가 같으므로, 그 글을
다시 정규화하면 모델을 돌리지 않고도 고친 규칙의 글을 얻는다.
"""

# [파일 역할] 저장된 음성 인식 보고서(eval/stt_eval.py의 JSON)에 지금의 정규화(eval/text_norm.py)를 다시
# 적용해 오류율만 다시 계산하는 스크립트다(T15). 원래 보고서는 그대로 두고 <원래 이름>_rescored.json·.md를
# 옆에 쓴다.
# [왜 필요한가] T15에서 한국어 아포스트로피 정규화를 고쳤을 때, 모델을 다시 돌리면 GPU 시간이 들고 결과가
# 실행마다 조금씩 흔들릴 수 있다. 보고서에는 이미 정규화된 정답·인식 결과가 남아 있고 정규화는 다시 적용해도
# 고친 부분 말고는 바뀌지 않으므로, 저장된 글에 새 정규화만 다시 적용하면 모델 없이 보정된 값을
# 얻는다(docs/experiments.md 1절 보정 표).
# [연결] 결과 파일(stt_t2_*_rescored)은 eval/reanalysis.py(T80)가 T2 비교의 입력으로 쓴다.

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

# 오류율 계산(전체 합산 CER·WER)과 정규화는 원래 측정과 같은 함수를 쓴다.
from eval.stt_eval import error_rate
from eval.text_norm import normalize


# 보고서의 모든 모델·언어·발화에 정규화를 다시 적용하고, 발화별 오류율과 언어별 전체 합산 오류율을 다시
# 계산한다. 이전 값은 error_before_rescore로 남겨 보고서에 '전 → 후'로 적는다. 입력 dict를 직접 고쳐서
# 돌려준다.
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


# 명령줄 진입점. 보고서 경로를 받아 _rescored.json과 '모델별 한국어 CER·영어 WER 전 → 후, 두 언어 평균' 표를
# 담은 .md를 쓴다.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="eval.stt_eval이 쓴 stt_*.json 보고서")
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
        # 선택 규칙에 쓰는 (한국어 CER + 영어 WER) / 2 평균을 보정 뒤 값으로 다시 낸다.
        mean = statistics.mean(values["error"] for values in by_language.values())
        lines.append(f"| {result['model']} | {cells[0]} | {cells[1]} | {mean:.2%} |")
    report = base.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {report}")


if __name__ == "__main__":
    main()
