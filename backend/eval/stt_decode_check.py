"""Candidate C check (docs/experiments.md 5): decoding in our code must not change what is recognized.

Recognizes each utterance twice with one loaded model, once with faster-whisper decoding the upload and once
with app.services.stt.decode_audio, and compares the texts.

Usage, from backend/ with the gpu and eval groups installed:
    uv run --no-sync python -m eval.stt_decode_check --split validation --limit 30
"""

# [파일 역할] T23 후보 C(음성 디코딩을 우리 코드에서 하기)가 인식 결과를 바꾸지 않는지 확인하는 점검
# 스크립트다(docs/experiments.md 5절).
# [왜 필요한가] 번역 중 다른 요청이 0.3초씩 막힌 원인은 faster-whisper의 decode_audio가 음성을 읽을 때마다
# 부르는 전체 가비지 컬렉션이었다. 후보 C는 같은 PyAV 리샘플링을 우리 코드(app.services.stt.decode_audio)에서
# 하되 gc.collect()만 뺀다. 디코딩만 바꾸는 것이므로 인식 글자가 하나도 달라지지 않아야 채택할 수 있었다(5절
# '후보 C: 측정 전에 정한 절차', 결과는 60/60 같음). 최종 결정은 B(gc.freeze) 유지이고 C는 STT_OWN_DECODE로 꺼
# 둔 채 남아 있다.
# [흐름] 모델을 한 번 올리고, 발화마다 own_decode를 끈 채·켠 채 두 번 인식해 글자를 비교한 뒤 JSON·Markdown
# 보고서를 쓴다.

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

# 서버의 인식기와 1절 측정 도구의 데이터 읽기(load_split)·인식 호출(recognize)을 그대로 쓴다.
from app.services.stt import WhisperSpeechToText
from eval.common import REPORTS
from eval.stt_eval import load_split, recognize


# 명령줄 진입점. --split과 --limit(언어별 앞 N개, 기본 30: 5절 규칙의 '각 30개')를 받는다.
# 모델은 기본 옵션의 large-v3-turbo 하나만 올리고 own_decode 속성만 바꿔 가며 같은 모델로 두 번 인식한다.
# 모델을 두 번 올리지 않으므로 두 결과의 차이는 디코딩 방식에서만 나온다.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=30, help="first N utterances per language")
    args = parser.parse_args()

    stt = WhisperSpeechToText("large-v3-turbo")
    items = []
    for language in ("ko", "en"):
        for row in load_split(language, args.split, args.limit):
            texts = []
            for own_decode in (False, True):
                stt.own_decode = own_decode
                texts.append(recognize(stt, row["audio"]["bytes"], language))
            items.append(
                {"language": language, "id": row["id"], "faster_whisper": texts[0], "own_decode": texts[1]}
            )

    # 글자가 다른 발화만 모아 보고서에 원문 두 개를 나란히 적는다. 전부 같으면 '같은 발화 N/N'만 남는다.
    different = [item for item in items if item["faster_whisper"] != item["own_decode"]]
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = REPORTS / f"stt_decode_check_{args.split}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [
        "# 디코딩 방식에 따른 인식 결과 비교 (T23 후보 C)",
        "",
        f"- 날짜: {datetime.now(UTC).isoformat()}",
        f"- 데이터: FLEURS {args.split}, 언어별 앞 {args.limit}개. large-v3-turbo, 기본 옵션",
        f"- 글자가 같은 발화: {len(items) - len(different)}/{len(items)}",
    ]
    for item in different:
        lines.append(
            f"- 다름: {item['language']} id {item['id']}: `{item['faster_whisper']}` / `{item['own_decode']}`"
        )
    report = base.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"same {len(items) - len(different)}/{len(items)}, written to {report}")


if __name__ == "__main__":
    main()
