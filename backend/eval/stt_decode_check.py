"""Candidate C check (docs/experiments.md 5): decoding in our code must not change what is recognized.

Recognizes each utterance twice with one loaded model, once with faster-whisper decoding the upload and once
with app.services.stt.decode_audio, and compares the texts.

Usage, from backend/ with the gpu and eval groups installed:
    uv run --no-sync python -m eval.stt_decode_check --split validation --limit 30
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from app.services.stt import WhisperSpeechToText
from eval.common import REPORTS
from eval.stt_eval import load_split, recognize


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
