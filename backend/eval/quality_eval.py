"""T65: translation quality of the whole flow, speech in (docs/experiments.md 9).

  uv run python -m eval.quality_eval --split test --tag t65_test
  uv run python -m eval.quality_eval --split test --en-ko hy-mt2-split --tag t83_quality_test   (T83)

For each FLEURS sentence ID present in both languages, the first recording of it in file order is
recognized with the server's settings and the result translated, as the speech API does. The same
sentence's reference transcription is translated as well. Both translations are scored with chrF against
the other language's transcription, so the difference is what recognition errors cost. --en-ko and --ko-en
pick the translation per direction as the server's EN_KO_TRANSLATION and KO_EN_TRANSLATION do (T83).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import pyarrow.parquet as pq
import sacrebleu

from app.config import Settings
from app.services.interfaces import ModelError, Translator
from app.services.models import load_translation
from app.services.stt import WhisperSpeechToText
from eval.common import DATA, FLEURS_CONFIG, MODELS, REPORTS, gpu_memory_mb
from eval.hymt_setup import QUANTS
from eval.stt_eval import error_rate, recognize
from eval.text_norm import normalize

DIRECTIONS = (("ko", "en"), ("en", "ko"))
EXAMPLES = 5


def load(split: str) -> tuple[dict[str, dict[int, dict]], list[int]]:
    """The first recording of each sentence ID per language, and the IDs both languages have."""
    rows: dict[str, dict[int, dict]] = {}
    for language, config in FLEURS_CONFIG.items():
        path = DATA / "fleurs" / config / f"{split}.parquet"
        columns = ["id", "audio", "raw_transcription", "transcription"]
        first: dict[int, dict] = {}
        for row in pq.read_table(path, columns=columns).to_pylist():
            first.setdefault(row["id"], row)
        rows[language] = first
    return rows, sorted(rows["ko"].keys() & rows["en"].keys())


def server_models(
    en_ko: str = "opus", ko_en: str = "opus", quant: str = "q8_0"
) -> tuple[WhisperSpeechToText, Translator]:
    """The models and settings app/services/models.py loads; Hy-MT2 must be the recorded build (T83)."""
    stt = WhisperSpeechToText("large-v3-turbo", vad_filter=True, device="cuda", compute_type="float16")
    _, _, _, model, digest = QUANTS[quant]
    settings = Settings(
        ct2_dir=MODELS / "ct2",
        en_ko_translation=en_ko,
        ko_en_translation=ko_en,
        hymt_model=model,
        hymt_digest=digest,
        hymt_timeout_s=120,
    )
    translator = load_translation(settings, {"device": "cuda", "compute_type": "float16"})
    return stt, translator


def translate(translator: Translator, text: str, source: str, target: str) -> str:
    """The translation, or "" when there is nothing to translate or the model fails (counted apart)."""
    if not text.strip():
        return ""
    try:
        return translator.translate(text, source, target)
    except ModelError:
        return ""


def run(args: argparse.Namespace) -> None:
    rows, ids = load(args.split)
    stt, translator = server_models(args.en_ko, args.ko_en, args.hymt_quant)
    results = {}
    for source, target in DIRECTIONS:
        items = []
        for number, sentence_id in enumerate(ids, 1):
            row, reference = rows[source][sentence_id], rows[target][sentence_id]["raw_transcription"].strip()
            heard = recognize(stt, row["audio"]["bytes"], source)
            items.append(
                {
                    "id": sentence_id,
                    "reference": reference,
                    "transcription": row["raw_transcription"].strip(),
                    "heard": heard,
                    "from_speech": translate(translator, heard, source, target),
                    "from_text": translate(translator, row["raw_transcription"].strip(), source, target),
                    "normalized_truth": normalize(row["transcription"], source),
                    "normalized_heard": normalize(heard, source),
                }
            )
            if number % 100 == 0:
                print(f"{source}->{target}: {number}/{len(ids)}", flush=True)
        references = [[item["reference"] for item in items]]
        for item in items:
            item["chrf_speech"] = sacrebleu.sentence_chrf(item["from_speech"], [item["reference"]]).score
            item["chrf_text"] = sacrebleu.sentence_chrf(item["from_text"], [item["reference"]]).score
        results[f"{source}-{target}"] = {
            "sentences": len(items),
            "chrf_speech": sacrebleu.corpus_chrf([i["from_speech"] for i in items], references).score,
            "chrf_text": sacrebleu.corpus_chrf([i["from_text"] for i in items], references).score,
            "recognition_error": error_rate(
                source, [i["normalized_truth"] for i in items], [i["normalized_heard"] for i in items]
            ),
            "nothing_heard": sum(1 for i in items if not i["heard"].strip()),
            "items": items,
        }
    write_report(results, args, translator.model_name)


def write_report(results: dict, args: argparse.Namespace, translator_name: str) -> None:
    gpu, _ = gpu_memory_mb()
    stamp = datetime.now(UTC)
    base = REPORTS / f"quality_{args.tag}_{stamp:%Y%m%d_%H%M%S}"
    base.with_suffix(".json").write_text(
        json.dumps(
            {"gpu": gpu, "translator": translator_name, "results": results}, indent=1, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    lines = [
        f"# 전체 흐름 번역 품질 ({args.tag})",
        "",
        f"- 날짜: {stamp.isoformat()}, FLEURS {args.split}, 문장 ID마다 첫 녹음 하나, GPU {gpu}",
        f"- 번역: {translator_name}",
        "- 음성 경로: 녹음 → 인식(서버 설정) → 번역. 전사 경로: 정답 전사 → 번역. 참조: 다른 언어의 전사",
        "",
        "| 방향 | 문장 | 음성 경로 chrF | 전사 경로 chrF | 차이 | 인식 오류(한 CER/영 WER) | "
        "인식 결과 없음 |",
        "|---|---|---|---|---|---|---|",
    ]
    for direction, r in results.items():
        lines.append(
            f"| {direction} | {r['sentences']} | {r['chrf_speech']:.1f} | {r['chrf_text']:.1f} | "
            f"{r['chrf_speech'] - r['chrf_text']:+.1f} | {r['recognition_error']:.2%} | "
            f"{r['nothing_heard']} |"
        )
    for direction, r in results.items():
        lines += ["", f"## {direction}: 인식 오류가 크게 깎은 문장 {EXAMPLES}개", ""]
        worst = sorted(r["items"], key=lambda i: i["chrf_speech"] - i["chrf_text"])[:EXAMPLES]
        for item in worst:
            lines += [
                f"- id {item['id']}: chrF {item['chrf_text']:.0f} → {item['chrf_speech']:.0f}",
                f"  - 전사: {item['transcription']}",
                f"  - 인식: {item['heard']}",
            ]
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--tag", required=True)
    # T83 (docs/experiments.md 15): the translation per direction, as the server's settings name it.
    parser.add_argument("--en-ko", choices=("opus", "hy-mt2", "hy-mt2-split"), default="opus")
    parser.add_argument("--ko-en", choices=("opus", "hy-mt2"), default="opus")
    parser.add_argument("--hymt-quant", choices=("q8_0", "q4_k_m"), default="q8_0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
