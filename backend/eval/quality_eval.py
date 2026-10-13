"""T65: translation quality of the whole flow, speech in (docs/experiments.md 9).

  uv run python -m eval.quality_eval --split test --tag t65_test
  uv run python -m eval.quality_eval --split test --en-ko hy-mt2-split --tag t83_quality_test   (T83)

For each FLEURS sentence ID present in both languages, the first recording of it in file order is
recognized with the server's settings and the result translated, as the speech API does. The same
sentence's reference transcription is translated as well. Both translations are scored with chrF against
the other language's transcription, so the difference is what recognition errors cost. --en-ko and --ko-en
pick the translation per direction as the server's EN_KO_TRANSLATION and KO_EN_TRANSLATION do (T83).
"""

# [파일 역할] 음성 → 인식 → 번역 전체 흐름의 번역 품질을 재는 확인 측정 스크립트다(T65, docs/experiments.md
# 9절). 같은 문장을 (1) 녹음을 인식한 글자로 번역한 것(음성 경로)과 (2) 정답 전사로 번역한 것(전사 경로)을
# 나란히 chrF로 채점해, 인식 오류가 번역 품질을 얼마나 깎는지 본다.
# [왜 필요한가] 2절(T3)의 번역 측정은 정답 글자를 넣었으므로 실제 서비스(음성 입력)의 품질을 말해 주지 않는다.
# 두 경로의 차이가 곧 인식 오류의 몫이다. 고르는 것이 없는 확인 측정이라 판단 규칙이 없다.
# [도구 점검] 전사 경로는 2절 test와 같은 문장·같은 번역기라 같은 chrF(한→영 55.7, 영→한 36.1)가 나와야 한다.
# 결과는 같았다(9절).
# [연결] 인식은 eval/stt_eval.py의 recognize·error_rate, 정규화는 eval/text_norm.py, 번역기는 서버의
# load_translation을 그대로 쓴다. T83에서는 Hy-MT2 설정의 음성 경로 확인에도 쓰도록 --en-ko·--ko-en이
# 더해졌다.

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

# pyarrow.parquet: FLEURS parquet에서 음성(audio)과 두 가지 전사를 읽는다. sacrebleu: 문장·전체 chrF.
import pyarrow.parquet as pq
import sacrebleu

# 서버 쪽 구성 요소(Settings, load_translation, WhisperSpeechToText)를 그대로 가져와 서버와 같은 설정으로
# 잰다. 인식 호출과 오류율 계산은 1절 측정 도구(eval.stt_eval)와 공유해 같은 방법으로 센다.
from app.config import Settings
from app.services.interfaces import ModelError, Translator
from app.services.models import load_translation
from app.services.stt import WhisperSpeechToText
from eval.common import DATA, FLEURS_CONFIG, MODELS, REPORTS, gpu_memory_mb
from eval.hymt_setup import QUANTS
from eval.stt_eval import error_rate, recognize
from eval.text_norm import normalize

# 잴 방향(원문, 대상)과 보고서에 실을 '인식 오류가 크게 깎은 문장' 예시 수.
DIRECTIONS = (("ko", "en"), ("en", "ko"))
EXAMPLES = 5


# 분할의 FLEURS 데이터를 언어마다 읽어, 문장 ID마다 파일 순서상 첫 녹음 한 행만 남긴다(setdefault라 처음 본
# 행이 남음). 같은 문장을 여러 화자가 읽어 행이 겹치므로, 문장마다 녹음 하나만 써서 번역 평가의 문장 묶음을
# 2절(문장 ID마다 한 쌍)과 같게 맞춘다.
# 출력: 언어별 {ID: 행}과 두 언어에 모두 있는 ID의 정렬 목록.
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


# 서버와 같은 인식기·번역기를 만든다. 인식: large-v3-turbo, VAD 필터, GPU float16(1절·1-1절에서 고른 설정).
# 번역: Settings로 방향별 설정을 주고 서버의 load_translation이 만들게 한다. 기본은 opus 두 방향(영→한 문장
# 단위)이다.
# Hy-MT2를 고르면 측정한 빌드의 digest를 hymt_digest로 넘겨, Ollama에 다른 빌드가 있으면 서버 시작처럼 멈추게
# 한다(T83). hymt_timeout_s는 서버 기본값(30초)이 아니라 120초로 넉넉히 준다.
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


# 번역 결과를 돌려주되, 번역할 글자가 없으면(인식 결과가 비었을 때) 번역기를 부르지 않고 빈 문자열을, 모델
# 오류면 빈 문자열을 돌려준다. 빈 번역은 chrF에서 그대로 감점되고, 인식 결과가 빈 녹음 수는 보고서에 따로
# 센다.
def translate(translator: Translator, text: str, source: str, target: str) -> str:
    """The translation, or "" when there is nothing to translate or the model fails (counted apart)."""
    if not text.strip():
        return ""
    try:
        return translator.translate(text, source, target)
    except ModelError:
        return ""


# 측정 본체. 방향마다 모든 문장에 대해 녹음을 인식하고, 인식 결과와 정답 전사를 각각 번역해 문장별 항목을
# 만든다.
# 참조는 대상 언어 쪽 같은 문장 ID의 raw_transcription(문장부호 있는 원문)이다. 인식 오류율은 1절과 같은
# 정규화를 거친 transcription(소문자·문장부호 없음)과 인식 결과로 센다.
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
            # 오래 걸리는 측정이라 100문장마다 진행 상황을 찍는다.
            if number % 100 == 0:
                print(f"{source}->{target}: {number}/{len(ids)}", flush=True)
        # 문장별 chrF(나쁜 문장 찾기용)를 붙이고, 방향별 요약을 만든다. 품질 수치는 전체 합산 chrF이고, 인식
        # 오류율도 문장 평균이 아니라 전체 합산이다(error_rate). 'nothing_heard'는 인식 결과가 빈 녹음 수다.
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


# reports/quality_<tag>_<UTC 시각>.json(문장별 결과 포함)과 .md(요약 표와 예시)를 쓴다. 표에는 방향별 음성
# 경로·전사 경로 chrF, 그 차이, 인식 오류율, 인식 결과 없음 수가 들어간다.
# 예시는 '음성 경로 chrF − 전사 경로 chrF'가 가장 작은(인식 오류가 가장 크게 깎은) 문장들이다. 9절의 오류
# 유형(괄호 설명을 읽지 않음, 고유명사 표기, 약어, 첫 단어 오인식)은 이 목록을 읽고 정리한 것이다.
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


# 명령줄 진입점. 확인 측정이 기본이라 --split 기본값이 test다. --tag는 보고서 이름에 쓰여 필수다.
# --en-ko·--ko-en은 서버의 EN_KO_TRANSLATION·KO_EN_TRANSLATION과 같은 값을 받고, --hymt-quant로 Hy-MT2
# 양자화를 고른다(T83).
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
