"""seq2seq 번역 후보를 CTranslate2로 변환한다.

backend/에서 eval 그룹을 설치한 뒤 쓰는 법(변환에 필요한 transformers와 torch가 그 그룹에 있다):
    uv run python -m eval.mt_convert

data/models/ct2/<name>에 쓴다(git이 무시함). 서버 자체에는 ctranslate2와 sentencepiece만 필요하다.
- opus-mt-ko-en과 NLLB는 Hugging Face 저장소에서 변환한 뒤, app/services/translation.py가 만드는 토큰을
  FLEURS 문장에서 Hugging Face 토크나이저와 비교한다.
- tc-big 모델은 Hugging Face 이식본이 깨져 있어 Helsinki의 원본 MarianNMT 배포본에서
  변환한다(docs/experiments.md): 원문과 번역문 어휘를 따로 쓰는데 이식본은 하나만 두어, 한국어 조각
  대부분이 <unk>가 된다.
"""

# [파일 역할] 번역 후보 seq2seq 모델을 CTranslate2 형식(float16)으로 바꿔 data/models/ct2/<이름>에 두는 일회성
# 변환 스크립트. 바꾼 뒤 토큰화가 원래와 같은지 FLEURS 문장으로 확인한다.
# [설계 이유] 서버는 음성 인식과 같은 엔진(CTranslate2)으로 번역해 transformers·torch 없이 ctranslate2와
# sentencepiece만으로 돈다. tc-big은 Hugging Face 변환본의 어휘가 깨져 있어(한국어 조각 대부분이 <unk>)
# Helsinki의 원본 MarianNMT 배포본(각 약 740MB)을 받아 바꾼다(위 docstring, README '전체 실행').
# [연결] README의 첫 실행 준비 단계에서 `uv run python -m eval.mt_convert --models opus-mt-tc-big-ko-en
# opus-mt-tc-big-en-ko`로 부른다. 결과 폴더는 서버(services/models.load_translation, CT2_DIR)와
# eval/mt_eval.py가 읽는다.

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

# ctranslate2: 변환기(TransformersConverter, MarianConverter). pyarrow: FLEURS parquet에서 확인용 문장을 읽음.
# huggingface_hub·transformers: Hugging Face 모델을 받고, 원래 토크나이저와 비교한다(eval 의존성 그룹).
import ctranslate2
import httpx
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# 서버와 같은 토큰화 함수를 가져와, 서버가 실제로 만드는 토큰을 검사한다.
from app.services.translation import (
    NLLB_CODES,
    load_sentencepiece,
    marian_source_tokens,
    nllb_source_tokens,
    opus_preprocess,
)
from eval.common import DATA, FLEURS_CONFIG, MODELS

# HF_MODELS: Hugging Face 저장소에서 바로 바꾸는 후보(opus-mt-ko-en, NLLB). OPUS_RELEASE·OPUS_MODELS: tc-big
# 원본 배포본 주소와 언어 쌍(원문·번역문 어휘를 따로 쓰는 sepvoc 모델).
# 이름: (Hugging Face 저장소, 변환된 모델 옆에 둘 파일, 원문 언어. NLLB는 None)
HF_MODELS = {
    "opus-mt-ko-en": ("Helsinki-NLP/opus-mt-ko-en", ["source.spm", "target.spm"], "ko"),
    "nllb-200-distilled-600M": ("facebook/nllb-200-distilled-600M", ["sentencepiece.bpe.model"], None),
}
OPUS_RELEASE = "https://object.pouta.csc.fi/Tatoeba-MT-models/{pair}/opusTCv20210807-sepvoc_transformer-big_2022-07-28.zip"
# 이름: (배포본 주소의 언어 쌍, 원문 언어)
OPUS_MODELS = {"opus-mt-tc-big-ko-en": ("kor-eng", "ko"), "opus-mt-tc-big-en-ko": ("eng-kor", "en")}


# 확인용 문장: FLEURS validation의 원문을 문장 id마다 하나씩(같은 문장의 여러 녹음은 같은 원문이라 dict로 중복
# 제거). FLEURS를 먼저 받아 둬야 한다(eval.fleurs_download).
def sentences(language: str) -> list[str]:
    path = DATA / "fleurs" / FLEURS_CONFIG[language] / "validation.parquet"
    rows = pq.read_table(path, columns=["id", "raw_transcription"]).to_pylist()
    return list({row["id"]: row["raw_transcription"] for row in rows}.values())


# 변환된 모델의 원문 어휘. 어휘를 나눈 모델은 source_vocabulary.json, 하나를 공유하는 모델은
# shared_vocabulary.json에 있다.
def source_vocabulary(model_dir: Path) -> set[str]:
    for name in ("source_vocabulary.json", "shared_vocabulary.json"):
        if (model_dir / name).exists():
            return set(json.loads((model_dir / name).read_text(encoding="utf-8")))
    raise FileNotFoundError(f"no vocabulary in {model_dir}")


def check_hf_tokens(name: str, snapshot: str, output: Path, source: str | None) -> str:
    """CTranslate2는 어휘에 없는 조각을 <unk>로 바꾸므로, 그 단계 뒤의 토큰으로 비교한다.

    우리 토큰화(app/services/translation.py)가 Hugging Face 토크나이저와 같은지 모든 문장에서 비교해 다른
    문장 수를 돌려준다. Hugging Face 쪽은 어휘에 없는 조각을 <unk>로 바꿔 내므로, 우리 쪽도 변환된 어휘에
    없는 조각을 <unk>로 바꾼 뒤 비교한다. opus-mt는 한 방향(source), NLLB는 두 언어 모두 본다.
    """
    vocabulary = source_vocabulary(output)
    mismatches = total = 0
    for language in [source] if source else ["ko", "en"]:
        if source:
            reference = AutoTokenizer.from_pretrained(snapshot)
            processor = load_sentencepiece(Path(snapshot) / "source.spm")
        else:
            reference = AutoTokenizer.from_pretrained(snapshot, src_lang=NLLB_CODES[language])
            processor = load_sentencepiece(Path(snapshot) / "sentencepiece.bpe.model")
        for text in sentences(language):
            if source:
                ours = marian_source_tokens(processor, text)
                expected = reference.convert_ids_to_tokens(reference.encode(opus_preprocess(text)))
            else:
                ours = nllb_source_tokens(processor, text, language)
                expected = reference.convert_ids_to_tokens(reference.encode(text))
            total += 1
            mismatches += [p if p in vocabulary else "<unk>" for p in ours] != expected
    return f"{name}: tokens differ from the Hugging Face tokenizer in {mismatches}/{total} sentences"


# Hugging Face 저장소를 받아 float16으로 바꾸고, 토크나이저 파일(copy_files)을 옆에 복사한 뒤 토큰을 확인한다.
def convert_hf(name: str, output: Path) -> str:
    repo, files, source = HF_MODELS[name]
    snapshot = snapshot_download(repo)
    ctranslate2.converters.TransformersConverter(snapshot, copy_files=files).convert(
        str(output), quantization="float16"
    )
    return check_hf_tokens(name, snapshot, output, source)


def write_yaml_vocabulary(plain: Path, target: Path) -> None:
    """배포본은 한 줄에 토큰 하나를 적고(줄 번호가 색인), 변환기는 Marian의 YAML 형식을 읽는다.

    그래서 MarianConverter가 읽는 YAML("토큰": 번호) 형식으로 바꾼다. 역슬래시와 큰따옴표는 YAML 문자열
    안에서 이스케이프한다. 마지막 빈 줄은 토큰이 아니므로 뺀다.
    """
    tokens = plain.read_text(encoding="utf-8").split("\n")
    if tokens and tokens[-1] == "":
        tokens.pop()
    escaped = (token.replace("\\", "\\\\").replace('"', '\\"') for token in tokens)
    target.write_text("".join(f'"{token}": {i}\n' for i, token in enumerate(escaped)), encoding="utf-8")


# tc-big 변환: 배포본 zip을 (없으면) 받고 풀어서 → 원문·번역문 어휘를 YAML로 바꾸고 → 가장 좋은
# 체크포인트(*.best-perplexity.npz)를 MarianConverter로 바꾸고 → SentencePiece 모델 두 개를 옆에 복사한다.
# 마지막으로 FLEURS 문장의 원문 조각 중 어휘에 없는 조각 수를 센다(docs/experiments.md 2절: 한→영 3262개 중
# 2개, 영→한 3973개 중 1개였다).
def convert_opus(name: str, output: Path) -> str:
    pair, source = OPUS_MODELS[name]
    archive = MODELS / "opus" / f"{pair}-tc-big.zip"
    extracted = MODELS / "opus" / pair
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {OPUS_RELEASE.format(pair=pair)} (about 740 MB) ...", flush=True)
        with httpx.stream("GET", OPUS_RELEASE.format(pair=pair), follow_redirects=True, timeout=60) as reply:
            reply.raise_for_status()
            with archive.open("wb") as file:
                for chunk in reply.iter_bytes():
                    file.write(chunk)
    if not extracted.exists():
        with zipfile.ZipFile(archive) as release:
            release.extractall(extracted)
    vocabularies = []
    for side in ("src", "trg"):
        vocabularies.append(extracted / f"{side}.vocab.yml")
        write_yaml_vocabulary(next(extracted.glob(f"*.{side}.vocab")), vocabularies[-1])
    model = next(extracted.glob("*.best-perplexity.npz"))
    ctranslate2.converters.MarianConverter(str(model), [str(v) for v in vocabularies]).convert(
        str(output), quantization="float16"
    )
    for spm in ("source.spm", "target.spm"):
        shutil.copy(extracted / spm, output / spm)

    vocabulary = source_vocabulary(output)
    processor = load_sentencepiece(output / "source.spm")
    pieces = [piece for text in sentences(source) for piece in marian_source_tokens(processor, text)]
    missing = sum(piece not in vocabulary for piece in pieces)
    return f"{name}: {missing}/{len(pieces)} source pieces are missing from the vocabulary"


# 고른 모델마다 변환한다. 이미 있으면 --force가 없는 한 건너뛰고, 다시 할 때는 이전 결과 폴더를 지우고
# 처음부터 바꾼다.
def main() -> None:
    names = [*HF_MODELS, *OPUS_MODELS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=names, default=names)
    parser.add_argument("--force", action="store_true", help="결과가 이미 있어도 다시 변환한다")
    args = parser.parse_args()

    for name in args.models:
        output = MODELS / "ct2" / name
        if output.exists() and not args.force:
            print(f"{name}: already converted, skipping (use --force to redo)")
            continue
        shutil.rmtree(output, ignore_errors=True)
        print(f"converting {name} ...", flush=True)
        print(convert_hf(name, output) if name in HF_MODELS else convert_opus(name, output), flush=True)


if __name__ == "__main__":
    main()
