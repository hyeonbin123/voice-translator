"""Convert the seq2seq translation candidates to CTranslate2 (task T3).

Usage, from backend/ with the eval group installed (it brings transformers and torch for the conversion):
    uv run python -m eval.mt_convert

Writes data/models/ct2/<name> (git-ignored). The server itself needs only ctranslate2 and sentencepiece.
- opus-mt-ko-en and NLLB are converted from their Hugging Face repositories, then the tokens made by
  app/services/translation.py are compared with the Hugging Face tokenizer on the FLEURS sentences.
- The tc-big models are converted from Helsinki's original MarianNMT releases, because their Hugging Face
  ports are broken (docs/experiments.md): they use separate source and target vocabularies, but the ports
  keep one, so most Korean pieces become <unk>.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import ctranslate2
import httpx
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from app.services.translation import (
    NLLB_CODES,
    load_sentencepiece,
    marian_source_tokens,
    nllb_source_tokens,
    opus_preprocess,
)
from eval.common import DATA, FLEURS_CONFIG, MODELS

# name: (Hugging Face repo, files kept next to the converted model, source language or None for NLLB)
HF_MODELS = {
    "opus-mt-ko-en": ("Helsinki-NLP/opus-mt-ko-en", ["source.spm", "target.spm"], "ko"),
    "nllb-200-distilled-600M": ("facebook/nllb-200-distilled-600M", ["sentencepiece.bpe.model"], None),
}
OPUS_RELEASE = "https://object.pouta.csc.fi/Tatoeba-MT-models/{pair}/opusTCv20210807-sepvoc_transformer-big_2022-07-28.zip"
# name: (language pair in the release URL, source language)
OPUS_MODELS = {"opus-mt-tc-big-ko-en": ("kor-eng", "ko"), "opus-mt-tc-big-en-ko": ("eng-kor", "en")}


def sentences(language: str) -> list[str]:
    path = DATA / "fleurs" / FLEURS_CONFIG[language] / "validation.parquet"
    rows = pq.read_table(path, columns=["id", "raw_transcription"]).to_pylist()
    return list({row["id"]: row["raw_transcription"] for row in rows}.values())


def source_vocabulary(model_dir: Path) -> set[str]:
    for name in ("source_vocabulary.json", "shared_vocabulary.json"):
        if (model_dir / name).exists():
            return set(json.loads((model_dir / name).read_text(encoding="utf-8")))
    raise FileNotFoundError(f"no vocabulary in {model_dir}")


def check_hf_tokens(name: str, snapshot: str, output: Path, source: str | None) -> str:
    """CTranslate2 maps pieces missing from the vocabulary to <unk>, so compare after that step."""
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


def convert_hf(name: str, output: Path) -> str:
    repo, files, source = HF_MODELS[name]
    snapshot = snapshot_download(repo)
    ctranslate2.converters.TransformersConverter(snapshot, copy_files=files).convert(
        str(output), quantization="float16"
    )
    return check_hf_tokens(name, snapshot, output, source)


def write_yaml_vocabulary(plain: Path, target: Path) -> None:
    """The releases list one token per line (index = line number); the converter reads Marian's YAML form."""
    tokens = plain.read_text(encoding="utf-8").split("\n")
    if tokens and tokens[-1] == "":
        tokens.pop()
    escaped = (token.replace("\\", "\\\\").replace('"', '\\"') for token in tokens)
    target.write_text("".join(f'"{token}": {i}\n' for i, token in enumerate(escaped)), encoding="utf-8")


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


def main() -> None:
    names = [*HF_MODELS, *OPUS_MODELS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=names, default=names)
    parser.add_argument("--force", action="store_true", help="convert again even if the output exists")
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
