"""Hy-MT2-1.8B from Tencent's official GGUF into Ollama (task T83, docs/experiments.md 15).

The GGUF files are fetched at one fixed revision and checked against their SHA-256 (about 3 GB, kept in
data/models/hy-mt2, not committed). `modelfile` writes an Ollama Modelfile next to each file: the model's own
chat template (tokenizer.chat_template in the GGUF) for one user turn, written as a Go template, its
end-of-turn token as the stop, the user-turn token as a second stop, no system prompt (the model has none),
and the request options of app/services/translation.py. No community Ollama package is used. From backend/:

    uv run --no-sync python -m eval.hymt_setup download
    uv run --no-sync python -m eval.hymt_setup modelfile
    ollama create hy-mt2:1.8b-q8_0 -f ../data/models/hy-mt2/Modelfile.q8_0                       (host Ollama)
    docker compose exec ollama ollama create hy-mt2:1.8b-q8_0 -f /models/hy-mt2/Modelfile.q8_0   (compose)
    uv run --no-sync python -m eval.hymt_setup check --quant q8_0 [--ollama-url URL]

`check` compares what Ollama reports for the model (template, stops, architecture) with this file and prints
the model's digest, which must be the one docs/experiments.md 15 records for the measured build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.services.translation import HY_MT_OPTIONS
from eval.common import MODELS

REPO = "tencent/Hy-MT2-1.8B-GGUF"  # Apache-2.0 (LICENSE.txt read 2026-10-08)
REVISION = "a0c709d9fac510f2c807aa3af52872340dc37a4a"
FOLDER = MODELS / "hy-mt2"
# quant: (file, SHA-256 from the Hugging Face API, size in bytes, Ollama model name, digest of the build that
# Ollama 0.35.1 makes from the Modelfile below; first seen in a container without GPU, 2026-10-08)
QUANTS = {
    "q8_0": (
        "Hy-MT2-1.8B-Q8_0.gguf",
        "5c3fe0b1408a5ceb0143184ef247b11b579c525f4b02b060e6c851bb76fef1a4",
        1908528192,
        "hy-mt2:1.8b-q8_0",
        "4d64c0ffabe302bcc4151c9c1a6e93e032fbc32ba1b8cb67d5713985035f1fab",
    ),
    "q4_k_m": (
        "Hy-MT2-1.8B-Q4_K_M.gguf",
        "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699",
        1133080448,
        "hy-mt2:1.8b-q4_k_m",
        "e8d0f9b58e637743acc6a95c3b51a15385041ee183baf38c101b4d6f6d6aca3e",
    ),
}
BOS = "<｜hy_begin▁of▁sentence｜>"
USER = "<｜hy_User｜>"
ASSISTANT = "<｜hy_Assistant｜>"
END_OF_TURN = "<｜hy_place▁holder▁no▁2｜>"  # eos_token_id 120020 in the GGUF and the tokenizer
TEMPLATE = (
    BOS
    + '{{- range .Messages }}{{- if eq .Role "user" }}'
    + USER
    + '{{ .Content }}{{- else if eq .Role "assistant" }}'
    + ASSISTANT
    + "{{ .Content }}"
    + END_OF_TURN
    + "{{- end }}{{- end }}"
    + ASSISTANT
)
STOPS = (END_OF_TURN, USER)
ARCHITECTURE = "hunyuan-dense"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(quant: str, folder: Path = FOLDER) -> Path:
    name, expected, size, _, _ = QUANTS[quant]
    path = folder / name
    if not path.is_file() or path.stat().st_size != size or sha256(path) != expected:
        raise ValueError(f"{path} is missing or not the pinned file (SHA-256 {expected})")
    return path


def download(folder: Path = FOLDER) -> None:
    import httpx

    folder.mkdir(parents=True, exist_ok=True)
    for quant, (name, *_) in QUANTS.items():
        try:
            verify(quant, folder)
            print(f"{name}: already there")
            continue
        except ValueError:
            pass
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        part = folder / f"{name}.part"
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            with part.open("wb") as file:
                for block in response.iter_bytes(1 << 20):
                    file.write(block)
        part.replace(folder / name)
        verify(quant, folder)
        print(f"{name}: downloaded and checked")


def modelfile_text(quant: str) -> str:
    """The Modelfile, with FROM relative to the Modelfile's own folder (the GGUF's folder)."""
    lines = [f"FROM ./{QUANTS[quant][0]}", f'TEMPLATE """{TEMPLATE}"""']
    lines += [f'PARAMETER stop "{stop}"' for stop in STOPS]
    lines += [f"PARAMETER {key} {value}" for key, value in HY_MT_OPTIONS.items()]
    return "\n".join(lines) + "\n"


def write_modelfiles(folder: Path = FOLDER) -> None:
    for quant in QUANTS:
        verify(quant, folder)
        path = folder / f"Modelfile.{quant}"
        path.write_text(modelfile_text(quant), encoding="utf-8")
        print(f"wrote {path}")


def check(quant: str, ollama_url: str) -> dict:
    """What Ollama reports for the model, against this file. Raises SystemExit on a difference."""
    import httpx

    name = QUANTS[quant][3]
    with httpx.Client(base_url=ollama_url, timeout=30) as client:
        shown = client.post("/api/show", json={"model": name})
        shown.raise_for_status()
        show = shown.json()
        tags = client.get("/api/tags").json()["models"]
        version = client.get("/api/version").json()["version"]
    digest = next((m["digest"] for m in tags if m.get("name") == name), None)
    problems = (
        [] if digest == QUANTS[quant][4] else [f"digest {digest} is not the recorded {QUANTS[quant][4]}"]
    )
    if show.get("template") != TEMPLATE:
        problems.append("template differs")
    parameters = show.get("parameters", "")
    problems += [f"stop {stop} missing" for stop in STOPS if f'"{stop}"' not in parameters]
    if show.get("model_info", {}).get("general.architecture") != ARCHITECTURE:
        problems.append("architecture differs")
    result = {"model": name, "digest": digest, "ollama": version, "problems": problems}
    print(json.dumps(result, ensure_ascii=False))
    if problems or digest is None:
        raise SystemExit(f"{name} is not set up as eval/hymt_setup.py writes it: {problems}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="fetch the pinned GGUF files and check their SHA-256")
    commands.add_parser("modelfile", help="write Modelfile.<quant> next to each checked GGUF file")
    checking = commands.add_parser("check", help="compare the Ollama model with this file, print its digest")
    checking.add_argument("--quant", choices=list(QUANTS), default="q8_0")
    checking.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()
    if args.command == "download":
        download()
    elif args.command == "modelfile":
        write_modelfiles()
    else:
        check(args.quant, args.ollama_url)


if __name__ == "__main__":
    main()
