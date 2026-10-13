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

# [파일 역할] Hy-MT2-1.8B 번역 시험을 위해 Tencent 공식 GGUF 파일을 고정 리비전으로 받아 검증하고(download),
# Ollama Modelfile을 쓰고(modelfile), Ollama에 만든 모델이 이 파일이 정한 그대로인지 확인하는(check) 명령줄
# 도구.
# [설계 이유] 시험 결과가 '어떤 빌드를 쟀는지'에 묶이도록 파일은 리비전·크기·SHA-256으로, Ollama 빌드는
# digest로 고정한다. 커뮤니티 Ollama 패키지 대신 모델 자체의 채팅 템플릿과 종료 토큰으로 Modelfile을 직접
# 만든다(위 docstring). 서버는 HYMT_DIGEST로 같은 빌드인지
# 확인한다(app/services/translation.HyMtTranslator.prepare).
# [연결] 요청 옵션은 app/services/translation.HY_MT_OPTIONS를 그대로 Modelfile PARAMETER로 써서
# 서버·측정·빌드가 같은 값을 쓴다. eval/mt_eval.py가 QUANTS의 모델 이름·digest를 쓴다.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.services.translation import HY_MT_OPTIONS
from eval.common import MODELS

# 받을 저장소와 고정 리비전, 저장 폴더(data/models/hy-mt2, 커밋하지 않음).
REPO = "tencent/Hy-MT2-1.8B-GGUF"  # Apache-2.0 (LICENSE.txt read 2026-10-08)
REVISION = "a0c709d9fac510f2c807aa3af52872340dc37a4a"
FOLDER = MODELS / "hy-mt2"
# quant: (file, SHA-256 from the Hugging Face API, size in bytes, Ollama model name, digest of the build that
# Ollama 0.35.1 makes from the Modelfile below; first seen in a container without GPU, 2026-10-08)
# 양자화 두 가지: Q8_0(기본 시험 팔)과 Q4_K_M(지연에서만 떨어질 때 재는 대체 팔, docs/experiments.md 15절).
# 튜플 순서는 바로 위 영어 주석과 같다.
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
# 모델 토크나이저의 특수 토큰(문장 시작, 사용자 차례, 조수 차례, 차례 끝). 실제 차례 끝 토큰은 'place holder
# no 2'라는 이름의 토큰이다(오른쪽 주석: GGUF의 eos_token_id).
BOS = "<｜hy_begin▁of▁sentence｜>"
USER = "<｜hy_User｜>"
ASSISTANT = "<｜hy_Assistant｜>"
END_OF_TURN = "<｜hy_place▁holder▁no▁2｜>"  # eos_token_id 120020 in the GGUF and the tokenizer
# GGUF에 든 채팅 템플릿을 Ollama의 Go 템플릿으로 옮긴 것: 문장 시작 토큰 뒤에 메시지마다 사용자/조수 토큰과
# 내용을 붙이고, 마지막에 조수 토큰을 두어 모델이 답을 이어 쓰게 한다. 시스템 프롬프트 갈래는 없다(모델에 기본
# 시스템 프롬프트가 없음).
# STOPS: 차례 끝 토큰과, 모델이 다음 사용자 차례를 지어내기 시작하면 멈추도록 사용자 토큰도 멈춤 문자열로
# 둔다. ARCHITECTURE: check가 Ollama의 모델 정보와 대조하는 구조 이름.
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


# 큰 파일을 1MiB씩 읽어 SHA-256을 계산한다(약 3GB를 한꺼번에 메모리에 올리지 않음).
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# 파일이 있고 크기와 SHA-256이 고정값과 같은지 확인한다. 크기를 먼저 비교해 다르면 오래 걸리는 해시 계산을
# 건너뛴다(or의 단락 평가).
def verify(quant: str, folder: Path = FOLDER) -> Path:
    name, expected, size, _, _ = QUANTS[quant]
    path = folder / name
    if not path.is_file() or path.stat().st_size != size or sha256(path) != expected:
        raise ValueError(f"{path} is missing or not the pinned file (SHA-256 {expected})")
    return path


# 양자화마다: 이미 검증을 통과하는 파일이 있으면 건너뛰고, 아니면 고정 리비전 주소(resolve/<리비전>)에서
# .part로 받은 뒤 이름을 바꾸고 다시 검증한다.
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


# Modelfile 내용: FROM(같은 폴더의 GGUF), TEMPLATE, 멈춤 문자열, 서비스와 같은 요청 옵션(temperature 0,
# repeat_penalty, num_predict, num_ctx).
def modelfile_text(quant: str) -> str:
    """The Modelfile, with FROM relative to the Modelfile's own folder (the GGUF's folder)."""
    lines = [f"FROM ./{QUANTS[quant][0]}", f'TEMPLATE """{TEMPLATE}"""']
    lines += [f'PARAMETER stop "{stop}"' for stop in STOPS]
    lines += [f"PARAMETER {key} {value}" for key, value in HY_MT_OPTIONS.items()]
    return "\n".join(lines) + "\n"


# 검증된 GGUF 옆에 Modelfile.<양자화>를 쓴다. 그 뒤 `ollama create`는 사람이 실행한다(위 docstring의 명령).
def write_modelfiles(folder: Path = FOLDER) -> None:
    for quant in QUANTS:
        verify(quant, folder)
        path = folder / f"Modelfile.{quant}"
        path.write_text(modelfile_text(quant), encoding="utf-8")
        print(f"wrote {path}")


# Ollama가 보고하는 모델 정보(/api/show의 템플릿·매개변수·구조, /api/tags의 digest, /api/version)를 이 파일의
# 값과 대조한다. 다른 점이 하나라도 있거나 모델이 목록에 없으면 SystemExit으로 멈춘다. 결과(digest, Ollama
# 버전, 문제 목록)는 JSON 한 줄로 출력해 측정 기록에 남길 수 있게 한다.
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


# 하위 명령 download / modelfile / check를 고른다. 도움말에 모듈 docstring(사용 순서)을 그대로 보여 준다.
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
