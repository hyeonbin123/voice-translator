"""Supertonic 3 speech synthesis with ONNX Runtime on the CPU (T78, docs/experiments.md 12) or, as a trial,
on the GPU through ONNX Runtime's CUDA provider (T82, docs/experiments.md 14).

Supertonic 3 (Supertone Inc., released 2026-04-29) is four ONNX graphs: a duration predictor, a text encoder,
a vector estimator that turns noise into a speech latent in N flow-matching steps, and a vocoder. The model
files are under the BigScience OpenRAIL-M license, whose use restrictions bind this service and its users
(README). Supertone archived the project on 2026-09-09: there are no upstream fixes, so the files are pinned
by revision and SHA-256 below and loaded from a local copy (eval/supertonic_download.py).

The text front end and the inference steps follow the archived reference code, py/helper.py of
github.com/supertone-oss-archive/supertonic at commit 1e9799e964ea4c0dad7cde993b65c3c813a7b373, under this
license:

    MIT License

    Copyright (c) 2025 Supertone Inc.

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
    EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
    THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Changes from the reference: characters the model has no index for are dropped (the reference would pass -1
to the model), each chunk is trimmed to its own predicted length before chunks are joined, the noise comes
from a generator per engine, and one engine synthesizes one text at a time. The reference refuses the GPU
("GPU mode is not fully tested"); here the CUDA provider gets options fixed before any measurement and a load
fails unless every session really runs on it.
"""

# [파일 역할] Supertonic 3 한국어 음성 합성을 ONNX Runtime으로 직접 돌리는 엔진. 파일 고정·검증, 글
# 전처리(정규화·글자 색인), 네 개의 ONNX 그래프(길이 예측 → 글 인코더 → 흐름 정합 N단계 → 보코더) 실행, 긴 글
# 나누기를 맡는다.
# [설계 이유] 두 번의 시험(CPU, GPU)을 위한 비교용 설정이고 서비스 기본은 MeloTTS다(docs/experiments.md
# 12·14절). 만든 곳이 저장소를 보관 처리해 수정이 없으므로, 리비전과 파일별 SHA-256을 고정해 받고 올릴 때마다
# 확인한다(README). 참조 코드(MIT, 위 docstring)를 옮기되 바꾼 점은 docstring 끝에 적혀 있다.
# [연결] services/tts.SupertonicTextToSpeech가 SupertonicEngine.load로 쓰고, eval/supertonic_download.py가
# REPO·REVISION·PINNED_FILES·verify_files로 파일을 받아 검증한다. 테스트: tests/test_supertonic.py.

from __future__ import annotations

import hashlib
import json
import re
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# numpy: 그래프 입출력(ONNX Runtime은 numpy 배열을 주고받음)과 잡음 생성. onnxruntime은 설치되지 않은
# 환경에서도 모듈을 불러올 수 있게 load() 안에서 import한다.
import numpy as np

REPO = "supertone-oss-archive/supertonic-3"
# The archive's own pinned snapshot (its README, 2026-09-09); the same files as Supertone/supertonic-3 at
# 3cadd1ee6394adea1bd021217a0e650ede09a323.
REVISION = "aafc6e32416a594460b32413efc49d7fe4ce6d46"
# Every file this project reads, by SHA-256, checked when downloaded and when loaded.
# 이 프로젝트가 읽는 파일 전부와 그 SHA-256. 받을 때와 올릴 때 모두 대조해, 다른 파일이 섞이면 합성을 끈 채
# 뜬다(docs/api.md SUPERTONIC_DIR).
PINNED_FILES: dict[str, str] = {
    "onnx/duration_predictor.onnx": "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
    "onnx/text_encoder.onnx": "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
    "onnx/vector_estimator.onnx": "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c",
    "onnx/vocoder.onnx": "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba",
    "onnx/tts.json": "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09",
    "onnx/unicode_indexer.json": "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f",
    "voice_styles/F1.json": "bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2",
    "LICENSE": "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f",
}
# 모델을 이루는 네 그래프의 이름(파일 이름과 같음).
ONNX_PARTS = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")
# Fixed before any listening (docs/experiments.md 12): the first female preset, since the service's other
# voices (MeloTTS KR, Kokoro af_heart) are female too.
DEFAULT_VOICE = "F1"
# The reference code's defaults.
# 참조 코드의 기본값: 말 속도 1.05(길이 예측을 이 값으로 나눔), 조각 사이 무음 0.3초, 한국어·일본어 조각 최대
# 120자(그 밖의 언어 300자). PROVIDERS: 허용하는 ONNX Runtime 제공자.
SPEED = 1.05
SILENCE_S = 0.3
MAX_CHUNK_CHARS = {"ko": 120, "ja": 120}
OTHER_MAX_CHUNK_CHARS = 300
PROVIDERS = ("cpu", "cuda")
# T82 (docs/experiments.md 14), fixed before any GPU run. ONNX Runtime's default convolution search
# (EXHAUSTIVE) searches again for every new input shape, and speech changes shape with every sentence; the
# default arena grows by powers of two; gpu_mem_limit caps the arena (weights about 400 MB in fp32 plus the
# activations), not the CUDA context or the cuBLAS and cuDNN handles.
# GPU 시험용 CUDA 제공자 옵션(측정 전에 고정, 이유는 위 영어 주석): 합성곡 알고리즘은 입력 모양이 바뀔 때마다
# 다시 찾지 않도록 HEURISTIC, 메모리 아레나는 요청한 만큼만 늘리고, 아레나 상한은 1GiB(1 << 30바이트).
CUDA_PROVIDER_OPTIONS: dict[str, Any] = {
    "device_id": 0,
    "cudnn_conv_algo_search": "HEURISTIC",
    "arena_extend_strategy": "kSameAsRequested",
    "gpu_mem_limit": 1 << 30,
}
# 모델이 아는 언어 태그. 이 서비스는 한국어만 Supertonic으로 합성한다.
LANGUAGES = frozenset(
    "en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro ru sk sl sv tr uk vi na".split()
)

# 참조 코드의 글 정규화 규칙: 이모지는 지우고, 대시·둥근 따옴표 등은 기본 문자로, 괄호·슬래시 같은 기호는
# 공백으로, 장식 문자는 지우고, 몇몇 표현(@, e.g.,)은 말로 풀어 읽게 한다. _SPACE_BEFORE: 문장부호 앞 공백을
# 없앨 부호. _ENDING: 글이 이미 문장 끝 부호로 끝나는지.
_EMOJI = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff☀-⛿✀-➿\U0001f1e6-\U0001f1ff]+"
)
_SYMBOLS = {
    "–": "-",
    "‑": "-",
    "—": "-",
    "_": " ",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "´": "'",
    "`": "'",
    "[": " ",
    "]": " ",
    "|": " ",
    "/": " ",
    "#": " ",
    "→": " ",
    "←": " ",
}
_DECORATIONS = re.compile(r"[♥☆♡©\\]")
_EXPRESSIONS = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}
_SPACE_BEFORE = (",", ".", "!", "?", ";", ":", "'")
_ENDING = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")
# 문장 경계: . ! ? 뒤의 공백. 앞이 Mr. Dr. etc. 같은 약어나 대문자 이니셜이면 고정 길이 lookbehind를 여러 개
# 이어 나누지 않는다.
_SENTENCE_BREAK = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)"
    r"(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)"
    r"(?<=[.!?])\s+"
)


# onnxruntime.InferenceSession에서 이 엔진이 쓰는 run 메서드만 정의한 Protocol. 테스트가 가짜 세션을 넣을 수
# 있게 한다.
class Session(Protocol):
    def run(self, output_names: Any, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


# 목소리 프리셋(voice_styles/*.json)의 두 스타일 벡터. 배열 모양은 오른쪽 주석과 같다.
@dataclass(frozen=True)
class Voice:
    ttl: np.ndarray  # style for the text encoder and the vector estimator, (1, 50, 256)
    dp: np.ndarray  # style for the duration predictor, (1, 8, 16)


# 고정한 파일이 모두 있고 SHA-256이 맞는지 확인한다. 큰 ONNX 파일을 한꺼번에 메모리에 올리지 않도록 1MiB씩
# 읽어 해시한다. 없으면 FileNotFoundError(받는 명령을 안내), 다르면 ValueError.
def verify_files(model_dir: Path, files: Mapping[str, str]) -> None:
    """Raise unless every pinned file is there with its SHA-256."""
    for name, expected in files.items():
        path = Path(model_dir) / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Supertonic file {name} is missing in {model_dir} (eval.supertonic_download)"
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Supertonic file {name} does not match its pinned SHA-256")


# 긴 글을 모델에 넣을 조각으로 나눈다: 빈 줄로 문단을 나누고, 문단 안에서는 문장을 max_len 글자 안에 들어가는
# 만큼 이어 붙인다. 한 문장이 max_len보다 길면 그 문장 하나가 한 조각이 된다(더 자르지 않음).
def chunk_text(text: str, max_len: int) -> list[str]:
    """The reference's chunker: paragraphs, then sentences joined while they fit in max_len characters."""
    chunks = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n+", text.strip())):
        if not paragraph:
            continue
        current = ""
        for sentence in _SENTENCE_BREAK.split(paragraph):
            if len(current) + len(sentence) + 1 <= max_len:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
    return chunks


# 글을 모델 입력으로 바꾼다. indexer는 유니코드 코드 포인트 → 모델 글자 번호 표(unicode_indexer.json)이고,
# -1이면 모델이 모르는 글자다.
class TextProcessor:
    """Text to the model's input: the reference's normalization, then one index per character."""

    def __init__(self, indexer: list[int]) -> None:
        self._indexer = indexer

    def _readable(self, char: str) -> bool:
        code = ord(char)
        return code < len(self._indexer) and self._indexer[code] != -1

    # 순서: NFKD 정규화(한글 음절을 자모로 분해) → 참조 규칙으로 기호 정리 → 모델이 모르는 글자는 공백으로
    # 바꾸고 그 수를 센다(참조 코드는 -1을 그대로 넘겼지만 여기서는 버림) → 부호 앞 공백 제거, 겹친 따옴표
    # 하나로, 공백 정리 → 끝 부호가 없으면 마침표를 붙임 → <ko>...</ko>처럼 언어 태그로 감싼다. 읽을 글자가
    # 하나도 없으면 ValueError.
    def prepare(self, text: str, language: str) -> tuple[str, int]:
        """The tagged text the model reads, and how many characters it could not read and dropped."""
        if language not in LANGUAGES:
            raise ValueError(f"Supertonic does not know the language {language}")
        text = unicodedata.normalize("NFKD", text)
        text = _EMOJI.sub("", text)
        for old, new in _SYMBOLS.items():
            text = text.replace(old, new)
        text = _DECORATIONS.sub("", text)
        for old, new in _EXPRESSIONS.items():
            text = text.replace(old, new)
        dropped = sum(1 for char in text if not char.isspace() and not self._readable(char))
        text = "".join(char if self._readable(char) else " " for char in text)
        for mark in _SPACE_BEFORE:
            text = text.replace(f" {mark}", mark)
        text = re.sub(r"([\"'`])\1+", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("there is nothing the model can read in the text")
        if not _ENDING.search(text):
            text += "."
        return f"<{language}>{text}</{language}>", dropped

    # 태그를 포함한 글을 글자 번호 배열 (1, 길이)와 모두 1인 마스크 (1, 1, 길이)로 바꾼다. 배치 크기는 1이다.
    def ids(self, prepared: str) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([[self._indexer[ord(char)] for char in prepared]], dtype=np.int64)
        return ids, np.ones((1, 1, ids.shape[1]), dtype=np.float32)


# 목소리 JSON의 data(평탄한 숫자 목록)를 dims 모양의 float32 배열로 되살린다.
def load_voice(path: Path) -> Voice:
    style = json.loads(Path(path).read_text(encoding="utf-8"))

    def array(key: str) -> np.ndarray:
        return np.array(style[key]["data"], dtype=np.float32).reshape(style[key]["dims"])

    return Voice(ttl=array("style_ttl"), dp=array("style_dp"))


# 합성 엔진. 설정(tts.json)에서 표본 추출률과 잠재 공간 크기를 읽는다: 잠재 프레임 하나가 표본 base_chunk_size
# × chunk_compress_factor개에 해당하고, 잠재 채널 수는 latent_dim × chunk_compress_factor다. rng를 엔진마다
# 두어 잡음을 만들고(테스트는 0만 내는 가짜 생성기를 넣는다), Lock으로 한 번에 한 글만 합성한다.
class SupertonicEngine:
    """Speech for one text at a time. ONNX Runtime releases the GIL while a graph runs. The graphs' inputs and
    outputs are numpy arrays on either provider, so on the GPU each run copies them to and from the device."""

    def __init__(
        self,
        config: Mapping[str, Any],
        processor: TextProcessor,
        sessions: Mapping[str, Session],
        voice: Voice,
        rng: Any = None,
        providers: Mapping[str, list[str]] | None = None,
    ) -> None:
        self.sample_rate: int = config["ae"]["sample_rate"]
        # What each session reports it runs on (ONNX Runtime's get_providers), for the measurement records.
        self.providers = dict(providers or {})
        compress = config["ttl"]["chunk_compress_factor"]
        self._samples_per_frame = config["ae"]["base_chunk_size"] * compress
        self._latent_channels = config["ttl"]["latent_dim"] * compress
        self.processor = processor
        self._sessions = sessions
        self._voice = voice
        self._rng = rng if rng is not None else np.random.default_rng()
        self._lock = threading.Lock()

    # 파일 검증 → onnxruntime 세션 네 개 생성 → (GPU면) 모든 세션이 실제로 CUDA 제공자로 올라갔는지 확인 →
    # 설정·색인·목소리를 읽어 엔진을 만든다.
    # ONNX Runtime은 CUDA 제공자를 못 쓰면 경고만 하고 CPU로 돌기 때문에, 첫 제공자가 CUDA가 아닌 세션이
    # 하나라도 있으면 RuntimeError로 적재를 실패시킨다(조용히 CPU로 돌면 GPU 시험 결과가 틀려짐).
    # preload_dlls()는 nvidia pip 패키지의 CUDA·cuDNN 라이브러리를 미리 불러온다.
    # intra_op_num_threads는 한 연산 안의 병렬 스레드 수(SUPERTONIC_THREADS), inter_op는 1로 고정한다.
    @classmethod
    def load(
        cls,
        model_dir: Path,
        *,
        voice: str = DEFAULT_VOICE,
        threads: int = 2,
        verify: bool = True,
        provider: str = "cpu",
    ) -> SupertonicEngine:
        """provider "cpu" (T78) or "cuda" (T82): the CUDA provider needs the onnxruntime-gpu package, and the
        load fails unless every session runs on it (ONNX Runtime would only warn and use the CPU)."""
        model_dir = Path(model_dir)
        if provider not in PROVIDERS:
            raise ValueError(f"the ONNX Runtime provider must be one of {PROVIDERS}, not {provider}")
        if f"voice_styles/{voice}.json" not in PINNED_FILES:
            raise ValueError(f"the voice {voice} is not pinned")
        if verify:
            verify_files(model_dir, PINNED_FILES)
        import onnxruntime

        if provider == "cuda":
            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                raise RuntimeError(
                    "this ONNX Runtime has no CUDAExecutionProvider: the GPU needs onnxruntime-gpu"
                )
            # The CUDA and cuDNN libraries of the nvidia pip packages, which the image already carries.
            onnxruntime.preload_dlls()
            providers: list[Any] = [
                ("CUDAExecutionProvider", dict(CUDA_PROVIDER_OPTIONS)),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        sessions = {
            part: onnxruntime.InferenceSession(
                str(model_dir / "onnx" / f"{part}.onnx"), sess_options=options, providers=providers
            )
            for part in ONNX_PARTS
        }
        active = {part: list(session.get_providers()) for part, session in sessions.items()}
        if provider == "cuda":
            off = sorted(part for part, names in active.items() if names[:1] != ["CUDAExecutionProvider"])
            if off:
                raise RuntimeError(
                    f"the Supertonic sessions {off} do not run on CUDAExecutionProvider: {active[off[0]]}"
                )
        config = json.loads((model_dir / "onnx" / "tts.json").read_text(encoding="utf-8"))
        indexer = json.loads((model_dir / "onnx" / "unicode_indexer.json").read_text(encoding="utf-8"))
        return cls(
            config,
            TextProcessor(indexer),
            sessions,
            load_voice(model_dir / "voice_styles" / f"{voice}.json"),
            providers=active,
        )

    # 그래프 하나를 실행하고 첫 번째 출력만 돌려준다(output_names=None은 모든 출력).
    def _run(self, part: str, **feeds: np.ndarray) -> np.ndarray:
        return self._sessions[part].run(None, feeds)[0]

    # 글 조각 하나의 합성: (1) 길이 예측 그래프로 말 길이(초)를 구하고 speed로 나눠 조금 빠르게 (2) 글
    # 인코더로 글 임베딩 (3) 길이에 맞는 잠재 프레임 수만큼 표준 정규 잡음을 만든다(프레임 수는 올림) (4) 벡터
    # 추정 그래프를 steps번 돌려 잡음을 말 잠재 표현으로 바꾼다(흐름 정합, 단계가 적을수록 빠르고 음질이 낮음)
    # (5) 보코더로 파형을 만들고 예측 길이(samples)로 자른다. 조각마다 자기 예측 길이로 자르는 것이 참조
    # 코드와 바꾼 점이다.
    def _speak(self, text: str, language: str, steps: int, speed: float) -> np.ndarray:
        ids, mask = self.processor.ids(self.processor.prepare(text, language)[0])
        voice = self._voice
        duration = self._run("duration_predictor", text_ids=ids, style_dp=voice.dp, text_mask=mask) / speed
        text_emb = self._run("text_encoder", text_ids=ids, style_ttl=voice.ttl, text_mask=mask)
        samples = int((duration * self.sample_rate).astype(np.int64)[0])
        frames = (samples + self._samples_per_frame - 1) // self._samples_per_frame
        latent_mask = np.ones((1, 1, frames), dtype=np.float32)
        latent = self._rng.standard_normal((1, self._latent_channels, frames), dtype=np.float32) * latent_mask
        total = np.array([steps], dtype=np.float32)
        for step in range(steps):
            latent = self._run(
                "vector_estimator",
                noisy_latent=latent,
                text_emb=text_emb,
                style_ttl=voice.ttl,
                text_mask=mask,
                latent_mask=latent_mask,
                current_step=np.array([step], dtype=np.float32),
                total_step=total,
            )
        return self._run("vocoder", latent=latent)[0, :samples].astype(np.float32)

    # 글 전체 합성: 언어별 최대 길이로 조각을 나눠 차례로 합성하고 사이에 무음(silence_s)을 넣어 잇는다. Lock
    # 안에서 돌아 같은 엔진의 잡음 생성기와 세션을 동시에 쓰지 않는다. ONNX Runtime은 그래프를 도는 동안 GIL을
    # 놓으므로(위 docstring) 그동안 이벤트 루프와 다른 스레드가 계속 돈다.
    def synthesize(
        self, text: str, language: str, steps: int, speed: float = SPEED, silence_s: float = SILENCE_S
    ) -> np.ndarray:
        """Mono float samples at sample_rate: long text in chunks, joined by silence_s of silence."""
        chunks = chunk_text(text, MAX_CHUNK_CHARS.get(language, OTHER_MAX_CHUNK_CHARS))
        if not chunks:
            raise ValueError("there is nothing the model can read in the text")
        gap = np.zeros(int(silence_s * self.sample_rate), dtype=np.float32)
        pieces: list[np.ndarray] = []
        with self._lock:  # one generator and one set of sessions per engine
            for index, chunk in enumerate(chunks):
                if index:
                    pieces.append(gap)
                pieces.append(self._speak(chunk, language, steps, speed))
        return np.concatenate(pieces)
