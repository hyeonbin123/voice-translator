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

import numpy as np

REPO = "supertone-oss-archive/supertonic-3"
# The archive's own pinned snapshot (its README, 2026-09-09); the same files as Supertone/supertonic-3 at
# 3cadd1ee6394adea1bd021217a0e650ede09a323.
REVISION = "aafc6e32416a594460b32413efc49d7fe4ce6d46"
# Every file this project reads, by SHA-256, checked when downloaded and when loaded.
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
ONNX_PARTS = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")
# Fixed before any listening (docs/experiments.md 12): the first female preset, since the service's other
# voices (MeloTTS KR, Kokoro af_heart) are female too.
DEFAULT_VOICE = "F1"
# The reference code's defaults.
SPEED = 1.05
SILENCE_S = 0.3
MAX_CHUNK_CHARS = {"ko": 120, "ja": 120}
OTHER_MAX_CHUNK_CHARS = 300
PROVIDERS = ("cpu", "cuda")
# T82 (docs/experiments.md 14), fixed before any GPU run. ONNX Runtime's default convolution search
# (EXHAUSTIVE) searches again for every new input shape, and speech changes shape with every sentence; the
# default arena grows by powers of two; gpu_mem_limit caps the arena (weights about 400 MB in fp32 plus the
# activations), not the CUDA context or the cuBLAS and cuDNN handles.
CUDA_PROVIDER_OPTIONS: dict[str, Any] = {
    "device_id": 0,
    "cudnn_conv_algo_search": "HEURISTIC",
    "arena_extend_strategy": "kSameAsRequested",
    "gpu_mem_limit": 1 << 30,
}
LANGUAGES = frozenset(
    "en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro ru sk sl sv tr uk vi na".split()
)

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
_SENTENCE_BREAK = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)"
    r"(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)"
    r"(?<=[.!?])\s+"
)


class Session(Protocol):
    def run(self, output_names: Any, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class Voice:
    ttl: np.ndarray  # style for the text encoder and the vector estimator, (1, 50, 256)
    dp: np.ndarray  # style for the duration predictor, (1, 8, 16)


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


class TextProcessor:
    """Text to the model's input: the reference's normalization, then one index per character."""

    def __init__(self, indexer: list[int]) -> None:
        self._indexer = indexer

    def _readable(self, char: str) -> bool:
        code = ord(char)
        return code < len(self._indexer) and self._indexer[code] != -1

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

    def ids(self, prepared: str) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([[self._indexer[ord(char)] for char in prepared]], dtype=np.int64)
        return ids, np.ones((1, 1, ids.shape[1]), dtype=np.float32)


def load_voice(path: Path) -> Voice:
    style = json.loads(Path(path).read_text(encoding="utf-8"))

    def array(key: str) -> np.ndarray:
        return np.array(style[key]["data"], dtype=np.float32).reshape(style[key]["dims"])

    return Voice(ttl=array("style_ttl"), dp=array("style_dp"))


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

    def _run(self, part: str, **feeds: np.ndarray) -> np.ndarray:
        return self._sessions[part].run(None, feeds)[0]

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
