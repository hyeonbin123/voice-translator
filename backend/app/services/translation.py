"""Text translation (task T3).

The candidates and the selection rule are in docs/experiments.md. Seq2seq models (opus-mt, NLLB) run on
CTranslate2, the same engine as speech recognition, after a one-off conversion (eval/mt_convert.py).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import httpx

from app.services.cuda import add_cuda_dll_dirs
from app.services.interfaces import Language, ModelError, Translator

T = TypeVar("T")

LANGUAGE_NAMES = {"ko": "Korean", "en": "English"}
NLLB_CODES = {"ko": "kor_Hang", "en": "eng_Latn"}
EOS = "</s>"

# The character rules of OPUS-MT's preprocess.sh, which the Marian models were trained with.
# Its "１" -> '"' rule looks like a typo but is what training saw, so it is kept.
_OPUS_CHARS = str.maketrans(
    {
        "，": ",", "、": ",", "”": '"', "“": '"', "∶": ":", "：": ":", "？": "?", "《": '"', "》": '"',
        "）": ")", "！": "!", "（": "(", "；": ";", "１": '"', "」": '"', "「": '"', "０": "0", "３": "3",
        "２": "2", "５": "5", "６": "6", "９": "9", "７": "7", "８": "8", "４": "4", "～": "~", "’": "'",
        "…": "...", "━": "-", "〈": "<", "〉": ">", "【": "[", "】": "]", "％": "%",
    }
)  # fmt: skip


def opus_preprocess(text: str) -> str:
    """Python port of OPUS-MT's preprocess.sh: ASCII punctuation, no control characters, single spaces."""
    text = re.sub(r"[。．] *", ". ", text).translate(_OPUS_CHARS)
    # Unicode "other" characters (control, format such as zero-width space) become spaces.
    text = "".join(" " if unicodedata.category(c).startswith("C") and c != "\n" else c for c in text)
    return re.sub(r" +", " ", text).strip(" ")


# Sentence ends: . ? ! (and full-width forms) followed by white space (docs/experiments.md 2-1, T17).
_SENTENCE_END = re.compile(r"(?<=[.?!。？！])\s+")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e"}


def _is_abbreviation(word: str) -> bool:
    # Initials and dotted capitals ("J.", "U.N.") or a common abbreviation ("Dr.", "e.g.").
    return word.endswith(".") and (
        re.fullmatch(r"(?:[A-Z]\.)+", word) is not None or word[:-1].lower() in _ABBREVIATIONS
    )


def split_sentences(text: str) -> list[str]:
    """Split before each sentence, but not after an abbreviation; a number such as 3.5 has no space."""
    sentences: list[str] = []
    for piece in _SENTENCE_END.split(text.strip()):
        # The last word whatever white space precedes it: a newline or tab before "Dr." too (T27).
        if sentences and _is_abbreviation(sentences[-1].split()[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        elif piece:
            sentences.append(piece)
    return sentences


def load_sentencepiece(path: Path):
    import sentencepiece

    # Pass the bytes: SentencePiece's own file loader can't open non-ASCII paths on Windows.
    return sentencepiece.SentencePieceProcessor(model_proto=path.read_bytes())


def marian_source_tokens(processor, text: str) -> list[str]:
    # preprocess.sh, then what transformers' MarianTokenizer does: SentencePiece pieces and end-of-sentence.
    return processor.encode(opus_preprocess(text), out_type=str) + [EOS]


def nllb_source_tokens(processor, text: str, language: Language) -> list[str]:
    # Same as transformers' NllbTokenizer (non-legacy): language code, pieces, end-of-sentence token.
    return [NLLB_CODES[language], *processor.encode(text, out_type=str), EOS]


def _check_text(text: str, source: Language, target: Language) -> None:
    if source == target:
        raise ModelError("source and target languages must differ")
    if not text.strip():
        raise ModelError("text is empty")


def _nonempty(translation: object) -> str:
    if not isinstance(translation, str) or not translation.strip():
        raise ModelError("the model returned an empty translation")
    return translation.strip()


def _guarded(run: Callable[[], T]) -> T:
    """Run an engine or tokenizer call; any failure inside it becomes a ModelError.

    CTranslate2 raises RuntimeError for CUDA errors such as running out of memory. Callers (the
    pipeline) only need to know that translation failed, to answer 503 without a history record.
    """
    try:
        return run()
    except ModelError:
        raise
    except Exception as exc:  # noqa: BLE001 - the boundary around third-party engines
        raise ModelError(f"the translation model failed: {type(exc).__name__}") from exc


class _Ct2Model:
    def __init__(self, model_dir: Path, device: str, compute_type: str, beam_size: int) -> None:
        import ctranslate2

        if device == "cuda":
            add_cuda_dll_dirs()
        self.model = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type)
        self.beam_size = beam_size

    def generate(self, tokens: list[str], target_prefix: list[str] | None = None) -> list[str]:
        results = self.model.translate_batch(
            [tokens],
            target_prefix=[target_prefix] if target_prefix else None,
            beam_size=self.beam_size,
            max_decoding_length=256,
        )
        return results[0].hypotheses[0]


class MarianTranslator:
    """One opus-mt model, which translates in one direction only.

    With by_sentence, the input is split into sentences, each translated on its own and joined with a
    space: opus-mt was trained on single sentences (docs/experiments.md 2-1, T17).
    """

    def __init__(
        self,
        model_dir: Path,
        source: Language,
        target: Language,
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 4,
        by_sentence: bool = False,
    ) -> None:
        self.direction = (source, target)
        self.model_name = f"ctranslate2/{model_dir.name}"
        self.by_sentence = by_sentence
        self._model = _Ct2Model(model_dir, device, compute_type, beam_size)
        self._source_sp = load_sentencepiece(model_dir / "source.spm")
        self._target_sp = load_sentencepiece(model_dir / "target.spm")

    def translate(self, text: str, source: Language, target: Language) -> str:
        _check_text(text, source, target)
        if (source, target) != self.direction:
            raise ModelError(f"{self.model_name} only translates {self.direction[0]}->{self.direction[1]}")
        sentences = split_sentences(text) if self.by_sentence else [text]
        return " ".join(self._translate_one(sentence) for sentence in sentences)

    def _translate_one(self, text: str) -> str:
        pieces = _guarded(lambda: self._model.generate(marian_source_tokens(self._source_sp, text)))
        return _nonempty(_guarded(lambda: self._target_sp.decode(pieces)))


class NllbTranslator:
    """NLLB-200 translates in both directions with one model."""

    def __init__(
        self, model_dir: Path, device: str = "cuda", compute_type: str = "float16", beam_size: int = 4
    ) -> None:
        self.model_name = f"ctranslate2/{model_dir.name}"
        self._model = _Ct2Model(model_dir, device, compute_type, beam_size)
        self._sp = load_sentencepiece(model_dir / "sentencepiece.bpe.model")

    def translate(self, text: str, source: Language, target: Language) -> str:
        _check_text(text, source, target)
        tokens = _guarded(lambda: nllb_source_tokens(self._sp, text, source))
        pieces = _guarded(lambda: self._model.generate(tokens, target_prefix=[NLLB_CODES[target]]))
        return _nonempty(_guarded(lambda: self._sp.decode(pieces[1:])))  # drop the target language code


class OllamaTranslator:
    """An instruction-tuned LLM served by Ollama, prompted to return the translation only."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout_s: float = 60) -> None:
        self.model_name = f"ollama/{model}"
        self._model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def translate(self, text: str, source: Language, target: Language) -> str:
        _check_text(text, source, target)
        system = (
            f"You are a translator. Translate the user's {LANGUAGE_NAMES[source]} text into "
            f"{LANGUAGE_NAMES[target]}. Reply with the translation only, without notes or quotes."
        )
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 512},
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ModelError(f"the translation model is unavailable: {type(exc).__name__}") from exc
        return _nonempty(content)


# Hy-MT2 (T83, docs/experiments.md 15): the model card's "Default Translation" English prompt, as the only
# message (the model has no default system prompt), with the target language's English name.
HY_MT_PROMPT = (
    "Translate the following text into {target}. Note that you should only output the translated result "
    "without any additional explanation:\n\n{text}"
)
# Greedy (the card samples at 0.7; the service needs the same output for the same input), the card's
# repetition penalty, opus's output cap of 256 tokens, and a fixed context size: the longest request (500
# characters, docs/api.md) with the prompt and the cap fits well inside 2048 tokens.
HY_MT_OPTIONS = {"temperature": 0, "repeat_penalty": 1.05, "num_predict": 256, "num_ctx": 2048}


class HyMtTranslator:
    """Tencent's Hy-MT2 (Apache-2.0), a translation model served by Ollama, a trial (T83).

    Ollama runs the official GGUF through a Modelfile with the model's own chat template and its end-of-turn
    token as the stop (eval/hymt_setup.py). With by_sentence the input is split as for opus en->ko (T17).
    `calls` keeps what Ollama reported for each call of the last translation, for the evaluation tools.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30,
        by_sentence: bool = False,
        expected_digest: str | None = None,
        prepare_timeout_s: float = 300,
    ) -> None:
        self.model_name = f"ollama/{model}" + ("/split" if by_sentence else "")
        self.by_sentence = by_sentence
        self.digest: str | None = None
        self.calls: list[dict] = []
        self._model = model
        self._expected_digest = expected_digest
        self._timeout_s = timeout_s
        self._prepare_timeout_s = prepare_timeout_s
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def translate(self, text: str, source: Language, target: Language) -> str:
        _check_text(text, source, target)
        self.calls = []
        sentences = split_sentences(text) if self.by_sentence else [text]
        return " ".join(self._translate_one(sentence, target) for sentence in sentences)

    def _translate_one(self, text: str, target: Language) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": HY_MT_PROMPT.format(target=LANGUAGE_NAMES[target], text=text)}
            ],
            "stream": False,
            "keep_alive": -1,
            "options": dict(HY_MT_OPTIONS),
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
            content = body["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ModelError(f"the translation model is unavailable: {type(exc).__name__}") from exc
        self.calls.append({key: body.get(key) for key in ("done_reason", "prompt_eval_count", "eval_count")})
        # A reply cut at the output cap ("length") is kept, as opus keeps one cut at 256 tokens.
        return _nonempty(content)

    def prepare(self, timeout_s: float | None = None) -> str:
        """Check that Ollama has the model, and the expected build of it; load it to stay; return its digest.

        `timeout_s`, what is left of the startup wait, caps every request here (T86): the checks otherwise
        wait up to the request limit, the load up to `prepare_timeout_s`.
        Raises ModelError while Ollama or the model is not there (worth waiting for), and RuntimeError for
        another build (not worth waiting for).
        """

        def limit(default_s: float) -> float:
            return default_s if timeout_s is None else min(default_s, timeout_s)

        try:
            shown = self._client.post(
                "/api/show", json={"model": self._model}, timeout=limit(self._timeout_s)
            )
            if shown.status_code == 404:
                raise ModelError(
                    f"Ollama has no model {self._model}: create it with eval.hymt_setup "
                    "(docs/experiments.md 15)"
                )
            shown.raise_for_status()
            tags = self._client.get("/api/tags", timeout=limit(self._timeout_s))
            tags.raise_for_status()
            digests = [
                m.get("digest")
                for m in tags.json()["models"]
                if self._model in (m.get("name"), m.get("model"))
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ModelError(f"Ollama is not available: {type(exc).__name__}") from exc
        if not digests or not digests[0]:
            raise ModelError(f"Ollama does not list {self._model}")
        digest = digests[0]
        if self._expected_digest and digest != self._expected_digest:
            raise RuntimeError(f"{self._model} is build {digest}, not the measured {self._expected_digest}")
        try:
            # A request without messages only loads the model; keep_alive -1 keeps it until Ollama stops.
            loaded = self._client.post(
                "/api/generate",
                json={"model": self._model, "keep_alive": -1},
                timeout=limit(self._prepare_timeout_s),
            )
            loaded.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelError(f"Ollama could not load {self._model}: {type(exc).__name__}") from exc
        self.digest = digest
        return digest


class DirectionalTranslator:
    """Routes each direction to its own model, since the best model may differ by direction."""

    def __init__(self, by_direction: dict[tuple[Language, Language], Translator]) -> None:
        self._by_direction = by_direction
        self.model_name = ", ".join(f"{s}->{t}: {m.model_name}" for (s, t), m in by_direction.items())

    def translate(self, text: str, source: Language, target: Language) -> str:
        _check_text(text, source, target)
        model = self._by_direction.get((source, target))
        if model is None:
            raise ModelError(f"no translation model for {source}->{target}")
        return model.translate(text, source, target)
