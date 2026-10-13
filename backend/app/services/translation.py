"""Text translation (task T3).

The candidates and the selection rule are in docs/experiments.md. Seq2seq models (opus-mt, NLLB) run on
CTranslate2, the same engine as speech recognition, after a one-off conversion (eval/mt_convert.py).
"""

# [파일 역할] 번역 모델 구현 모음. 서비스가 쓰는 것은 opus-mt-tc-big(MarianTranslator, 방향별 모델 하나씩)과
# 방향별 라우터(DirectionalTranslator)이고, NLLB-200·Ollama LLM·Hy-MT2는 측정·비교용
# 후보다(docs/experiments.md 2절·15절).
# [설계 이유] seq2seq 모델은 음성 인식과 같은 엔진(CTranslate2)으로 돌려 의존성과 GPU 사용 방식을 맞춘다. 학습
# 때와 같은 입력이 되도록 OPUS-MT의 전처리 규칙과 토크나이저 동작을 파이썬으로 그대로 옮겼다. 엔진 안의 모든
# 오류는 ModelError로 바꿔 API가 503으로 답하게 한다.
# [연결] services/models.load_translation이 서비스용 번역기를 만들고, eval/mt_eval.py가 후보를 비교하며,
# eval/mt_convert.py가 변환 뒤 토큰 확인에 이 파일의 토큰화 함수를 쓴다.

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

# httpx: Ollama HTTP API 호출(Ollama 후보, Hy-MT2). ctranslate2·sentencepiece는 GPU·모델 의존성이라 쓰는
# 곳(함수 안)에서 import한다.
import httpx

from app.services.cuda import add_cuda_dll_dirs
from app.services.interfaces import Language, ModelError, Translator

T = TypeVar("T")

# LLM 지시문에 넣을 언어 이름, NLLB의 언어 코드(FLORES-200 형식), 문장 끝 토큰.
LANGUAGE_NAMES = {"ko": "Korean", "en": "English"}
NLLB_CODES = {"ko": "kor_Hang", "en": "eng_Latn"}
EOS = "</s>"

# 전각·중국어식 문장부호와 숫자를 ASCII로 바꾸는 문자 대응표(str.maketrans). 학습 때 입력과 같게 하려고 원본
# preprocess.sh의 규칙을 그대로 옮겼다(docs/experiments.md 2절).
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


# opus-mt 입력 전처리: (1) 전각 마침표 뒤를 ". "로 (2) 위 대응표 적용 (3) 줄바꿈을 뺀 유니코드 C
# 범주(제어·서식) 문자를 공백으로 (4) 연속 공백을 하나로 줄이고 양끝 공백 제거. schemas/translate.py가 C 범주
# 문자만 있는 입력을 422로 거절하는 것은 이 단계 뒤에 번역할 글자가 남지 않기 때문이다.
def opus_preprocess(text: str) -> str:
    """Python port of OPUS-MT's preprocess.sh: ASCII punctuation, no control characters, single spaces."""
    text = re.sub(r"[。．] *", ". ", text).translate(_OPUS_CHARS)
    # Unicode "other" characters (control, format such as zero-width space) become spaces.
    text = "".join(" " if unicodedata.category(c).startswith("C") and c != "\n" else c for c in text)
    return re.sub(r" +", " ", text).strip(" ")


# 문장 나누기 규칙(docs/experiments.md 2-1절 B): 문장 끝 부호 뒤에 공백·줄바꿈이 오면 나눈다.
# lookbehind(?<=...)라 부호는 앞 문장에 남는다. 이니셜·"U.N." 같은 점 찍힌 대문자와 흔한 약어 뒤에서는 나누지
# 않는다.
# Sentence ends: . ? ! (and full-width forms) followed by white space (docs/experiments.md 2-1, T17).
_SENTENCE_END = re.compile(r"(?<=[.?!。？！])\s+")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e"}


# 단어가 약어로 끝나는지: 점으로 끝나고, '대문자+점'의 반복이거나 약어 목록(소문자로 비교)에 있으면 참.
def _is_abbreviation(word: str) -> bool:
    # Initials and dotted capitals ("J.", "U.N.") or a common abbreviation ("Dr.", "e.g.").
    return word.endswith(".") and (
        re.fullmatch(r"(?:[A-Z]\.)+", word) is not None or word[:-1].lower() in _ABBREVIATIONS
    )


# 먼저 문장 끝 규칙으로 자른 뒤, 앞 조각의 마지막 단어가 약어면 다시 이어 붙인다. 빈 조각은 버린다.
# MarianTranslator(by_sentence)와 HyMtTranslator(hy-mt2-split)가 쓴다.
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


# SentencePiece 토크나이저 모델을 읽는다. 파일 경로 대신 바이트(model_proto)를 넘기는 이유는 아래 주석대로
# Windows의 한글 경로 문제다.
def load_sentencepiece(path: Path):
    import sentencepiece

    # Pass the bytes: SentencePiece's own file loader can't open non-ASCII paths on Windows.
    return sentencepiece.SentencePieceProcessor(model_proto=path.read_bytes())


# opus-mt 입력 토큰: 전처리 → SentencePiece 조각 → 끝에 </s>. transformers의 MarianTokenizer와 같은 결과가
# 되게 맞췄다(eval/mt_convert.py가 변환 뒤 비교에 쓴다).
def marian_source_tokens(processor, text: str) -> list[str]:
    # preprocess.sh, then what transformers' MarianTokenizer does: SentencePiece pieces and end-of-sentence.
    return processor.encode(opus_preprocess(text), out_type=str) + [EOS]


# NLLB 입력 토큰: [출발 언어 코드, 조각들, </s>]. 비교 후보 전용.
def nllb_source_tokens(processor, text: str, language: Language) -> list[str]:
    # Same as transformers' NllbTokenizer (non-legacy): language code, pieces, end-of-sentence token.
    return [NLLB_CODES[language], *processor.encode(text, out_type=str), EOS]


# 모든 번역기 공통 입력 검사: 같은 언어 쌍이나 빈 글이면 ModelError. API 검증을 거치지 않는 호출(측정 도구
# 등)도 같은 규칙을 따르게 한다.
def _check_text(text: str, source: Language, target: Language) -> None:
    if source == target:
        raise ModelError("source and target languages must differ")
    if not text.strip():
        raise ModelError("text is empty")


# 모델이 빈 번역(또는 문자열이 아닌 값)을 내면 실패로 본다. 빈 번역을 기록에 남기지 않기 위해서다.
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


# CTranslate2 번역 엔진 감싸개. 빔 크기 4, 최대 256토큰은 모든 seq2seq 후보에 같게 둔 측정
# 조건이다(docs/experiments.md 2절). target_prefix는 NLLB처럼 도착 언어 코드를 출력 앞에 강제할 때 쓴다. 한
# 문장씩(배치 크기 1) 번역하고 가장 좋은 가설 하나를 돌려준다.
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


# Hugging Face에 올라온 tc-big 변환본은 어휘가 깨져 있어, Helsinki의 원본 배포본을 eval/mt_convert.py로
# CTranslate2로 바꾼 폴더(model_dir)를 읽는다(README). 원문·번역문 어휘가 따로라 source.spm과 target.spm을
# 각각 읽는다.
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

    # 자기 방향이 아니면 ModelError. by_sentence면 문장마다 따로 번역해 공백 하나로 잇는다. 서버는 영→한만
    # 이렇게 한다: 여러 문장을 통째로 넣으면 문장을 빠뜨리는 일이 있었고, 나누니 누락이 줄었다(README 설계
    # 판단).
    def translate(self, text: str, source: Language, target: Language) -> str:
        _check_text(text, source, target)
        if (source, target) != self.direction:
            raise ModelError(f"{self.model_name} only translates {self.direction[0]}->{self.direction[1]}")
        sentences = split_sentences(text) if self.by_sentence else [text]
        return " ".join(self._translate_one(sentence) for sentence in sentences)

    # 토큰화·생성·복원 각 단계를 _guarded로 감싸 어떤 엔진 오류든 ModelError로 바꾼다.
    def _translate_one(self, text: str) -> str:
        pieces = _guarded(lambda: self._model.generate(marian_source_tokens(self._source_sp, text)))
        return _nonempty(_guarded(lambda: self._target_sp.decode(pieces)))


# 비교 후보(서비스에서 쓰지 않음). 한 모델이 두 방향을 모두 번역하므로 도착 언어 코드를 target_prefix로 주고,
# 출력의 첫 토큰(그 코드)은 빼고 복원한다.
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


# 비교 후보(qwen2.5 7B 등, 서비스에서 쓰지 않음). 지시문으로 번역문만 답하게 하고 temperature 0으로 부른다.
# 통신·형식 오류는 ModelError.
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
# Hy-MT2 설정(비교용, 서비스 기본은 opus). 프롬프트는 모델 카드의 기본 번역 프롬프트 그대로, 옵션은 위 영어
# 주석의 이유대로 고정한다. HY_MT_OPTIONS는 eval/hymt_setup.py가 Ollama Modelfile의 PARAMETER로도 써 넣어
# 서비스와 모델 빌드가 같은 값을 쓴다.
HY_MT_PROMPT = (
    "Translate the following text into {target}. Note that you should only output the translated result "
    "without any additional explanation:\n\n{text}"
)
# Greedy (the card samples at 0.7; the service needs the same output for the same input), the card's
# repetition penalty, opus's output cap of 256 tokens, and a fixed context size: the longest request (500
# characters, docs/api.md) with the prompt and the cap fits well inside 2048 tokens.
HY_MT_OPTIONS = {"temperature": 0, "repeat_penalty": 1.05, "num_predict": 256, "num_ctx": 2048}
# A request that starts with no time left before prepare's deadline still gets a valid limit (T87).
HY_MT_MIN_REQUEST_S = 0.01


# Hy-MT2-1.8B를 Ollama로 부르는 번역기(비교용 시험 설정). 시험 결과 COMET-22는 opus보다 높았지만 chrF 차이가
# 유의하지 않아 채택하지 않았다(README 알려진 한계, docs/experiments.md 15절). model_name에 "/split"을 붙여
# 문장 단위 설정을 기록에서 구분한다.
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

    # 번역할 때마다 calls를 비우고, 문장(또는 통째) 단위 호출마다 Ollama가 알려 준 종료 이유와 토큰 수를
    # 쌓는다(측정 도구가 읽음).
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

    # 서버 시작 때 services/models._wait_until_ready가 되풀이해 부른다. 순서: /api/show로 모델 존재 확인 →
    # /api/tags에서 그 모델의 digest를 찾음 → 기대한 빌드인지 비교 → 프롬프트 없는 /api/generate로
    # 적재(keep_alive -1).
    # 기다릴 가치가 있는 실패(Ollama가 아직 안 떴거나 모델이 없음)는 ModelError라 재시도되고, 다른 빌드는
    # 기다려도 바뀌지 않으므로 RuntimeError로 바로 시작을 멈춘다.
    def prepare(self, timeout_s: float | None = None) -> str:
        """Check that Ollama has the model, and the expected build of it; load it to stay; return its digest.

        `timeout_s`, what is left of the startup wait, sets a deadline: each request here (show, tags, the
        load) gets at most the time left when it starts (T86, T87), so slow checks leave the load only the
        rest. Without it the checks wait up to the request limit, the load up to `prepare_timeout_s`.
        Raises ModelError while Ollama or the model is not there (worth waiting for), and RuntimeError for
        another build (not worth waiting for).
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s

        # 요청마다 그 요청이 시작될 때 남은 시간만 제한으로 준다(기본 제한보다 길지 않게, 최소 0.01초). 시작
        # 시점의 남은 시간을 모든 요청에 똑같이 주면 한 시도가 한도의 몇 배까지 길어질 수 있었다.
        def limit(default_s: float) -> float:
            if deadline is None:
                return default_s
            return min(default_s, max(deadline - time.monotonic(), HY_MT_MIN_REQUEST_S))

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
            # /api/tags 목록에서 이름(name 또는 model 필드)이 같은 모델의 digest를 모은다.
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


# 방향(출발, 도착)마다 다른 번역기로 보내는 라우터. 서비스는 한→영·영→한 opus 모델 두 개를 이것으로 묶어
# Translator 하나처럼 쓴다. model_name은 "ko->en: ..., en->ko: ..." 형태로 기록과 /api/health에 나간다.
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
