"""글 번역.

후보와 선택 규칙은 docs/experiments.md에 있다. seq2seq 모델(opus-mt, NLLB)은 한 번 변환한
뒤(eval/mt_convert.py) 음성 인식과 같은 엔진인 CTranslate2로 돈다.
"""

# [파일 역할] 번역 모델 구현 모음. 서비스가 쓰는 것은 opus-mt-tc-big(MarianTranslator, 방향별 모델 하나씩)과
# 방향별 라우터(DirectionalTranslator)이고, NLLB-200·Ollama LLM·Hy-MT2는 측정·비교용
# 후보다(docs/experiments.md 2절·15절).
# [설계 이유] seq2seq 모델은 음성 인식과 같은 엔진(CTranslate2)으로 돌려 의존성과 GPU 사용 방식을 맞춘다. 학습
# 때와 같은 입력이 되도록 OPUS-MT의 전처리 규칙과 토크나이저 동작을 파이썬으로 그대로 옮겼다. 엔진 안의 모든
# 오류는 ModelError로 바꿔 API가 503으로 답하게 한다.
# [연결] services/models.load_translation이 서비스용 번역기를 만들고, eval/mt_eval.py가 후보를 비교하며,
# eval/mt_convert.py가 변환 뒤 토큰 확인에 이 파일의 토큰화 함수를 쓴다. MarianTranslator.candidates(빔 탐색의
# n-best 후보)는 디코딩 시험(docs/experiments.md 20절)의 eval/mbr.py만 쓰고, 서비스 경로(translate)는 빔 4의
# 1-best 그대로다. FormalRegisterTranslator(말투 후처리 감싸개, 21절)는 KO_REGISTER=formal일 때만
# services/models.py가 영→한 번역기에 씌운다(기본 asis는 감싸지 않음).

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

# 전각·중국어식 문장부호와 숫자를 ASCII로 바꾸는 문자 대응표(str.maketrans). Marian 모델이 학습 때 쓴
# OPUS-MT preprocess.sh의 문자 규칙을 그대로 옮겨 학습 때 입력과 같게 한다(docs/experiments.md 2절). 그
# 규칙의 "１" -> '"'는 오타처럼 보이지만 학습이 본 그대로라 남겨 둔다.
_OPUS_CHARS = str.maketrans(
    {
        "，": ",", "、": ",", "”": '"', "“": '"', "∶": ":", "：": ":", "？": "?", "《": '"', "》": '"',
        "）": ")", "！": "!", "（": "(", "；": ";", "１": '"', "」": '"', "「": '"', "０": "0", "３": "3",
        "２": "2", "５": "5", "６": "6", "９": "9", "７": "7", "８": "8", "４": "4", "～": "~", "’": "'",
        "…": "...", "━": "-", "〈": "<", "〉": ">", "【": "[", "】": "]", "％": "%",
    }
)  # fmt: skip


def opus_preprocess(text: str) -> str:
    """OPUS-MT preprocess.sh의 파이썬 이식: ASCII 문장부호, 제어 문자 없음, 공백 하나.

    순서: (1) 전각 마침표 뒤를 ". "로 (2) 위 대응표 적용 (3) 줄바꿈을 뺀 유니코드 C 범주(제어·서식) 문자를
    공백으로 (4) 연속 공백을 하나로 줄이고 양끝 공백 제거. schemas/translate.py가 C 범주 문자만 있는 입력을
    422로 거절하는 것은 이 단계 뒤에 번역할 글자가 남지 않기 때문이다.
    """
    text = re.sub(r"[。．] *", ". ", text).translate(_OPUS_CHARS)
    # 유니코드 "기타"(C 범주) 문자(제어, 폭 없는 공백 같은 서식)는 공백이 된다.
    text = "".join(" " if unicodedata.category(c).startswith("C") and c != "\n" else c for c in text)
    return re.sub(r" +", " ", text).strip(" ")


# 문장 나누기 규칙(docs/experiments.md 2-1절 B): 문장 끝 부호(. ? !와 그 전각 형태) 뒤에 공백·줄바꿈이 오면
# 나눈다. lookbehind(?<=...)라 부호는 앞 문장에 남는다. 이니셜·"U.N." 같은 점 찍힌 대문자와 흔한 약어
# 뒤에서는 나누지 않는다.
_SENTENCE_END = re.compile(r"(?<=[.?!。？！])\s+")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e"}


# 단어가 약어로 끝나는지: 점으로 끝나고, 이니셜·점 찍힌 대문자('대문자+점'의 반복, 예: "J.", "U.N.")이거나
# 흔한 약어 목록(소문자로 비교, 예: "Dr.", "e.g.")에 있으면 참.
def _is_abbreviation(word: str) -> bool:
    return word.endswith(".") and (
        re.fullmatch(r"(?:[A-Z]\.)+", word) is not None or word[:-1].lower() in _ABBREVIATIONS
    )


def split_sentences(text: str) -> list[str]:
    """문장마다 나누되 약어 뒤에서는 나누지 않는다. 3.5 같은 숫자는 공백이 없어 나뉘지 않는다.

    먼저 문장 끝 규칙으로 자른 뒤, 앞 조각의 마지막 단어가 약어면 다시 이어 붙인다. 빈 조각은 버린다.
    MarianTranslator(by_sentence)와 HyMtTranslator(hy-mt2-split)가 쓴다.
    """
    sentences: list[str] = []
    for piece in _SENTENCE_END.split(text.strip()):
        # 앞에 어떤 공백이 오든 마지막 단어를 본다: "Dr." 앞이 줄바꿈이나 탭이어도 마찬가지다.
        if sentences and _is_abbreviation(sentences[-1].split()[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        elif piece:
            sentences.append(piece)
    return sentences


# SentencePiece 토크나이저 모델을 읽는다. 파일 경로 대신 바이트(model_proto)를 넘기는 이유는 아래 주석대로
# Windows의 한글 경로 문제다.
def load_sentencepiece(path: Path):
    import sentencepiece

    # 바이트를 넘긴다: SentencePiece 자체 파일 로더는 Windows에서 ASCII가 아닌 경로(한글 경로 등)를 열지
    # 못한다.
    return sentencepiece.SentencePieceProcessor(model_proto=path.read_bytes())


# opus-mt 입력 토큰: 전처리 → SentencePiece 조각 → 끝에 </s>. transformers의 MarianTokenizer와 같은 결과가
# 되게 맞췄다(eval/mt_convert.py가 변환 뒤 비교에 쓴다).
def marian_source_tokens(processor, text: str) -> list[str]:
    return processor.encode(opus_preprocess(text), out_type=str) + [EOS]


# NLLB 입력 토큰: [출발 언어 코드, 조각들, </s>]. transformers의 NllbTokenizer(legacy가 아닌 동작)와 같다.
# 비교 후보 전용.
def nllb_source_tokens(processor, text: str, language: Language) -> list[str]:
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


# n-best 후보 하나가 쓸 만한 번역인지(문자열이고 공백만이 아님). _nonempty와 같은 기준이지만 예외 대신
# 참·거짓을 돌려줘, 후보 중 빈 것만 버릴 때 쓴다(MarianTranslator.candidates).
def _usable(translation: object) -> bool:
    return isinstance(translation, str) and bool(translation.strip())


def _guarded(run: Callable[[], T]) -> T:
    """엔진이나 토크나이저 호출을 실행하고, 그 안의 어떤 실패든 ModelError로 바꾼다.

    CTranslate2는 메모리 부족 같은 CUDA 오류에 RuntimeError를 낸다. 호출하는 쪽(번역 흐름)은 번역이
    실패했다는 것만 알면 되고, 기록 없이 503으로 답한다.
    """
    try:
        return run()
    except ModelError:
        raise
    except Exception as exc:  # noqa: BLE001 - 외부 엔진을 감싸는 예외 경계
        raise ModelError(f"the translation model failed: {type(exc).__name__}") from exc


# n-best 가설 수 검사: 1 이상 빔 크기 이하. 빔 탐색의 마지막 빔에는 빔 크기만큼만 가설이 남으므로 그보다 많이
# 달라고 할 수 없다. 엔진을 부르기 전에 ValueError로 멈춰, 설정 실수가 번역 실패(ModelError)로 묻히지 않게
# 한다.
def _check_nbest(beam_size: int, n: int) -> None:
    if not 1 <= n <= beam_size:
        raise ValueError(f"need between 1 and beam_size={beam_size} hypotheses, got {n}")


# CTranslate2 번역 엔진 감싸개. 빔 크기 4, 최대 256토큰은 모든 seq2seq 후보에 같게 둔 측정
# 조건이다(docs/experiments.md 2절). target_prefix는 NLLB처럼 도착 언어 코드를 출력 앞에 강제할 때 쓴다. 한
# 문장씩(배치 크기 1) 번역하고 가장 좋은 가설 하나를 돌려준다(generate, 서비스가 쓰는 경로). generate_nbest는
# 디코딩 시험(docs/experiments.md 20절)의 측정 도구만 쓴다.
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

    def generate_nbest(self, tokens: list[str], beam_size: int, n: int) -> list[tuple[list[str], float]]:
        """빔 탐색의 가설 n개를 (조각 목록, 엔진 점수)로, 점수가 높은 것부터 돌려준다.

        MBR 디코딩 시험(docs/experiments.md 20절)이 후보를 만드는 데 쓴다. 엔진에는 빔 크기와 가설 수
        (`num_hypotheses`), 점수 반환(`return_scores`)만 더 넘기고, 최대 256토큰을 비롯한 나머지 옵션은
        generate와 같은 기본값이다. 빔 크기는 생성할 때 정한 값이 아니라 인수로 받는다(같은 모델로 빔 4와 빔
        8을 견주기 위해). CTranslate2 문서는 점수를 '가설마다의 점수'라고만 하고 가설 순서를 적지 않아, 점수로
        다시 줄 세운다: 점수가 같으면 엔진이 준 순서를 지키고(안정 정렬), 엔진이 더 많이 돌려줘도 n개까지만
        쓴다. 점수 수가 가설 수와 다르면(점수가 오지 않음) 순위를 매길 수 없어 ValueError다. 가설 수가 1~빔
        크기가 아니면 엔진을 부르지 않고 ValueError다.
        """
        _check_nbest(beam_size, n)
        result = self.model.translate_batch(
            [tokens],
            beam_size=beam_size,
            num_hypotheses=n,
            return_scores=True,
            max_decoding_length=256,
        )[0]
        scored = list(zip(result.hypotheses, result.scores, strict=True))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:n]


class MarianTranslator:
    """opus-mt 모델 하나. 한 방향으로만 번역한다.

    by_sentence를 켜면 입력을 문장으로 나눠 따로 번역하고 공백 하나로 잇는다: opus-mt는 한 문장 단위로
    학습됐다(docs/experiments.md 2-1절). Hugging Face에 올라온 tc-big 변환본은 어휘가 깨져 있어,
    Helsinki의 원본 배포본을 eval/mt_convert.py로 CTranslate2로 바꾼 폴더(model_dir)를 읽는다(README).
    원문·번역문 어휘가 따로라 source.spm과 target.spm을 각각 읽는다.
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
        # translate()의 디코딩 설정. 측정 도구(eval/mt_eval.py)가 보고서에 그대로 남겨, 판정
        # 도구(eval/mbr.py)가 팔이 등록한 설정(빔 크기·문장 단위)으로 돌았는지 확인한다. 서비스는 읽지 않는다.
        self.decoding = {
            "beam_size": beam_size,
            "num_hypotheses": 1,
            "selection": "1-best",
            "by_sentence": by_sentence,
        }
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

    def candidates(
        self, text: str, source: Language, target: Language, beam_size: int, n: int
    ) -> list[list[tuple[str, float]]]:
        """문장마다 빔 탐색의 번역 후보 n개를 (번역문, 엔진 점수)로, 점수가 높은 것부터 돌려준다.

        MBR 디코딩 시험(docs/experiments.md 20절)의 측정 도구(eval/mbr.py)만 쓰고, 서비스(translate)는 쓰지
        않는다. 입력 검사와 문장 나누기는 translate()와 같다: 같은 언어 쌍·빈 글·다른 방향은 ModelError,
        by_sentence면 split_sentences로 나눈 문장마다(순서 그대로) 목록 하나, 아니면 목록 하나. 후보마다
        목표 언어 SentencePiece로 복원하고 앞뒤 공백을 지운다(translate()의 결과와 같은 모양). 빈 글이 된
        후보는 버리고, 한 문장에서 남는 것이 없으면 translate()처럼 빈 번역 ModelError다. 엔진·토크나이저
        오류는 _guarded가 ModelError로 바꾼다. 가설 수가 1~빔 크기가 아니면 설정 실수라 엔진을 부르기 전에
        ValueError로 멈춘다(ModelError로 바꾸면 측정 도구가 빈 번역으로 세며 계속 돈다).
        """
        _check_text(text, source, target)
        if (source, target) != self.direction:
            raise ModelError(f"{self.model_name} only translates {self.direction[0]}->{self.direction[1]}")
        _check_nbest(beam_size, n)
        sentences = split_sentences(text) if self.by_sentence else [text]
        return [self._candidates_one(sentence, beam_size, n) for sentence in sentences]

    # 한 문장의 후보: 토큰화·n-best 생성·후보마다 복원을 _guarded로 감싼다. 겹치는 글도 그대로 둔다(가설
    # 하나가 후보 하나, MBR에서 의사 참조 하나).
    def _candidates_one(self, text: str, beam_size: int, n: int) -> list[tuple[str, float]]:
        nbest = _guarded(
            lambda: self._model.generate_nbest(marian_source_tokens(self._source_sp, text), beam_size, n)
        )
        decoded = [
            (_guarded(lambda pieces=pieces: self._target_sp.decode(pieces)), score) for pieces, score in nbest
        ]
        kept = [(candidate.strip(), float(score)) for candidate, score in decoded if _usable(candidate)]
        if not kept:
            raise ModelError("the model returned an empty translation")
        return kept


class NllbTranslator:
    """NLLB-200은 모델 하나로 두 방향을 모두 번역한다(비교 후보, 서비스에서 쓰지 않음).

    도착 언어 코드를 target_prefix로 주고, 출력의 첫 토큰(그 코드)은 빼고 복원한다.
    """

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
        return _nonempty(_guarded(lambda: self._sp.decode(pieces[1:])))  # 도착 언어 코드는 뺀다


class OllamaTranslator:
    """Ollama가 서비스하는 지시 학습 LLM. 번역문만 답하도록 지시한다.

    비교 후보(qwen2.5 7B 등, 서비스에서 쓰지 않음). temperature 0으로 부르고, 통신·형식 오류는 ModelError.
    """

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


# Hy-MT2 설정(비교용, 서비스 기본은 opus. docs/experiments.md 15절). 프롬프트는 모델 카드의 "Default
# Translation" 영어 프롬프트 그대로이고(도착 언어의 영어 이름을 넣음), 모델에 기본 시스템 프롬프트가 없으므로
# 유일한 메시지로 보낸다. HY_MT_OPTIONS는 eval/hymt_setup.py가 Ollama Modelfile의 PARAMETER로도 써 넣어
# 서비스와 모델 빌드가 같은 값을 쓴다.
HY_MT_PROMPT = (
    "Translate the following text into {target}. Note that you should only output the translated result "
    "without any additional explanation:\n\n{text}"
)
# 탐욕 디코딩(temperature 0. 카드는 0.7로 샘플링하지만 서비스는 같은 입력에 같은 출력이 필요하다), 카드의
# 반복 벌점, opus와 같은 출력 상한 256토큰, 고정 문맥 크기: 가장 긴 요청(500자, docs/api.md)에 프롬프트와
# 출력 상한을 더해도 2048토큰 안에 넉넉히 들어간다.
HY_MT_OPTIONS = {"temperature": 0, "repeat_penalty": 1.05, "num_predict": 256, "num_ctx": 2048}
# prepare의 마감까지 남은 시간이 없을 때 시작한 요청도 유효한 제한을 받게 하는 최소값(초).
HY_MT_MIN_REQUEST_S = 0.01


class HyMtTranslator:
    """Tencent의 Hy-MT2(Apache-2.0). Ollama가 서비스하는 번역 모델로, 시험 설정이다.

    Ollama는 공식 GGUF를 Modelfile로 돌리며, 모델 자체의 채팅 템플릿과 턴 끝 토큰을 stop으로
    쓴다(eval/hymt_setup.py). by_sentence를 켜면 opus 영→한처럼 입력을 문장으로 나눈다. `calls`는 마지막
    번역의 호출마다 Ollama가 알려 준 값을 평가 도구용으로 남긴다. 시험 결과 COMET-22는 opus보다 높았지만
    chrF 차이가 유의하지 않아 채택하지 않았다(README 알려진 한계, docs/experiments.md 15절). model_name에
    "/split"을 붙여 문장 단위 설정을 기록에서 구분한다.
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
        # 출력 상한에서 잘린 답("length")도 그대로 쓴다. opus가 256토큰에서 잘린 번역을 그대로 쓰는 것과 같다.
        return _nonempty(content)

    def prepare(self, timeout_s: float | None = None) -> str:
        """Ollama에 모델의 기대한 빌드가 있는지 확인하고 계속 올라가 있게 적재한 뒤 digest를 돌려준다.

        서버 시작 때 services/models._wait_until_ready가 되풀이해 부른다. 순서: /api/show로 모델 존재 확인
        → /api/tags에서 그 모델의 digest를 찾음 → 기대한 빌드인지 비교 → 프롬프트 없는 /api/generate로
        적재(keep_alive -1). `timeout_s`(서버 시작 대기의 남은 시간)를 주면 마감이 생긴다: 여기의 각
        요청(show, tags, 적재)은 그 요청이 시작될 때 남은 시간까지만 받으므로, 확인이 느리면 적재는 나머지
        시간만 받는다. 주지 않으면 확인은 요청 제한까지, 적재는 `prepare_timeout_s`까지 기다린다.
        Ollama나 모델이 아직 없으면(기다릴 가치가 있음) ModelError라 재시도되고, 다른 빌드는 기다려도 바뀌지
        않으므로 RuntimeError로 바로 시작을 멈춘다.
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
            # 메시지 없는 요청은 모델을 올리기만 한다. keep_alive -1은 Ollama가 멈출 때까지 모델을 올려 둔다.
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
    """방향마다 그 방향의 모델로 보낸다. 가장 좋은 모델이 방향마다 다를 수 있기 때문이다.

    서비스는 한→영·영→한 opus 모델 두 개를 이것으로 묶어 Translator 하나처럼 쓴다. model_name은
    "ko->en: ..., en->ko: ..." 형태로 기록과 /api/health에 나간다.
    """

    def __init__(self, by_direction: dict[tuple[Language, Language], Translator]) -> None:
        self._by_direction = by_direction
        self.model_name = ", ".join(f"{s}->{t}: {m.model_name}" for (s, t), m in by_direction.items())

    def translate(self, text: str, source: Language, target: Language) -> str:
        _check_text(text, source, target)
        model = self._by_direction.get((source, target))
        if model is None:
            raise ModelError(f"no translation model for {source}->{target}")
        return model.translate(text, source, target)


class FormalRegisterTranslator:
    """번역기를 감싸, 한국어 번역 결과의 문장 끝 해라체 평서를 합쇼체로 바꿔 돌려준다(KO_REGISTER=formal).

    영→한은 문장마다 따로 번역해(docs/experiments.md 2-1절) 한 응답 안에서 '…있다. …있습니다.'처럼 말투가 섞일
    수 있어, 그것을 맞추는 시험 설정이다(21절). 후처리 규칙은 app/services/register.py에 있고, 여기서는 받은
    함수(formalize: 글 → 글)를 부르기만 한다. 서비스는 영→한 번역기 하나를 감싸므로(services/models.py)
    동시통역 자막과 최종 번역 모두에 걸리고, 감싼 번역기(MarianTranslator·HyMtTranslator)는 바꾸지 않는다.
    도착 언어가 한국어가 아니면 그대로 돌려준다. 감싼 번역기의 실패는 그대로 올리고, 후처리 안의 실패(분석기
    오류 등)는 _guarded처럼 번역 실패(ModelError, API 503)로 바꾸며, 바꾼 결과가 비면 빈 번역 실패다.
    model_name 끝의 '+formal'이 번역 기록·/api/health에 남아 후처리를 건 결과인지 알 수 있다.
    """

    def __init__(self, inner: Translator, formalize: Callable[[str], str]) -> None:
        self._inner = inner
        self._formalize = formalize
        self.model_name = f"{inner.model_name}+formal"

    def translate(self, text: str, source: Language, target: Language) -> str:
        translation = self._inner.translate(text, source, target)
        if target != "ko":
            return translation
        try:
            formal = self._formalize(translation)
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - 후처리를 감싸는 예외 경계(번역 실패로 바꿈)
            raise ModelError(f"the formal register conversion failed: {type(exc).__name__}") from exc
        return _nonempty(formal)
