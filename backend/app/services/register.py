"""영→한 번역문의 문장 끝 해라체 평서를 합쇼체로 바꾸는 결정적 후처리(docs/experiments.md 21절).

영→한은 문장마다 따로 번역해(docs/experiments.md 2-1절) 문장마다 말투가 따로 정해지므로, 한 응답 안에서
'…이루어져 있다. …이루어져 있습니다.'처럼 해라체와 합쇼체가 섞일 수 있다. 이 모듈은 형태소 분석기
python-mecab-ko(mecab-ko-dic 사전)로 문장 끝 어절의 종결어미를 찾아 해라체 평서(…다·…ㄴ다·…는다)만
합쇼체(…습니다·…ㅂ니다)로 바꾼다. 같은 글이면 늘 같은 결과이고, 합쇼체 글은 그대로다(멱등).
"""

# [파일 역할] 말투 후처리의 규칙(formal_word: 어절 하나, formalize: 글 하나)과 실제 분석기(MecabAnalyzer),
# 서버가 쓰는 함수를 만드는 load_formalizer. 번역기 감싸개(translation.FormalRegisterTranslator)가
# load_formalizer가 준 함수를 영→한 결과에 건다(KO_REGISTER=formal, 기본 asis는 감싸지 않음).
# [설계 이유] 모델이나 LLM으로 다시 쓰지 않고 형태소 규칙으로만 바꾼다: 결과가 결정적이고, 바뀌는 곳이 문장 끝
# 어미뿐이라 뜻을 건드릴 여지가 작으며, CPU에서 짧은 시간에 끝난다. 규칙과 그대로 두는 경우는 측정 전에 21절에
# 적었다. 바꾸는 법을 모르는 경우(분석이 이상함, 활용을 모름)는 틀리게 바꾸느니 그대로 둔다: 틀린 활용(예:
# 길습니다)은 섞인 말투보다 나쁘다. 규칙 설계에 본 데이터는 FLEURS validation 저장 번역문과 직접 지은
# 문장뿐이다.
# [연결] services/models.load_translation이 설정에 따라 load_formalizer를 부르고, eval/register_eval.py가 같은
# 함수를 저장된 번역문에 걸어 측정한다. 테스트: tests/test_register.py.

from __future__ import annotations

import functools
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.services.interfaces import ModelError


@dataclass(frozen=True)
class Morpheme:
    """형태소 분석 결과의 토큰 하나.

    start·end는 분석한 글 안의 글자 위치(end는 포함하지 않음)다. parts는 (형태, 품사) 목록이다: mecab-ko-dic이
    활용형을 한 토큰으로 내는 경우(예: '간다' = 가/VV + ᆫ다/EF, '했' = 하/VV + 았/EP)는 원형 열(expression)을
    나눈 것이고, 아니면 (표면, 품사) 하나다. 품사는 mecab-ko-dic의 세종 품사 태그다.
    """

    surface: str
    start: int
    end: int
    parts: tuple[tuple[str, str], ...]


# 분석기의 모양: 글 → 토큰 목록(글 안의 위치 순서). 테스트는 정해 둔 토큰을 돌려주는 가짜를 넣는다.
Analyzer = Callable[[str], Sequence[Morpheme]]

# 한글 음절 글자(U+AC00~U+D7A3)는 0xAC00 + (초성 × 21 + 중성) × 28 + 종성이다. 종성 번호: 0 없음, 4 ㄴ, 8 ㄹ,
# 17 ㅂ(유니코드 표준의 한글 음절 조합 규칙). 겹받침(ㄶ·ㄺ·ㄼ 등)은 다른 번호라 ㄴ·ㄹ로 보지 않는다.
_SYLLABLE_FIRST, _SYLLABLE_LAST = 0xAC00, 0xD7A3
_NO_FINAL, _FINAL_N, _FINAL_L, _FINAL_B = 0, 4, 8, 17

# mecab-ko-dic은 어미 'ㄴ다'의 ㄴ을 사전 항목에 따라 호환 자모(ㄴ U+3134)나 종성 자모(ᆫ U+11AB)로 쓴다(설계 때
# '간다'는 ᆫ다, '안다'는 ㄴ다로 나옴). 초성 자모(ᄂ U+1102)까지 하나로 맞춰 같은 어미로 본다.
_NIEUN = str.maketrans({"\u11ab": "\u3134", "\u1102": "\u3134"})

# 바꾸는 종결어미의 원형(해라체 평서)과 그 어미에 붙는 품사. mecab-ko-dic이 문장 끝 'ㄴ다'를 연결어미 EC로
# 다는 경우가 있어(설계 때 '연다'·'운다'·'건다') EC도 받는다. 문장 끝 어절인지는 문장부호로 따로 확인한다.
_PLAIN_ENDINGS = {"다", "ㄴ다", "는다"}
_ENDING_TAGS = {"EF", "EC"}
# 용언 어간 품사: 동사·형용사·보조 용언·동사 파생 접미사(하다·되다의 하·되)·형용사 파생 접미사.
_STEMS = {"VV", "VA", "VX", "XSV", "XSA"}
# '다'가 바로 붙을 수 있는 앞 형태소: 용언 어간, 선어말어미(었·겠·시 등), 서술격 조사 '이',
# 부정 지정사 '아니'.
_BEFORE_DA = _STEMS | {"EP", "VCP", "VCN"}
# 앞 음절 받침 ㄴ + '다'를 어미 'ㄴ다'로 읽는 앞 형태소. mecab-ko-dic이 '만든다'를 만든/VV + 다/EF로,
# '머문다'를 머물/VV + ᆫ/ETM + 다/EF로 나눈 것을 설계 때 봤다(받침 ㄴ은 어간이 아니라 어미 'ㄴ다'의 것).
# 받침 ㄴ으로 끝나는 동사 어간의 해라체 현재는 '는다'(신는다·안는다)라 '…ㄴ다'로 끝나는 동사 어절은 거의 늘
# 어미 'ㄴ다'다.
_N_AS_NDA = {"VV", "VX", "XSV", "ETM"}


def _final(syllable: str) -> int | None:
    """한글 음절의 종성 번호(0 = 받침 없음), 한글 음절이 아니면 None."""
    code = ord(syllable)
    if not _SYLLABLE_FIRST <= code <= _SYLLABLE_LAST:
        return None
    return (code - _SYLLABLE_FIRST) % 28


def _with_final(syllable: str, final: int) -> str:
    """한글 음절의 받침을 final(종성 번호)로 바꾼 음절. 예: 가 → 갑, 길 → 깁(ㄹ을 빼고 ㅂ)."""
    code = ord(syllable) - _SYLLABLE_FIRST
    return chr(_SYLLABLE_FIRST + code - code % 28 + final)


def _b_nida(word: str) -> str:
    """'…X다'의 X에 받침 ㅂ을 붙이고(원래 받침은 버림) '다'를 '니다'로: 가다 → 갑니다, 만든다 → 만듭니다."""
    return word[:-2] + _with_final(word[-2], _FINAL_B) + "니다"


def _previous(morphs: Sequence[Morpheme]) -> tuple[str, str] | None:
    """어미 바로 앞 형태소의 (형태, 품사).

    어미와 한 토큰이면 그 토큰의 앞 부분, 아니면 바로 앞에 붙은 토큰의 끝 부분이다. 둘 다 없으면 None.
    """
    last = morphs[-1]
    if len(last.parts) >= 2:
        return last.parts[-2]
    if len(morphs) >= 2 and morphs[-2].end == last.start:
        return morphs[-2].parts[-1]
    return None


def formal_word(word: str, morphs: Sequence[Morpheme]) -> str | None:
    """문장 끝 어절 하나(끝 문장부호를 뗀 것)를 합쇼체로 바꾼 글. 바꾸지 않으면 None.

    morphs는 그 어절의 토큰이고 위치는 어절 기준이다. 마지막 토큰이 어절 끝에 닿고 그 끝 형태소가 해라체 평서
    어미(다·ㄴ다·는다)일 때만 바꾼다(21절 규칙):
    - '는다' → '습니다' (먹는다 → 먹습니다, 듣는다 → 듣습니다)
    - 'ㄴ다' → 앞 음절의 받침 ㄴ을 ㅂ으로 + '니다' (간다 → 갑니다, 만든다 → 만듭니다, 모른다 → 모릅니다)
    - '다': 받침 없는 명사 뒤 서술격 조사 생략형(학교다)은 '입니다'(의존 명사 '거' 뒤는 '겁니다'), 그 밖에는
      앞 음절로 정한다 — 받침 없음 → ㅂ + '니다'(크다 → 큽니다, 이다 → 입니다), ㄹ 받침인 용언 어간 → ㄹ을
      빼고 ㅂ + '니다'(길다 → 깁니다), ㄴ 받침인 동사류(분석기가 'ㄴ다'를 나눈 것) → 'ㄴ다'와 같이, 그 밖 받침
      → '습니다'(했다 → 했습니다, 춥다 → 춥습니다).
    르·ㅂ·ㄷ 불규칙은 -습니다/-ㅂ니다 앞에서 어간이 바뀌지 않아 같은 규칙으로 된다. '니다'로 끝나면 이미
    합쇼체이거나 분석이 틀린 것으로 보고 바꾸지 않는다('아니다'만 예외).
    """
    if len(word) < 2 or not word.endswith("다") or not morphs or morphs[-1].end != len(word):
        return None
    last = morphs[-1]
    form, tag = last.parts[-1]
    ending = form.translate(_NIEUN)
    if tag not in _ENDING_TAGS or ending not in _PLAIN_ENDINGS:
        return None
    before = _previous(morphs)
    if ending == "는다":
        return word[:-2] + "습니다" if word.endswith("는다") else None
    if ending == "ㄴ다":
        return _b_nida(word) if _final(word[-2]) == _FINAL_N else None
    # 어미 '다'
    if word.endswith("니다") and before != ("아니", "VCN"):
        return None
    if last.surface == "다" and last.parts == (("이", "VCP"), (form, tag)):
        # 서술격 조사 '이'가 글에 없는 생략형(학교다 = 학교/NNG + 다(이/VCP+다/EF)). 생략은 받침 없는 말
        # 뒤에서만 되므로, 앞 글자가 받침 있는 음절이면 분석 오류로 보고 그대로 둔다. 로마자·숫자 뒤(NASA다)는
        # '입니다'.
        noun = word[:-1]
        if _final(noun[-1]) not in (None, _NO_FINAL):
            return None
        if noun.endswith("거") and len(morphs) >= 2 and morphs[-2].parts[-1] == ("거", "NNB"):
            return noun[:-1] + "겁니다"  # 갈 거다 → 갈 겁니다('거입니다'는 쓰지 않는 꼴)
        return noun + "입니다"
    final = _final(word[-2])
    if final is None or before is None or before[1] not in _BEFORE_DA | _N_AS_NDA:
        return None
    if final == _NO_FINAL:
        return _b_nida(word)
    if final == _FINAL_L:
        return _b_nida(word) if before[1] in _STEMS else None
    if final == _FINAL_N:
        return _b_nida(word) if before[1] in _N_AS_NDA else None
    if before[1] not in _BEFORE_DA:
        return None
    return word[:-1] + "습니다"


# 어절 끝에서 떼어 보는 문장부호: 문장을 끝내는 것(. ! ? … 와 전각 형태)과 닫는 따옴표·괄호.
_SENTENCE_MARKS = set(".!?…。！？")
_QUESTION_MARKS = set("?？")
_CLOSERS = set("\"'”’)]}」』〉》】")
_TRAILING = "".join(_SENTENCE_MARKS | _CLOSERS)
# 짝이 있는 괄호·따옴표(여는 것 → 닫는 것). 곧은 따옴표 " ' 는 여닫는 글자가 같아 번갈아 열고 닫는다.
_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "「": "」",
    "『": "』",
    "〈": "〉",
    "《": "》",
    "【": "】",
    "“": "”",
    "‘": "’",
}
_STRAIGHT = {'"', "'"}
_WORD = re.compile(r"\S+")


def _latin(char: str) -> bool:
    return char.isascii() and char.isalpha()


def _quoted(text: str) -> list[bool]:
    """글자마다 따옴표·괄호 안인지(True).

    닫히지 않은 여는 것 뒤는 글 끝까지 안으로 본다(바꾸지 않는 쪽이 안전하다).
    로마자 두 글자 사이의 ' ’는 영어 단어 안의 생략 부호(McDonald's)로 보고 따옴표로 세지 않는다. 한글 사이의
    '는 따옴표다('번호 하나'이며). 짝이 맞지 않는 닫는 것은 무시한다.
    """
    inside, stack = [], []
    for index, char in enumerate(text):
        apostrophe = (
            char in ("'", "’")
            and 0 < index < len(text) - 1
            and _latin(text[index - 1])
            and _latin(text[index + 1])
        )
        if apostrophe:
            pass
        elif char in _STRAIGHT and stack and stack[-1] == char:
            stack.pop()
        elif char in _STRAIGHT:
            stack.append(char)
        elif char in _PAIRS:
            stack.append(_PAIRS[char])
        elif stack and char == stack[-1]:
            stack.pop()
        inside.append(bool(stack))
    return inside


def _sentence_ends(text: str) -> list[tuple[int, int, str]]:
    """문장 끝 어절마다 (어절 시작, 끝 문장부호를 뗀 끝, 뗀 문장부호).

    어절은 공백(줄바꿈 포함)으로 나눈 덩어리다.
    문장 끝 어절: 끝에서 뗀 문장부호에 . ! ? … 중 하나가 있거나, 글의 마지막 어절. 쉼표·쌍점이 붙은 어절은
    문장 가운데라 들지 않는다(…좋다, 그래서).
    """
    words = list(_WORD.finditer(text))
    ends = []
    for index, match in enumerate(words):
        word = match.group()
        core = word.rstrip(_TRAILING)
        marks = word[len(core) :]
        if index == len(words) - 1 or any(char in _SENTENCE_MARKS for char in marks):
            ends.append((match.start(), match.start() + len(core), marks))
    return ends


def formalize(text: str, analyze: Analyzer) -> str:
    """글의 문장 끝 해라체 평서를 합쇼체로 바꾼 글(21절 규칙). 바꿀 것이 없으면 받은 글 그대로다.

    문장 끝 어절 중 그대로 두는 것: 끝 문장부호에 ?가 있음(의문), 끝에 닫는 따옴표·괄호가 붙음, 끝 글자가
    따옴표·괄호 안, '다'로 끝나지 않음(해요체·의문·명령·청유·명사), '니다'로 끝남('아니다'만 빼고). 남은
    어절이 있을 때만 글 전체를 한 번 분석하고(문맥째 분석해 분석기 결과가 서비스와 측정에서 같게), 어절마다
    formal_word로 바꾼다. 공백·줄바꿈·문장부호는 그대로 남는다.
    """
    quoted = _quoted(text)
    candidates = []
    for start, end, marks in _sentence_ends(text):
        word = text[start:end]
        if (
            not word.endswith("다")
            or any(char in _QUESTION_MARKS or char in _CLOSERS for char in marks)
            or quoted[end - 1]
            or (word.endswith("니다") and not word.endswith("아니다"))
        ):
            continue
        candidates.append((start, end))
    if not candidates:
        return text
    morphemes = analyze(text)
    pieces, done = [], 0
    for start, end in candidates:
        inside = [
            Morpheme(m.surface, m.start - start, m.end - start, m.parts)
            for m in morphemes
            if start <= m.start and m.end <= end
        ]
        formal = formal_word(text[start:end], inside)
        if formal is not None:
            pieces += [text[done:start], formal]
            done = end
    pieces.append(text[done:])
    return "".join(pieces)


def from_mecab(morpheme) -> Morpheme:
    """python-mecab-ko의 Morpheme(span·surface·feature)을 이 모듈의 Morpheme으로.

    원형 열(feature.expression, 예: '가/VV/*+ᆫ다/EF/*')이 있으면 '+'로 나눈 형태소마다 (형태, 품사)를 parts로
    둔다. 원형 열이 없거나 모양이 다르면 (표면, 품사) 하나다.
    """
    feature = morpheme.feature
    parts = ((morpheme.surface, feature.pos),)
    if feature.expression:
        split = [piece.split("/") for piece in feature.expression.split("+")]
        if all(len(piece) >= 2 and piece[0] and piece[1] for piece in split):
            parts = tuple((piece[0], piece[1]) for piece in split)
    return Morpheme(morpheme.surface, morpheme.span.start, morpheme.span.end, parts)


class MecabAnalyzer:
    """python-mecab-ko(mecab-ko-dic 사전)로 글을 토큰으로 나누는 분석기.

    서버 이미지의 tts 의존성 그룹에 이미 있는 패키지다(MeloTTS의 한국어 전처리 g2pkk가 쓰는 것과 같음). 번역은
    모델 스레드(MODEL_THREADS가 2 이상이면 여럿)와 동시통역 자막 갱신에서 불릴 수 있는데, 같은 Tagger를 여러
    스레드가 함께 불러도 되는지 python-mecab-ko 문서로 확인하지 못해 분석 한 번을 잠금으로 감싼다.
    """

    def __init__(self) -> None:
        # mecab: python-mecab-ko(tts 그룹). 없으면 ImportError가 그대로 올라가 load_formalizer가 분명한 오류로
        # 바꾼다.
        import mecab

        self._tagger = mecab.MeCab()
        self._lock = threading.Lock()

    def __call__(self, text: str) -> list[Morpheme]:
        with self._lock:
            parsed = self._tagger.parse(text)
        return [from_mecab(morpheme) for morpheme in parsed]


def load_formalizer() -> Callable[[str], str]:
    """서버와 측정 도구가 쓰는 후처리 함수(글 → 글): 실제 분석기를 올려 formalize에 묶는다.

    python-mecab-ko가 없으면 ModelError로 멈춘다: KO_REGISTER=formal인데 후처리 없이 서버가 몰래 뜨지 않게,
    서버 시작(services/models.load_translation)이 이 오류로 멈춘다.
    """
    try:
        analyzer = MecabAnalyzer()
    except ImportError as exc:
        raise ModelError(
            "KO_REGISTER=formal needs the Korean morphological analyzer python-mecab-ko "
            "(the tts dependency group of the server image)"
        ) from exc
    return functools.partial(formalize, analyze=analyzer)
