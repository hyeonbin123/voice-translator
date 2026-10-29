"""영→한 말투 후처리(app/services/register.py)와 그 감싸개(FormalRegisterTranslator)를 검증한다.

후처리 규칙은 docs/experiments.md 21절에 측정 전에 적은 것이다. 형태소 분석기(python-mecab-ko)가 없어도 도는
가짜 형태소 열 테스트와, 분석기가 있을 때만 도는 실제 문장 테스트(맨 아래)로 나뉜다.
"""

# [파일 역할] 문장 끝 해라체 평서(…다·…ㄴ다·…는다)를 합쇼체(…습니다·…ㅂ니다)로 바꾸는 규칙, 그 규칙을 문장 끝
# 어절에만 거는 글 단위 처리, 영→한 번역기를 감싸는 FormalRegisterTranslator, 분석기를 올리는
# load_formalizer를 확인한다.
# [무엇을 확인하나] (1) 활용 규칙: 어간 받침(없음·ㄹ·ㄴ·그 밖)과 선어말어미·서술격 조사(생략 포함)에 따른
# 어미, 르·ㅂ·ㄷ 불규칙, mecab-ko-dic이 활용형을 한 토큰으로 내거나 'ㄴ다'를 이상하게 나누는 경우 (2) 그대로
# 두는 것: 이미 높임, 의문·명령·청유, 따옴표·괄호 안, 문장 가운데, 분석이 어절 끝과 맞지 않음 (3) 멱등(합쇼체
# 입력은 그대로) (4) 감싸개는 한국어 결과에만 걸고 실패를 ModelError로 바꿈 (5) 분석기가 없으면 분명한 오류.
# [왜 중요한가] 측정(21절)과 서비스(KO_REGISTER=formal)가 같은 규칙을 써야 측정 결과가 서비스에 들어맞는다.
# 틀린 활용(예: 길습니다, 만든습니다)은 번역 품질을 직접 깎으므로, 바꾸지 못하는 경우는 그대로 두는 쪽이 맞다.

import sys
from types import SimpleNamespace

import pytest

from app.services import register
from app.services.interfaces import ModelError
from app.services.register import Morpheme, formal_word, formalize
from app.services.translation import FormalRegisterTranslator
from tests.fakes import FakeTranslator


def morphs(spec: str) -> tuple[str, list[Morpheme]]:
    """가짜 형태소 열 표기에서 글과 형태소 목록(글 안의 글자 위치 포함)을 만든다.

    토큰은 공백으로 나누고, 토큰 '|'는 글 안의 공백 한 칸이다. 토큰은 '표면/품사'(형태소 하나) 또는
    '표면=형태/품사+형태/품사'(mecab-ko-dic이 활용형을 한 토큰으로 내고 원형 열(expression)을 따로 주는
    경우)다. 예: '그/NP 는/JX | 갔=가/VV+았/EP 다/EF ./SF' → 글 '그는 갔다.'
    """
    text, tokens = "", []
    for token in spec.split():
        if token == "|":
            text += " "
            continue
        if "=" in token:
            surface, expression = token.split("=", 1)
            parts = tuple(tuple(part.rsplit("/", 1)) for part in expression.split("+"))
        else:
            surface, tag = token.rsplit("/", 1)
            parts = ((surface, tag),)
        tokens.append(Morpheme(surface, len(text), len(text) + len(surface), parts))
        text += surface
    return text, tokens


def analyzer(spec: str):
    """표기로 만든 글에만 그 형태소를 돌려주는 가짜 분석기와, 받은 글을 쌓는 호출 기록."""
    text, tokens = morphs(spec)
    calls: list[str] = []

    def analyze(given: str) -> list[Morpheme]:
        calls.append(given)
        assert given == text, "the analyzer must get the whole text"
        return tokens

    return text, analyze, calls


def no_analyzer(given: str) -> list[Morpheme]:
    """불리면 안 되는 분석기: 바꿀 후보가 없는 글은 분석하지 않고 그대로 돌려줘야 한다."""
    raise AssertionError(f"the analyzer must not be called for {given!r}")


# ---- 어절 하나의 활용 규칙 (계획서의 단위 테스트 목록 + 21절 규칙의 갈래마다)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        # '는다'(받침 있는 동사 어간 뒤) → '습니다'
        ("먹/VV 는다/EF", "먹습니다"),
        # 'ㄴ다' → 앞 음절의 받침 ㄴ을 ㅂ으로 + '니다'. mecab-ko-dic은 'ㄴ'을 호환 자모(ㄴ)나 종성 자모(ᆫ)로
        # 쓴다
        ("간다=가/VV+ㄴ다/EF", "갑니다"),
        ("간다=가/VV+ᆫ다/EF", "갑니다"),
        # ㄹ 받침 어간: 'ㄴ다' 앞에서 이미 ㄹ이 빠진 표면(만든)이라 같은 규칙으로 된다
        ("만든다=만들/VV+ㄴ다/EF", "만듭니다"),
        ("안다=알/VV+ㄴ다/EF", "압니다"),
        # 받침 없는 형용사 어간 + '다' → ㅂ + '니다'
        ("크/VA 다/EF", "큽니다"),
        # 그 밖 받침 + '다' → '습니다'
        ("있/VA 다/EF", "있습니다"),
        # 활용형 한 토큰(했 = 하/VV + 였/EP) 뒤 '다': 선어말어미 뒤 → '습니다'
        ("했=하/VV+였/EP 다/EF", "했습니다"),
        ("하/VV 겠/EP 다/EF", "하겠습니다"),
        # 서술격 조사 '이다' → '입니다', 받침 없는 명사 뒤 생략형('학교다' = 이/VCP+다/EF) → '입니다'
        ("학생/NNG 이/VCP 다/EF", "학생입니다"),
        ("학교/NNG 다=이/VCP+다/EF", "학교입니다"),
        ("NASA/SL 다=이/VCP+다/EF", "NASA입니다"),
        # 의존 명사 '거' 뒤 생략형은 '거입니다'가 아니라 '겁니다'
        ("거/NNB 다=이/VCP+다/EF", "겁니다"),
        # 불규칙: -습니다/-ㅂ니다 앞에서는 어간이 바뀌지 않는다(르 불규칙 모르다, ㅂ 불규칙 춥다, ㄷ 불규칙
        # 듣다)
        ("모른다=모르/VV+ㄴ다/EF", "모릅니다"),
        ("춥/VA 다/EF", "춥습니다"),
        ("듣/VV 는다/EF", "듣습니다"),
        # ㄹ 받침 용언 어간 + '다' → ㄹ을 빼고 ㅂ + '니다'
        ("길/VA 다/EF", "깁니다"),
        ("멀/VA 다/EF", "멉니다"),
        # 부정 지정사 '아니다'는 '니다'로 끝나도 해라체다
        ("아니/VCN 다/EF", "아닙니다"),
        # 높임 선어말어미 '시': 'ㄴ다'와 붙으면 '신다' → '십니다', '었'과 붙은 '셨' 뒤는 '습니다'
        ("신다=시/EP+ᆫ다/EF", "십니다"),
        ("셨=시/EP+었/EP 다/EF", "셨습니다"),
        # 파생 접미사 뒤: 하/XSV + 다(받침 없음), 되/XSV + ㄴ다
        ("하/XSV 다/EF", "합니다"),
        ("된다=되/XSV+ᆫ다/EF", "됩니다"),
        # 겹받침(ㄶ·ㄼ)은 ㄹ로 보지 않는다
        ("않/VX 다/EF", "않습니다"),
        ("넓/VA 다/EF", "넓습니다"),
        # mecab-ko-dic이 문장 끝 'ㄴ다'를 연결어미(EC)로 다는 경우(연다·운다·건다)도 같은 어미다
        ("연다=열/VV+ᆫ다/EC", "엽니다"),
        # mecab-ko-dic이 'ㄴ다'를 '만든/VV + 다/EF'나 '머문/VV+ETM + 다/EF'처럼 나누는 경우: 받침 ㄴ을
        # 'ㄴ다'로 읽는다
        ("만든/VV 다/EF", "만듭니다"),
        ("머문=머물/VV+ᆫ/ETM 다/EF", "머뭅니다"),
    ],
)
def test_a_plain_sentence_end_becomes_formal(spec, expected):
    word, tokens = morphs(spec)
    assert formal_word(word, tokens) == expected


@pytest.mark.parametrize(
    "spec",
    [
        # 이미 합쇼체·해요체
        "먹/VV 습니다/EF",
        "합니다=하/XSV+ᄇ니다/EF",
        "고마워요=고맙/VA+어요/EC",
        # mecab-ko-dic이 '알립니다'를 '알/VV+ETM + 립니/NNP + 다'로 잘못 나눈 경우: '니다'로 끝나면 건드리지
        # 않는다
        "알=알/VV+ᆯ/ETM 립니/NNP 다=이/VCP+다/EF",
        # 받침 없는 어간의 사전형 '다니다'처럼 '니다'로 끝나는 해라체도 이미 높임과 글자로 가를 수 없어 그대로
        # 둔다
        "다니/VV 다/EF",
        # 의문·명령·청유와 '다'로 끝나지 않는 어미
        "그럴까=그러/VV+ㄹ까/EF",
        "가/VV 자/EF",
        "먹/VV 어라/EF",
        "좋/VA 다고/EC",
        # 서술격 조사 생략은 받침 없는 명사 뒤에서만 된다: 받침 있는 명사 뒤 '다'는 분석 오류로 보고 그대로
        "산/NNG 다=이/VCP+다/EF",
        # 앞 형태소가 용언·선어말어미·서술격 조사가 아니면(숫자 뒤 '다' 등) 활용을 모르므로 그대로
        "10/SN 다/EF",
        # 앞 음절이 한글이 아니면(서술격 조사 생략형이 아닌데 숫자·로마자 바로 뒤 '다') 활용을 모르므로 그대로
        "A/SL 다/EF",
    ],
)
def test_other_endings_are_left_alone(spec):
    word, tokens = morphs(spec)
    assert formal_word(word, tokens) is None


def test_an_analysis_that_does_not_reach_the_word_end_is_left_alone():
    word, tokens = morphs("갔=가/VV+았/EP 다/EF")
    assert formal_word(word, tokens[:1]) is None


# ---- 글 단위: 문장 끝 어절만, 따옴표·괄호·의문은 그대로


def test_every_sentence_end_of_a_mixed_output_becomes_formal():
    # 한 출력 안에서 해라체와 합쇼체가 섞인 경우(문장 단위 번역의 부작용, 21절 배경)
    text, analyze, calls = analyzer(
        "섬/NNG 은/JX | 이루어져=이루어지/VV+어/EC | 있/VX 다/EF ./SF | "
        "북쪽/NNG 은/JX | 숲/NNG 이/VCP 다/EF ./SF | 그들/NP 은/JX | 갔=가/VV+았/EP 습니다/EF ./SF"
    )
    assert formalize(text, analyze) == "섬은 이루어져 있습니다. 북쪽은 숲입니다. 그들은 갔습니다."
    assert calls == [text]  # 글 전체를 한 번만 분석한다


def test_a_plain_ending_in_the_middle_of_a_sentence_is_left_alone():
    text, analyze, _ = analyzer("날씨/NNG 가/JKS | 좋/VA 다/EF ,/SC | 그래서/MAJ | 갔=가/VV+았/EP 다/EF ./SF")
    assert formalize(text, analyze) == "날씨가 좋다, 그래서 갔습니다."


def test_the_last_word_without_punctuation_is_a_sentence_end():
    text, analyze, _ = analyzer("그/NP 는/JX | 갔=가/VV+았/EP 다/EF")
    assert formalize(text, analyze) == "그는 갔습니다"


@pytest.mark.parametrize(("mark", "tag"), [("!", "SF"), ("…", "SE")])
def test_an_exclamation_or_ellipsis_still_ends_the_sentence(mark, tag):
    text, analyze, _ = analyzer(f"그/NP 는/JX | 놀랐=놀라/VV+았/EP 다/EF {mark}/{tag}")
    assert formalize(text, analyze) == f"그는 놀랐습니다{mark}"


def test_a_question_is_left_alone():
    text, analyze, _ = analyzer("정말/MAG | 그랬=그러/VV+었/EP 다/EF ?/SF")
    assert formalize(text, analyze) == text


def test_quoted_and_bracketed_sentences_are_left_alone():
    # 따옴표 안의 문장 끝(간다.)과 괄호로 끝나는 문장은 그대로, 따옴표 밖의 문장 끝만 바꾼다
    text, analyze, _ = analyzer(
        '"/SY 나/NP 는/JX | 간다=가/VV+ᆫ다/EF ./SF | 너/NP 도/JX | 와라=오/VV+아라/EF ./SF "/SY | '
        "그/NP 는/JX | 말했=말/NNG+하/XSV+았/EP 다/EF ./SF | "
        "(/SSO 그/NP 도/JX | 갔=가/VV+았/EP 다/EF ./SF )/SSC"
    )
    assert formalize(text, analyze) == '"나는 간다. 너도 와라." 그는 말했습니다. (그도 갔다.)'


def test_an_unclosed_quote_keeps_the_rest_unchanged():
    # 닫히지 않은 따옴표 뒤는 모두 따옴표 안으로 본다(바꾸지 않는 쪽이 안전하다)
    text, analyze, _ = analyzer(
        "그/NP 는/JX | 갔=가/VV+았/EP 다/EF ./SF | “/SY 나/NP 는/JX | 간다=가/VV+ᆫ다/EF ./SF"
    )
    assert formalize(text, analyze) == "그는 갔습니다. “나는 간다."


def test_single_quotes_close_between_korean_letters_but_not_inside_latin_words():
    # '번호 하나'이며 의 닫는 따옴표는 한글 사이에 있어도 따옴표다. McDonald's의 '는 영어 단어 안의 생략
    # 부호다
    text, analyze, _ = analyzer(
        "'/SY 번호/NNG | 하나/NR '/SY 이/VCP 다/EF ./SF | "
        "McDonald/SL '/SY s/SL 에/JKB | 갔=가/VV+았/EP 다/EF ./SF"
    )
    assert formalize(text, analyze) == "'번호 하나'입니다. McDonald's에 갔습니다."


def test_output_without_a_plain_sentence_end_is_not_analyzed():
    # 이미 합쇼체이거나 '다'로 끝나는 문장 끝이 없으면 분석기를 부르지 않는다(서비스의 흔한 경우를 빠르게)
    text = "그들은 갔습니다. 그것은 무엇입니까? 하지 마세요. 조지 W."
    assert formalize(text, no_analyzer) == text


def test_formal_output_is_left_as_it_is():
    # 멱등: 바꾼 결과를 다시 넣어도 그대로이고, 분석할 것도 없다
    text, analyze, _ = analyzer("그/NP 는/JX | 간다=가/VV+ᆫ다/EF ./SF | 크/VA 다/EF ./SF")
    once = formalize(text, analyze)
    assert once == "그는 갑니다. 큽니다."
    assert formalize(once, no_analyzer) == once


def test_newlines_and_runs_of_spaces_are_kept():
    # 줄바꿈·여러 공백도 어절 사이로 보고, 바꾼 글에 그대로 남는다(분석기 위치는 원래 글 기준)
    text = "크다.\n  크다."
    tokens = [
        Morpheme("크", 0, 1, (("크", "VA"),)),
        Morpheme("다", 1, 2, (("다", "EF"),)),
        Morpheme(".", 2, 3, ((".", "SF"),)),
        Morpheme("크", 6, 7, (("크", "VA"),)),
        Morpheme("다", 7, 8, (("다", "EF"),)),
        Morpheme(".", 8, 9, ((".", "SF"),)),
    ]
    assert formalize(text, lambda given: tokens) == "큽니다.\n  큽니다."


def test_a_token_that_runs_past_the_word_end_is_left_alone():
    # 분석기가 '다.'를 한 토큰으로 내면 어절 끝(다)과 토큰 끝이 맞지 않아 바꾸지 않는다
    tokens = [Morpheme("크", 0, 1, (("크", "VA"),)), Morpheme("다.", 1, 3, (("다.", "EF"),))]
    assert formalize("크다.", lambda given: tokens) == "크다."


# ---- 분석기 결과 바꾸기와 분석기 올리기


def test_mecab_morphemes_become_parts_from_the_expression_column():
    # python-mecab-ko의 Morpheme 모양(span·surface·feature.pos·feature.expression)을 흉내 낸다
    fused = SimpleNamespace(
        span=SimpleNamespace(start=7, end=9),
        surface="간다",
        feature=SimpleNamespace(pos="VV+EF", expression="가/VV/*+ᆫ다/EF/*"),
    )
    single = SimpleNamespace(
        span=SimpleNamespace(start=9, end=10), surface=".", feature=SimpleNamespace(pos="SF", expression=None)
    )
    assert register.from_mecab(fused) == Morpheme("간다", 7, 9, (("가", "VV"), ("ᆫ다", "EF")))
    assert register.from_mecab(single) == Morpheme(".", 9, 10, ((".", "SF"),))


def test_the_formal_register_needs_the_morphological_analyzer(monkeypatch):
    # python-mecab-ko가 없으면(import 실패) 서버가 시작할 때 분명한 오류로 멈춰야 한다
    monkeypatch.setitem(sys.modules, "mecab", None)
    with pytest.raises(ModelError, match="python-mecab-ko"):
        register.load_formalizer()


# ---- 영→한 번역기 감싸개


def test_the_wrapper_makes_korean_translations_formal():
    seen = []

    def fake_formalize(text):
        seen.append(text)
        return text + " (formal)"

    translator = FormalRegisterTranslator(FakeTranslator(), fake_formalize)
    assert translator.translate("hello", "en", "ko") == "[en->ko] hello (formal)"
    assert seen == ["[en->ko] hello"]
    assert translator.model_name == "fake-mt+formal"


def test_the_wrapper_leaves_other_targets_alone():
    translator = FormalRegisterTranslator(FakeTranslator(), lambda text: pytest.fail("not for English"))
    assert translator.translate("안녕", "ko", "en") == "[ko->en] 안녕"


def test_a_failing_conversion_is_a_translation_failure():
    # 후처리 안의 어떤 실패든 번역 실패(ModelError, API 503)로 바꾼다: 번역 모듈의 오류 경계와 같다
    def broken(text):
        raise RuntimeError("analyzer crashed")

    with pytest.raises(ModelError, match="RuntimeError") as caught:
        FormalRegisterTranslator(FakeTranslator(), broken).translate("hello", "en", "ko")
    assert isinstance(caught.value.__cause__, RuntimeError)
    with pytest.raises(ModelError, match="empty"):
        FormalRegisterTranslator(FakeTranslator(), lambda text: "  ").translate("hello", "en", "ko")


def test_the_inner_translation_failure_passes_through():
    error = ModelError("out of memory")
    translator = FormalRegisterTranslator(FakeTranslator(error), lambda text: pytest.fail("not called"))
    with pytest.raises(ModelError) as caught:
        translator.translate("hello", "en", "ko")
    assert caught.value is error


# ---- 실제 형태소 분석기(python-mecab-ko + mecab-ko-dic)로 실제 문장
# 분석기는 tts 그룹(서버 이미지)에만 있어 기본 pytest에서는 건너뛴다. CI는 uv.lock과 같은 판을 얹은 별도
# 단계에서 돌린다. 문장은 직접 지었다(FLEURS 문장을 옮기지 않음). 섞인 출력 세 개는 21절 배경에 적은 섞임의
# 모양(…있다. …있습니다. / …보내졌다. …사용합니다. / …정착했다. …불렀습니다.)을 따랐다.


@pytest.fixture(scope="module")
def real_formalize():
    pytest.importorskip("mecab")
    return register.load_formalizer()


REAL_SENTENCES = [
    ("나는 학교에 간다.", "나는 학교에 갑니다."),
    ("그는 밥을 먹는다.", "그는 밥을 먹습니다."),
    ("그녀는 집을 만든다.", "그녀는 집을 만듭니다."),
    ("산이 크다.", "산이 큽니다."),
    ("책이 있다.", "책이 있습니다."),
    ("그는 일을 했다.", "그는 일을 했습니다."),
    ("그는 학생이다.", "그는 학생입니다."),
    ("여기는 학교다.", "여기는 학교입니다."),
    ("내일 비가 오겠다.", "내일 비가 오겠습니다."),
    ("나는 모른다.", "나는 모릅니다."),
    ("날씨가 춥다.", "날씨가 춥습니다."),
    ("소리를 듣는다.", "소리를 듣습니다."),
    ("길이 길다.", "길이 깁니다."),
    ("그 길은 멀다.", "그 길은 멉니다."),
    ("그것은 사실이 아니다.", "그것은 사실이 아닙니다."),
    ("그것은 중요하다.", "그것은 중요합니다."),
    ("물건이 사용된다.", "물건이 사용됩니다."),
    ("로켓이 우주로 보내졌다.", "로켓이 우주로 보내졌습니다."),
    ("그는 갈 것이다.", "그는 갈 것입니다."),
    ("그는 갈 거다.", "그는 갈 겁니다."),
    ("선생님이 오신다.", "선생님이 오십니다."),
    ("선생님이 오셨다.", "선생님이 오셨습니다."),
    ('그는 "나는 간다."라고 말했다.', '그는 "나는 간다."라고 말했습니다.'),
    ("그는 학교에 가야 한다.", "그는 학교에 가야 합니다."),
    ("그것은 좋지 않다.", "그것은 좋지 않습니다."),
    ("그는 먹지 않는다.", "그는 먹지 않습니다."),
    ("그는 왔었다.", "그는 왔었습니다."),
    ("그는 그것을 안다.", "그는 그것을 압니다."),
    ("그는 놀랐다!", "그는 놀랐습니다!"),
    ("가장 높은 산은 에베레스트다.", "가장 높은 산은 에베레스트입니다."),
    ("그 회사는 IBM이다.", "그 회사는 IBM입니다."),
    ("그는 NASA에서 일한다.", "그는 NASA에서 일합니다."),
    ("그들은 행복했다.", "그들은 행복했습니다."),
    ("그것은 학교였다.", "그것은 학교였습니다."),
    ("그것은 사실이었다.", "그것은 사실이었습니다."),
    ("그는 좋은 사람이었다고 한다.", "그는 좋은 사람이었다고 합니다."),
    ("그는 문을 연다.", "그는 문을 엽니다."),
    ("아기가 운다.", "아기가 웁니다."),
    ("그녀는 집에 머문다.", "그녀는 집에 머뭅니다."),
    ("물이 흐른다.", "물이 흐릅니다."),
    ("그것은 빠르다.", "그것은 빠릅니다."),
    ("그는 매일 걷는다.", "그는 매일 걷습니다."),
    ("그들은 서울에 산다.", "그들은 서울에 삽니다."),
    ("이것은 2010년의 일이다.", "이것은 2010년의 일입니다."),
    ("그 숫자는 10이다.", "그 숫자는 10입니다."),
    ("그는 집에 갔다", "그는 집에 갔습니다"),
    ("날씨가 좋다, 그래서 그는 밖에 나갔다.", "날씨가 좋다, 그래서 그는 밖에 나갔습니다."),
    ("(그는 그곳에 갔다.) 그녀도 갔다.", "(그는 그곳에 갔다.) 그녀도 갔습니다."),
    (
        "그 섬은 산호초로 이루어져 있다. 섬의 북쪽은 숲으로 이루어져 있습니다.",
        "그 섬은 산호초로 이루어져 있습니다. 섬의 북쪽은 숲으로 이루어져 있습니다.",
    ),
    (
        "그 망원경은 1990년에 우주로 보내졌다. 과학자들은 지금도 이 망원경을 사용합니다.",
        "그 망원경은 1990년에 우주로 보내졌습니다. 과학자들은 지금도 이 망원경을 사용합니다.",
    ),
    (
        "그들은 그 섬에 처음 정착했다. 그들은 그 섬을 새 고향이라고 불렀습니다.",
        "그들은 그 섬에 처음 정착했습니다. 그들은 그 섬을 새 고향이라고 불렀습니다.",
    ),
]

# 바꾸지 않아야 하는 실제 문장: 의문·명령·청유, 이미 높임, 괄호로 끝남, mecab-ko-dic이 '알립니다'를 잘못 나눔
REAL_UNCHANGED = [
    "정말 그랬다?",
    "왜 그럴까?",
    "가자.",
    "먹어라.",
    "홀로코스트에 대한 농담을 하지 마라.",
    "그는 이미 떠났습니다.",
    "고마워요.",
    "그 도시는 1990년에 세워졌다 (공식 기록).",
    "회원임을 알립니다.",
]


@pytest.mark.parametrize(("text", "expected"), REAL_SENTENCES)
def test_real_sentences_become_formal(real_formalize, text, expected):
    assert real_formalize(text) == expected
    assert real_formalize(expected) == expected  # 멱등


@pytest.mark.parametrize("text", REAL_UNCHANGED)
def test_real_sentences_that_stay(real_formalize, text):
    assert real_formalize(text) == text


def test_the_real_analyzer_reads_fused_tokens(real_formalize):
    # 활용형 한 토큰(간다 = 가/VV + ᆫ다/EF)의 원형 열과 글자 위치를 읽는다(real_formalize는 분석기가 없으면
    # 건너뛰게 함)
    tokens = register.MecabAnalyzer()("나는 학교에 간다.")
    assert Morpheme("간다", 7, 9, (("가", "VV"), ("ᆫ다", "EF"))) in tokens
