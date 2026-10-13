# [파일 역할] eval/contam_set.py(T83 오염 점검 세트 만들기)의 문장 고르기·HTML 읽기 규칙을 네트워크 없이
# 검증하는 테스트 묶음이다.
# [배경] FLEURS는 FLORES 문장이라 Hy-MT2가 학습 때 봤을 수 있다. 그래서 모델 공개일 뒤에 나온
# 정책브리핑(공공누리 제1유형)과 Global Voices(CC BY 3.0) 글에서 기사마다 '첫 평범한 문장' 하나씩 200문장을
# 고정해 참조 없는 점수로 견주기로 했다(docs/experiments.md 15절 '오염 점검 세트').
# [왜 테스트하나] 세트는 한 번 만들어 해시로 고정하므로, 사진 설명·서명·후원 안내·다른 매체에서 옮긴 글처럼
# '사람이 말할 문장이 아닌 것'이나 라이선스 조건이 다른 글이 섞이면 되돌릴 수 없다. 실제 페이지 모양을 줄인
# HTML 조각으로 각 걸러내기 규칙을 확인한다.

from datetime import date

import pytest

# 검증 대상: usable(평범한 문장인지), paragraphs(본문 문단 추출), first_sentence(첫 쓸 수 있는 문장),
# korea_article(정책브리핑 기사 페이지 읽기), gv_items(Global Voices RSS 읽기), Global Voices 정규식 셋(자기
# 글 표시 줄, 다른 매체 게재 안내, 꼬리말·사진 설명).
from eval.contam_set import (
    GV_BOILERPLATE,
    GV_OWN_LINE,
    GV_PARTNER,
    first_sentence,
    gv_items,
    korea_article,
    paragraphs,
    usable,
)


# usable의 경계 사례. 한국어: 평범한 문장은 통과, 너무 짧음·통신사 머리('=')·사진 표시('▲')·누리집 주소·마침표
# 없음은 탈락. 영어: 평범한 문장과 둥근 따옴표(’)가 든 문장은 통과(라틴 문자 검사에서 문장부호는 보지 않음),
# 후원 안내·서명(By ...)·다른 매체 첫 게재 안내·사진 설명·40자 미만은 탈락(15절의 영어 기준 40~300자, 한국어
# 20~200자).
@pytest.mark.parametrize(
    ("sentence", "language", "ok"),
    [
        ("정부는 내년부터 청년 주거 지원을 두 배로 늘리기로 했다.", "ko", True),
        ("짧다.", "ko", False),
        ("(서울=연합뉴스) 정부는 내년부터 지원을 늘리기로 했다.", "ko", False),
        ("▲ 행사장 전경을 보여 주는 사진이다.", "ko", False),
        ("자세한 내용은 누리집(www.korea.kr)에서 볼 수 있다.", "ko", False),
        ("정부는 내년부터 청년 주거 지원을 두 배로 늘리기로 했다", "ko", False),  # no full stop
        (
            "Room to protest is narrowing across much of the continent, as governments lean on arrests.",
            "en",
            True,
        ),
        (
            "Cambodia’s law was supposed to ensure the right of citizens to organize public assemblies.",
            "en",
            True,
        ),
        ("You can support this coverage by donating here.", "en", False),
        ("By Adele Roeder, Orit Novak, Chelsea Adams and Zahiris P.", "en", False),
        ("This article by Fabiola Chambi first appeared in CONNECTAS, in Spanish.", "en", False),
        ("Photo of the square taken by a protester in the early morning hours.", "en", False),
        ("Short sentence here.", "en", False),
    ],
)
def test_usable(sentence, language, ok):
    assert usable(sentence, language) is ok


# 본문 문단만 뽑는지: 인용 블록(다른 글을 옮긴 것)과 그림·설명은 빼고, 태그를 지우며 생긴 '쉼표 앞 공백'을
# 붙이고, HTML 엔티티(&amp;)를 풀어야 한다.
def test_paragraphs_skip_quotes_figures_and_close_up_punctuation():
    fragment = (
        "<p>First <a href='x'>part</a> , then more.</p>"
        "<blockquote><p>A quoted post.</p></blockquote>"
        "<figure><figcaption>A caption.</figcaption></figure>"
        "<p>Second &amp; last.</p>"
    )
    assert paragraphs(fragment) == ["First part, then more.", "Second & last."]


# 첫 문장이 너무 짧으면 건너뛰고 같은 문단의 다음 쓸 수 있는 문장을 고르는지, 쓸 수 있는 문장이 없으면
# None인지.
def test_first_sentence_takes_the_first_usable_sentence_of_the_body():
    fragment = "<p>Short one. Room to protest is narrowing across much of the continent this year.</p>"
    assert (
        first_sentence(fragment, "en")
        == "Room to protest is narrowing across much of the continent this year."
    )
    assert first_sentence("<p>Too short.</p>", "en") is None


# 정책브리핑 기사 페이지에서 게시일, '공공누리 제1유형' 표시 여부, 본문을 읽는지. 기사가 아닌 페이지는 None,
# 제4유형(변경 금지 등 다른 조건)이면 제1유형이 아님으로 읽어야 한다. 제1유형만 쓰기로 했기 때문이다.
def test_korea_article_reads_date_licence_and_body():
    page = (
        '<meta property="article:published_time" content="2026-10-08T16:59:00Z">'
        '<div class="article_body"><div class="view_cont">'
        "<p>정부는 내년부터 청년 주거 지원을 늘리기로 했다.</p></div>"
        '<div class="article_footer"><span>이 자료는 텍스트에 한하여 공공누리 제1유형(출처표시)의 조건</span>'
    )
    published, kogl, body = korea_article(page)
    assert published == date(2026, 10, 8) and kogl
    assert first_sentence(body, "ko") == "정부는 내년부터 청년 주거 지원을 늘리기로 했다."
    assert korea_article("<html>no article</html>") is None
    assert korea_article(page.replace("공공누리 제1유형", "공공누리 제4유형"))[1] is False


# Global Voices RSS를 흉내 낸 조각들. OWN은 자기 글(모든 글에 붙는 'Originally published on Global Voices' 줄,
# 꼬리말, 사진 설명, 본문), PARTNER는 다른 매체에서 옮겨 실은 글(기울임 편집자 주)이다.
ITEM = (
    "<item><link>{link}</link><pubDate>{when}</pubDate>"
    "<content:encoded><![CDATA[{content}]]></content:encoded></item>"
)
OWN = (
    "<p class='originally-published'><small>Originally published on Global Voices</small></p>"
    "<p><big class='tagline'><em>A tagline that is not prose at all, really.</em></big></p>"
    '<div id="x" class="wp-caption alignnone"><p class="wp-caption-text">A caption of the picture.</p></div>'
    "<p>On a Saturday in late July, the country shut down its internet for the first time.</p>"
)
PARTNER = (
    "<p><em>This story was originally published by a partner site and is republished here.</em></p>"
    "<p>Tree planting was so central to planning for the capital that it was called a park city.</p>"
)
FEED = (
    "<rss><channel>"
    + ITEM.format(link="https://globalvoices.org/a/", when="Wed, 07 Oct 2026 18:02:48 +0000", content=OWN)
    + ITEM.format(link="https://globalvoices.org/b/", when="Tue, 06 Oct 2026 08:05:24 +0000", content=PARTNER)
    + "</channel></rss>"
)


# RSS 항목에서 게시 시각·링크·내용을 읽고, 모든 글에 붙는 자기 게재 표시 줄을 지운 뒤에는 자기 글이 '다른 매체
# 글'로 잘못 걸리지 않고 실제로 옮겨 실은 글만 걸리는지 확인한다. 'originally published'라는 말이 두 경우에
# 모두 나와 순서가 틀리면 모든 글이 빠지거나 옮긴 글이 섞인다. 끝으로 꼬리말·사진 설명을 지운 뒤 첫 문장이
# 본문 문장인지 본다.
def test_global_voices_own_line_is_not_a_partner_note_but_an_italic_partner_note_is():
    (when_a, link_a, a), (_, _, b) = gv_items(FEED)
    assert when_a.date() == date(2026, 10, 7) and link_a == "https://globalvoices.org/a/"
    a, b = GV_OWN_LINE.sub(" ", a), GV_OWN_LINE.sub(" ", b)
    assert GV_PARTNER.search(a) is None
    assert GV_PARTNER.search(b) is not None
    assert first_sentence(GV_BOILERPLATE.sub(" ", a), "en") == (
        "On a Saturday in late July, the country shut down its internet for the first time."
    )
