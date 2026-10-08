from datetime import date

import pytest

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


def test_paragraphs_skip_quotes_figures_and_close_up_punctuation():
    fragment = (
        "<p>First <a href='x'>part</a> , then more.</p>"
        "<blockquote><p>A quoted post.</p></blockquote>"
        "<figure><figcaption>A caption.</figcaption></figure>"
        "<p>Second &amp; last.</p>"
    )
    assert paragraphs(fragment) == ["First part, then more.", "Second & last."]


def test_first_sentence_takes_the_first_usable_sentence_of_the_body():
    fragment = "<p>Short one. Room to protest is narrowing across much of the continent this year.</p>"
    assert (
        first_sentence(fragment, "en")
        == "Room to protest is narrowing across much of the continent this year."
    )
    assert first_sentence("<p>Too short.</p>", "en") is None


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


def test_global_voices_own_line_is_not_a_partner_note_but_an_italic_partner_note_is():
    (when_a, link_a, a), (_, _, b) = gv_items(FEED)
    assert when_a.date() == date(2026, 10, 7) and link_a == "https://globalvoices.org/a/"
    a, b = GV_OWN_LINE.sub(" ", a), GV_OWN_LINE.sub(" ", b)
    assert GV_PARTNER.search(a) is None
    assert GV_PARTNER.search(b) is not None
    assert first_sentence(GV_BOILERPLATE.sub(" ", a), "en") == (
        "On a Saturday in late July, the country shut down its internet for the first time."
    )
