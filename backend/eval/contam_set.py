"""The frozen contamination-check set of the Hy-MT2 trial (task T83, docs/experiments.md 15).

FLEURS reuses FLORES sentences, which the vendor's report evaluates on, and the report has no statement that
they were kept out of training. So before any run, 100 Korean and 100 English sentences published after the
model's release (2026-05-21) are frozen and hashed, and Hy-MT2 and opus are scored on them without a
reference (MetricX-24 QE, eval/metricx_score.py). That check can only veto adoption, never support it.

- Korean: 정책브리핑 (korea.kr) policy news; only articles whose page says the text is under 공공누리 제1유형
  (KOGL Type 1: free use with the source named).
- English: Global Voices English posts, published under CC BY 3.0 (its attribution policy); posts republished
  from partners (other terms) are left out. English Wikinews, named in the plan, closed on 2026-05-03.

One sentence per article: the first body sentence that passes `usable`. The file stays in data/contam (not
committed); its SHA-256 is recorded in docs/experiments.md 15. From backend/:

    uv run --no-sync python -m eval.contam_set build --out ../data/contam/v1.jsonl
    uv run --no-sync python -m eval.contam_set hash ../data/contam/v1.jsonl
    uv run --no-sync python -m eval.contam_set translate --set ../data/contam/v1.jsonl --system hymt \
        --en-ko hy-mt2-split --ko-en hy-mt2 --out ../work/vt-n1/contam_outputs.json       (GPU step)
    uv run --no-sync python -m eval.contam_set translate --set ../data/contam/v1.jsonl --system opus \
        --out ../work/vt-n1/contam_outputs.json
"""

# [파일 역할] Hy-MT2 시험의 오염 점검 세트(모델 공개일 뒤에 게시된 한국어 100·영어 100문장)를 만들어
# 고정(build)하고, 해시를 출력하고(hash), 한 번역 시스템으로 번역해 결과 파일에 더하는(translate)
# 도구(docs/experiments.md 15절).
# [설계 이유] 평가 데이터 FLEURS는 FLORES 문장이라 Hy-MT2가 학습 때 봤을 수 있다. 공개일(2026-05-21) 뒤의 새
# 글이면 학습에 들어갔을 수 없으므로, 그 문장에서 Hy-MT2가 opus보다 뚜렷이 나쁘면 FLEURS 점수가 부풀었을
# 가능성을 의심하고 채택을 거부한다(거부만 할 수 있음, 위 docstring). 측정 전에 세트를 만들고 SHA-256을 15절에
# 적어 고정한다. 출처는 재사용이 허락된 글만 쓴다(정책브리핑 공공누리 제1유형, Global Voices CC BY 3.0).
# 세트는 data/contam에 두고 커밋하지 않는다.
# [연결] 번역 결과는 eval/mt_decide.py가 MetricX 입력(metricx-input)과 판정(contam)에 쓰고, 점수는
# eval/metricx_score.py가 매긴다. 번역기는 서버와 같은 services/models.load_translation으로 만든다.

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

# RELEASED: 이 날짜보다 뒤에 게시된 글만 쓴다. PER_LANGUAGE: 언어별 문장 수. USER_AGENT: 사이트에 누가 왜
# 요청하는지 밝히는 머리글(연구용, 저장소 주소). 이어서 두 출처의 목록·기사 주소와 공공누리 제1유형 표시
# 문자열.
RELEASED = date(2026, 5, 21)  # Hy-MT2 open-sourced (model card News); sentences must be published after it
PER_LANGUAGE = 100
USER_AGENT = "voice-translator-eval/1.0 (research; https://github.com/hyeonbin123/voice-translator)"
KOREA_LIST = "https://www.korea.kr/news/policyNewsList.do?pageIndex={page}"
KOREA_VIEW = "https://www.korea.kr/news/policyNewsView.do?newsId={news_id}"
KOGL_TYPE_1 = "공공누리 제1유형"
GV_FEED = "https://globalvoices.org/feed/?paged={page}"
# Global Voices 글 중 다른 매체에서 옮겨 실은 글(다른 라이선스)을 알아보는 표현. 옮겨 실은 글은 뺀다.
GV_PARTNER = re.compile(
    r"originally (?:published|appeared)|first appeared|republished|content[- ]sharing|partnership", re.I
)
# Every Global Voices post carries an "Originally published on Global Voices" line, which must not read as a
# partner's republication note (those are italic editor's notes, checked before they are dropped). That line,
# the tagline, image captions and italic notes are not the article's prose.
GV_OWN_LINE = re.compile(r"<p class='originally-published'>.*?</p>", re.S)
GV_BOILERPLATE = re.compile(
    r"<p><big class='tagline'>.*?</p>"
    r'|<div [^>]*class="wp-caption[^"]*".*?</div>|<p><(em|i)>.*?</\1></p>',
    re.S,
)
# SENTENCE_END: . ? ! 뒤의 공백에서 문장을 나눈다. TAG: HTML 태그.
SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
TAG = re.compile(r"<[^>]+>")
# Captions, credits, bullets, datelines, links and contact lines are not sentences a person would say.
NOT_PROSE = re.compile(r"https?://|www\.|@|ⓒ|©|사진|자료|출처|문의|=|▲|△|▶|►|■|□|◆|◇|●|○|※|☞|☎|\[|\]|\||…")
# 영어 글에서 본문 문장이 아닌 것(사진 설명, 구독·후원 안내, 'By ...' 같은 필자 줄 등)을 거르는 표현.
EN_NOT_PROSE = re.compile(
    r"\b(?:photo|image|screenshot|video|via|read more|click|subscribe|global voices|donat\w*|coverage)\b"
    r"|^(?:by|this (?:piece|article|story|post)|english translation)\b",
    re.I,
)


# 단순 문장 나누기(약어 처리 없음). 이상한 조각은 뒤의 usable이 거른다. _hangul_share는 글자 중 한글 완성형의
# 비율.
def split_sentences(text: str) -> list[str]:
    return [piece.strip() for piece in SENTENCE_END.split(text) if piece.strip()]


def _hangul_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("가" <= c <= "힣" for c in letters) / max(1, len(letters))


# 사람이 말할 법한 평범한 문장인지: 주소·기호·사진 출처 같은 표시가 없고 문장 끝 부호로 끝나야 한다. 한국어는
# 20~200자이고 한글 비율 80% 이상, 영어는 40~300자이고 비본문 표현이 없으며 라틴 글자가 ASCII뿐이어야
# 한다(15절에 적은 조건).
def usable(sentence: str, language: str) -> bool:
    """A plain prose sentence of ordinary length that ends like a sentence."""
    if NOT_PROSE.search(sentence) or not sentence.endswith((".", "?", "!")):
        return False
    if language == "ko":
        return 20 <= len(sentence) <= 200 and _hangul_share(sentence) >= 0.8
    latin = all(c.isascii() for c in sentence if c.isalpha())  # typographic quotes are fine
    return 40 <= len(sentence) <= 300 and not EN_NOT_PROSE.search(sentence) and latin


# HTML 조각에서 <p> 문단의 글자만 뽑는다. 인용·그림·표 안은 기사 본문이 아니므로 먼저 지운다. HTML 엔티티를
# 풀고, 공백을 하나로 줄이고, 태그를 지우며 생긴 '부호 앞 공백'을 붙인다.
def paragraphs(fragment: str) -> list[str]:
    """Text of each <p>, outside blockquotes and figures (quoted posts and captions are not the article)."""
    fragment = re.sub(r"<(blockquote|figure|figcaption|table)\b.*?</\1>", " ", fragment, flags=re.S | re.I)
    texts = [
        html.unescape(TAG.sub(" ", p)) for p in re.findall(r"<p\b[^>]*>(.*?)</p>", fragment, re.S | re.I)
    ]
    # Removed tags leave a space before punctuation ("Urumqi , the capital"); close it up.
    return [re.sub(r" ([,.;:!?])", r"\1", " ".join(text.split())) for text in texts if text.strip()]


# 본문에서 usable을 통과하는 첫 문장(기사당 한 문장). 없으면 None.
def first_sentence(fragment: str, language: str) -> str | None:
    for paragraph in paragraphs(fragment):
        for sentence in split_sentences(paragraph):
            if usable(sentence, language):
                return sentence
    return None


# 정책브리핑 기사 페이지에서 게시일(메타 태그), 공공누리 제1유형 표시가 있는지, 본문 HTML을 뽑는다. 구조가
# 다르면 None.
def korea_article(page: str) -> tuple[date, bool, str] | None:
    """Publication date, whether the text is KOGL Type 1, and the body of a korea.kr article page."""
    published = re.search(r'property="article:published_time" content="(\d{4}-\d{2}-\d{2})', page)
    body = re.search(r'<div class="view_cont">(.*?)<div class="article_footer">', page, re.S)
    if not published or not body:
        return None
    return date.fromisoformat(published.group(1)), KOGL_TYPE_1 in page, body.group(1)


# Global Voices RSS 피드 한 쪽에서 글마다 (게시 시각, 주소, 본문 HTML)을 뽑는다. 셋 중 하나라도 없는 항목은
# 건너뛴다.
def gv_items(feed: str) -> list[tuple[datetime, str, str]]:
    """(publication time, link, content) of each item of a Global Voices feed page."""
    items = []
    for item in re.findall(r"<item>(.*?)</item>", feed, re.S):
        link = re.search(r"<link>(.*?)</link>", item, re.S)
        when = re.search(r"<pubDate>(.*?)</pubDate>", item, re.S)
        content = re.search(r"<content:encoded><!\[CDATA\[(.*?)\]\]></content:encoded>", item, re.S)
        if link and when and content:
            items.append(
                (parsedate_to_datetime(when.group(1).strip()), link.group(1).strip(), content.group(1))
            )
    return items


# 요청마다 0.5초 쉬어 두 사이트에 부담을 주지 않는다. HTTP 오류는 예외로 올려 세트 만들기를 멈춘다.
def _get(client, url: str) -> str:
    time.sleep(0.5)  # be polite to both sites
    response = client.get(url)
    response.raise_for_status()
    return response.text


# 정책 뉴스 목록을 1쪽부터(최대 60쪽) 넘기며 기사마다: 처음 보는 기사인지 → 공개일 뒤 게시이고 공공누리
# 제1유형인지 → 첫 평범한 문장이 있고 이미 고른 문장과 겹치지 않는지 확인해 100개를 모은다. 행에는 문장과 함께
# 주소·게시일·출처·라이선스를 남긴다.
def build_korean(client, count: int = PER_LANGUAGE) -> list[dict]:
    rows, seen, page = [], set(), 1
    while len(rows) < count and page <= 60:
        listing = _get(client, KOREA_LIST.format(page=page))
        for news_id in dict.fromkeys(re.findall(r"policyNewsView\.do\?newsId=(\d+)", listing)):
            if news_id in seen or len(rows) >= count:
                continue
            seen.add(news_id)
            url = KOREA_VIEW.format(news_id=news_id)
            article = korea_article(_get(client, url))
            if article is None:
                continue
            published, kogl, body = article
            if published <= RELEASED or not kogl:
                continue
            sentence = first_sentence(body, "ko")
            if sentence and all(sentence != row["text"] for row in rows):
                rows.append(
                    {
                        "id": f"ko-{len(rows) + 1:03d}",
                        "lang": "ko",
                        "text": sentence,
                        "url": url,
                        "published": published.isoformat(),
                        "source": "정책브리핑 korea.kr",
                        "licence": "KOGL Type 1 (text)",
                    }
                )
        page += 1
    return rows


# Global Voices 피드를 1쪽부터(최대 40쪽) 넘기며 글마다: 모든 글에 붙는 'Originally published on Global
# Voices' 줄을 먼저 지운 뒤(옮겨 실은 글 표시로 오인하지 않게), 공개일 뒤 게시이고 옮겨 실은 글이 아닌지 →
# 머리말·그림 설명·기울임 안내를 뺀 본문의 첫 평범한 문장을 고른다.
def build_english(client, count: int = PER_LANGUAGE) -> list[dict]:
    rows, page = [], 1
    while len(rows) < count and page <= 40:
        for published, link, raw in gv_items(_get(client, GV_FEED.format(page=page))):
            content = GV_OWN_LINE.sub(" ", raw)
            if len(rows) >= count or published.date() <= RELEASED or GV_PARTNER.search(content):
                continue
            sentence = first_sentence(GV_BOILERPLATE.sub(" ", content), "en")
            if sentence and all(sentence != row["text"] for row in rows):
                rows.append(
                    {
                        "id": f"en-{len(rows) + 1:03d}",
                        "lang": "en",
                        "text": sentence,
                        "url": link,
                        "published": published.date().isoformat(),
                        "source": "Global Voices",
                        "licence": "CC BY 3.0",
                    }
                )
        page += 1
    return rows


# 두 언어를 모아 각각 정확히 100개가 아니면 멈춘다. 세트는 한 번만 고정하므로 같은 파일이 이미 있으면 덮어쓰지
# 않고 멈춘다(바꾸려면 새 버전 파일로). 쓰고 나면 해시와 게시일 범위를 출력한다.
def build(out: Path) -> None:
    import httpx

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as client:
        rows = build_korean(client) + build_english(client)
    counts = {language: sum(row["lang"] == language for row in rows) for language in ("ko", "en")}
    if counts != {"ko": PER_LANGUAGE, "en": PER_LANGUAGE}:
        raise SystemExit(f"not enough sentences: {counts}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} exists: the set is frozen once; write a new version instead")
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"{counts} -> {out}")
    print_hash(out)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# 세트 파일의 SHA-256, 문장 수, 게시일 범위를 출력한다. 15절에 적힌 해시와 대조해 세트가 바뀌지 않았는지
# 확인하는 데 쓴다.
def print_hash(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    rows = load(path)
    dates = sorted(row["published"] for row in rows)
    print(f"sha256 {digest}, {len(rows)} sentences, published {dates[0]} .. {dates[-1]}")
    return digest


# 서버와 같은 설정 경로(Settings → load_translation)로 번역기를 만든다. Hy-MT2 모델 이름과 기대 digest는
# hymt_setup.QUANTS에서 가져와 측정한 빌드만 쓰게 하고, 요청 제한은 120초로 넉넉히 준다.
# hymt 시스템이면 그 설정에서 Hy-MT2가 맡는 방향만 번역하고(영어 문장 → 영→한, 한국어 문장 → 한→영), opus는 두
# 방향 모두 번역한다. 결과 파일에 시스템 이름으로 더하고, 어떤 번역기·어떤 세트(해시)로 만들었는지 _meta에
# 남긴다.
def translate(args: argparse.Namespace) -> None:
    """Translate the frozen set with one system and add it to the outputs file (GPU step)."""
    from app.config import Settings
    from app.services.models import load_translation
    from eval.hymt_setup import QUANTS

    _, _, _, model, digest = QUANTS[args.hymt_quant]
    settings = Settings(
        en_ko_translation=args.en_ko,
        ko_en_translation=args.ko_en,
        hymt_model=model,
        hymt_digest=digest,
        hymt_timeout_s=120,
    )
    translator = load_translation(settings, {"device": "cuda", "compute_type": "float16"})
    rows = load(args.set)
    out = Path(args.out)
    outputs = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    system: dict[str, dict[str, str]] = {}
    hymt_directions = {"en-ko": args.en_ko != "opus", "ko-en": args.ko_en != "opus"}
    for direction, language in (("en-ko", "en"), ("ko-en", "ko")):
        if args.system == "hymt" and not hymt_directions[direction]:
            continue  # only the directions Hy-MT2 translates in this setting
        source, target = direction.split("-")
        system[direction] = {
            row["id"]: translator.translate(row["text"], source, target)
            for row in rows
            if row["lang"] == language
        }
    outputs[args.system] = system
    outputs.setdefault("_meta", {})[args.system] = {
        "translator": translator.model_name,
        "set_sha256": hashlib.sha256(Path(args.set).read_bytes()).hexdigest(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(outputs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.system}: {translator.model_name} -> {out}")


# 하위 명령 build / hash / translate.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    h = commands.add_parser("hash")
    h.add_argument("path", type=Path)
    t = commands.add_parser("translate")
    t.add_argument("--set", type=Path, required=True)
    t.add_argument("--system", choices=("hymt", "opus"), required=True)
    t.add_argument("--en-ko", choices=("opus", "hy-mt2", "hy-mt2-split"), default="opus")
    t.add_argument("--ko-en", choices=("opus", "hy-mt2"), default="opus")
    t.add_argument("--hymt-quant", choices=("q8_0", "q4_k_m"), default="q8_0")
    t.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
    elif args.command == "hash":
        print_hash(args.path)
    else:
        translate(args)


if __name__ == "__main__":
    main()
