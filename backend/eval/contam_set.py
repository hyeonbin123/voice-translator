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

RELEASED = date(2026, 5, 21)  # Hy-MT2 open-sourced (model card News); sentences must be published after it
PER_LANGUAGE = 100
USER_AGENT = "voice-translator-eval/1.0 (research; https://github.com/hyeonbin123/voice-translator)"
KOREA_LIST = "https://www.korea.kr/news/policyNewsList.do?pageIndex={page}"
KOREA_VIEW = "https://www.korea.kr/news/policyNewsView.do?newsId={news_id}"
KOGL_TYPE_1 = "공공누리 제1유형"
GV_FEED = "https://globalvoices.org/feed/?paged={page}"
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
SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
TAG = re.compile(r"<[^>]+>")
# Captions, credits, bullets, datelines, links and contact lines are not sentences a person would say.
NOT_PROSE = re.compile(r"https?://|www\.|@|ⓒ|©|사진|자료|출처|문의|=|▲|△|▶|►|■|□|◆|◇|●|○|※|☞|☎|\[|\]|\||…")
EN_NOT_PROSE = re.compile(
    r"\b(?:photo|image|screenshot|video|via|read more|click|subscribe|global voices|donat\w*|coverage)\b"
    r"|^(?:by|this (?:piece|article|story|post)|english translation)\b",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    return [piece.strip() for piece in SENTENCE_END.split(text) if piece.strip()]


def _hangul_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("가" <= c <= "힣" for c in letters) / max(1, len(letters))


def usable(sentence: str, language: str) -> bool:
    """A plain prose sentence of ordinary length that ends like a sentence."""
    if NOT_PROSE.search(sentence) or not sentence.endswith((".", "?", "!")):
        return False
    if language == "ko":
        return 20 <= len(sentence) <= 200 and _hangul_share(sentence) >= 0.8
    latin = all(c.isascii() for c in sentence if c.isalpha())  # typographic quotes are fine
    return 40 <= len(sentence) <= 300 and not EN_NOT_PROSE.search(sentence) and latin


def paragraphs(fragment: str) -> list[str]:
    """Text of each <p>, outside blockquotes and figures (quoted posts and captions are not the article)."""
    fragment = re.sub(r"<(blockquote|figure|figcaption|table)\b.*?</\1>", " ", fragment, flags=re.S | re.I)
    texts = [
        html.unescape(TAG.sub(" ", p)) for p in re.findall(r"<p\b[^>]*>(.*?)</p>", fragment, re.S | re.I)
    ]
    # Removed tags leave a space before punctuation ("Urumqi , the capital"); close it up.
    return [re.sub(r" ([,.;:!?])", r"\1", " ".join(text.split())) for text in texts if text.strip()]


def first_sentence(fragment: str, language: str) -> str | None:
    for paragraph in paragraphs(fragment):
        for sentence in split_sentences(paragraph):
            if usable(sentence, language):
                return sentence
    return None


def korea_article(page: str) -> tuple[date, bool, str] | None:
    """Publication date, whether the text is KOGL Type 1, and the body of a korea.kr article page."""
    published = re.search(r'property="article:published_time" content="(\d{4}-\d{2}-\d{2})', page)
    body = re.search(r'<div class="view_cont">(.*?)<div class="article_footer">', page, re.S)
    if not published or not body:
        return None
    return date.fromisoformat(published.group(1)), KOGL_TYPE_1 in page, body.group(1)


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


def _get(client, url: str) -> str:
    time.sleep(0.5)  # be polite to both sites
    response = client.get(url)
    response.raise_for_status()
    return response.text


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


def print_hash(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    rows = load(path)
    dates = sorted(row["published"] for row in rows)
    print(f"sha256 {digest}, {len(rows)} sentences, published {dates[0]} .. {dates[-1]}")
    return digest


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
