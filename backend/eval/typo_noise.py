"""Add typing mistakes to the FLEURS sentence pairs (task T32).

Usage, from backend/ with the eval group installed (it reads the FLEURS parquet files):
    uv run python -m eval.typo_noise --split validation

Writes data/typo/<split>.jsonl (git-ignored): each sentence pair with its clean text and a light and a
heavy typo version of both sides. The rules are in docs/experiments.md 6: every word (space-separated
chunk) gets one mistake with probability p. Korean mistakes: a jamo typed with the neighbouring key on
the 2-set (dubeolsik) keyboard, a dropped final consonant, a missing or extra space, two neighbouring
syllables swapped. English mistakes: a neighbouring QWERTY key, a dropped letter, a doubled letter, two
neighbouring letters swapped, a missing space. The seed is fixed, so the same files come out every time.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable

from eval.common import DATA

LEVELS = {"light": 0.1, "heavy": 0.25}
SEEDS = {"validation": 13, "test": 29}

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]
# The key each jamo sits on in the 2-set layout, and the reverse.
DUBEOLSIK = dict(
    zip("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡ", "qwertyuiopasdfghjklzxcvbnm", strict=True)
)
KEY_TO_JAMO = {key: jamo for jamo, key in DUBEOLSIK.items()}
ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def neighbours(key: str) -> list[str]:
    """Keys to the left and right on the same keyboard row."""
    for row in ROWS:
        if key in row:
            i = row.index(key)
            return [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
    return []


def is_syllable(char: str) -> bool:
    return "가" <= char <= "힣"


def decompose(char: str) -> tuple[int, int, int]:
    code = ord(char) - ord("가")
    return code // (21 * 28), code // 28 % 21, code % 28


def compose(cho: int, jung: int, jong: int) -> str:
    return chr(ord("가") + (cho * 21 + jung) * 28 + jong)


def _korean_neighbour_key(word: str, rng: random.Random) -> str | None:
    options = []
    for i, char in enumerate(word):
        if not is_syllable(char):
            continue
        cho, jung, jong = decompose(char)
        for slot, jamo in ((0, CHO[cho]), (1, JUNG[jung]), (2, JONG[jong])):
            for key in neighbours(DUBEOLSIK.get(jamo, "")):
                typed = KEY_TO_JAMO[key]
                parts = [cho, jung, jong]
                if slot == 0 and typed in CHO:
                    parts[0] = CHO.index(typed)
                elif slot == 1 and typed in JUNG:
                    parts[1] = JUNG.index(typed)
                elif slot == 2 and typed in JONG:
                    parts[2] = JONG.index(typed)
                else:
                    continue  # the neighbouring key types the other kind of jamo
                options.append(word[:i] + compose(*parts) + word[i + 1 :])
    return rng.choice(options) if options else None


def _korean_dropped_final(word: str, rng: random.Random) -> str | None:
    positions = [i for i, char in enumerate(word) if is_syllable(char) and decompose(char)[2]]
    if not positions:
        return None
    i = rng.choice(positions)
    cho, jung, _ = decompose(word[i])
    return word[:i] + compose(cho, jung, 0) + word[i + 1 :]


def _split_inside(word: str, rng: random.Random, letter: Callable[[str], bool]) -> str | None:
    cuts = [i for i in range(1, len(word)) if letter(word[i - 1]) and letter(word[i])]
    if not cuts:
        return None
    i = rng.choice(cuts)
    return word[:i] + " " + word[i:]


def _swap_neighbours(word: str, rng: random.Random, letter: Callable[[str], bool]) -> str | None:
    pairs = [
        i for i in range(len(word) - 1) if letter(word[i]) and letter(word[i + 1]) and word[i] != word[i + 1]
    ]
    if not pairs:
        return None
    i = rng.choice(pairs)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def _english_neighbour_key(word: str, rng: random.Random) -> str | None:
    options = []
    for i, char in enumerate(word):
        for key in neighbours(char.lower()):
            options.append(word[:i] + (key.upper() if char.isupper() else key) + word[i + 1 :])
    return rng.choice(options) if options else None


def _english_dropped(word: str, rng: random.Random) -> str | None:
    positions = [i for i, char in enumerate(word) if char.isalpha()]
    if len(positions) < 2:
        return None
    i = rng.choice(positions)
    return word[:i] + word[i + 1 :]


def _english_doubled(word: str, rng: random.Random) -> str | None:
    positions = [i for i, char in enumerate(word) if char.isalpha()]
    if not positions:
        return None
    i = rng.choice(positions)
    return word[: i + 1] + word[i] + word[i + 1 :]


MERGE = "merge"  # joined to the previous word: handled in add_typos
WORD_TYPOS: dict[str, list] = {
    "ko": [
        _korean_neighbour_key,
        _korean_dropped_final,
        MERGE,
        lambda word, rng: _split_inside(word, rng, is_syllable),
        lambda word, rng: _swap_neighbours(word, rng, is_syllable),
    ],
    "en": [
        _english_neighbour_key,
        _english_dropped,
        _english_doubled,
        lambda word, rng: _swap_neighbours(word, rng, str.isalpha),
        MERGE,
    ],
}


def add_typos(text: str, language: str, rate: float, rng: random.Random) -> str:
    """Give each word one mistake with probability `rate`; types are tried in random order."""
    words = text.split(" ")
    out: list[str] = []
    for index, word in enumerate(words):
        if word and rng.random() < rate:
            for typo in rng.sample(WORD_TYPOS[language], len(WORD_TYPOS[language])):
                if typo == MERGE:
                    if index > 0 and out:
                        out[-1] += word
                        break
                    continue
                changed = typo(word, rng)
                if changed is not None and changed != word:
                    out.append(changed)
                    break
            else:
                out.append(word)
            continue
        out.append(word)
    return " ".join(out)


def build(split: str) -> list[dict]:
    # Imported here: it needs the eval group (pyarrow, sacrebleu), the typo rules and their tests don't.
    from eval.mt_eval import load_pairs

    rows = []
    for pair in load_pairs(split, None):
        row = {"id": pair["id"], "ko": pair["ko"], "en": pair["en"]}
        for level, rate in LEVELS.items():
            for language in ("ko", "en"):
                # A string seed per sentence keeps every row reproducible on its own.
                rng = random.Random(f"{SEEDS[split]}:{pair['id']}:{language}:{level}")
                row[f"{language}_{level}"] = add_typos(pair[language], language, rate, rng)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(SEEDS), default="validation")
    args = parser.parse_args()

    rows = build(args.split)
    target = DATA / "typo" / f"{args.split}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    for language in ("ko", "en"):
        for level in LEVELS:
            changed = sum(row[language] != row[f"{language}_{level}"] for row in rows)
            print(f"{language}/{level}: {changed}/{len(rows)} sentences changed")
    print(f"example: {rows[0]['ko']}\n      -> {rows[0]['ko_heavy']}")
    print(f"written to {target}")


if __name__ == "__main__":
    main()
