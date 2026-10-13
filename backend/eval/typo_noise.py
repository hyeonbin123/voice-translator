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

# [파일 역할] FLEURS 병렬 문장에 규칙으로 자판 오타를 섞어 '오타 입력 번역' 평가
# 세트(data/typo/<split>.jsonl)를 만든다(T32, docs/experiments.md 6절). 한 행에 문장 ID, 깨끗한 한·영 원문,
# 언어마다 가벼운 오타(p=0.1)·심한 오타(p=0.25) 판이 들어간다. 참조 번역은 깨끗한 원문 그대로다.
# [왜 필요한가] 글자 입력에 오타가 있으면 번역이 잘 안 나오는 문제가 있었다(6절 문제). 오타 교정
# 후보(B·C·D)를 공정하게 비교하려면 누구나 똑같이 다시 만들 수 있는 고정된 오타 세트가 필요하다. 그래서 시드를
# 문장·언어·수준마다 고정했다.
# [한계] 합성 오타라 실제 사람의 오타 분포와 다르다(6절 한계).
# [연결] eval/typo_eval.py가 이 파일을 읽고, T83(15절)도 같은 세트를 쓴다. 오타 규칙은
# tests/test_typo_noise.py가 검증한다.

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable

# 데이터 폴더 경로만 가져온다. FLEURS를 읽는 load_pairs는 eval 그룹이 필요해 build 안에서 가져온다(규칙 함수와
# 테스트는 eval 그룹 없이 돈다).
from eval.common import DATA

# LEVELS: 오타 수준별 '단어 하나에 오타가 생길 확률'(가벼움 0.1 = 문장당 1~2개, 심함 0.25). SEEDS: 분할별
# 시드(validation 13, test 29, 6절 평가 세트).
LEVELS = {"light": 0.1, "heavy": 0.25}
SEEDS = {"validation": 13, "test": 29}

# 한글 음절을 초성·중성·종성으로 나누고 합치는 데 쓰는 자모 표. 순서는 유니코드 한글 음절 배열 순서와 같아야
# 한다(초성 19, 중성 21, 종성 28: 받침 없음 '' 포함).
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]
# 두벌식 자판에서 자모 → 영문 키 표와 그 역표. 기본 자모 26개만 있고, 쌍자음·겹모음·겹받침(Shift나 두 번
# 누름)은 표에 없다.
# The key each jamo sits on in the 2-set layout, and the reverse.
DUBEOLSIK = dict(
    zip("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡ", "qwertyuiopasdfghjklzxcvbnm", strict=True)
)
KEY_TO_JAMO = {key: jamo for jamo, key in DUBEOLSIK.items()}
# QWERTY 자판의 세 줄. '옆 키'는 같은 줄의 왼쪽·오른쪽 키로 본다(위아래 줄은 보지 않음).
ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


# 키 하나의 같은 줄 좌우 키 목록. 어느 줄에도 없으면 빈 목록.
# 주의: 빈 문자열을 넣으면 ''가 모든 문자열에 포함된 것으로 판정돼 첫 줄의 'q' 위치로 보고 ['w']를 돌려준다.
# _korean_neighbour_key는 표에 없는 자모(받침 없음, 쌍자음 등)에 빈 문자열을 넘기므로 이 경우가 생긴다.
def neighbours(key: str) -> list[str]:
    """Keys to the left and right on the same keyboard row."""
    for row in ROWS:
        if key in row:
            i = row.index(key)
            return [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
    return []


# 완성형 한글 음절(가~힣)인지.
def is_syllable(char: str) -> bool:
    return "가" <= char <= "힣"


# 한글 음절을 (초성, 중성, 종성) 번호로 나눈다. 음절 코드 = 0xAC00 + (초성 × 21 + 중성) × 28 + 종성 이라는
# 유니코드 규칙을 거꾸로 푼다.
def decompose(char: str) -> tuple[int, int, int]:
    code = ord(char) - ord("가")
    return code // (21 * 28), code // 28 % 21, code % 28


# (초성, 중성, 종성) 번호로 한글 음절을 만든다(decompose의 역).
def compose(cho: int, jung: int, jong: int) -> str:
    return chr(ord("가") + (cho * 21 + jung) * 28 + jong)


# 한국어 오타 '옆 키를 잘못 누름': 단어의 음절마다 초성·중성·종성 각각을 두벌식 자판의 옆 키 자모로 바꾼 모든
# 경우 중 하나를 무작위로 고른다. 옆 키가 다른 종류의 자모(예: 초성 자리에 모음 키)면 그 경우는 버린다. 바꿀
# 수 없으면 None.
# 주의: 받침이 없는 음절이나 쌍자음 초성처럼 DUBEOLSIK 표에 없는 자모는 neighbours('')가 ['w'](ㅈ)를 돌려줘,
# 'ㅈ' 받침이 새로 붙거나 초성이 ㅈ으로 바뀌는 경우도 후보에 섞인다. 실제 옆 키 오타는 아니다.
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


# 한국어 오타 '받침 빠짐': 받침이 있는 음절 하나를 골라 받침을 없앤다. 받침이 있는 음절이 없으면 None.
def _korean_dropped_final(word: str, rng: random.Random) -> str | None:
    positions = [i for i, char in enumerate(word) if is_syllable(char) and decompose(char)[2]]
    if not positions:
        return None
    i = rng.choice(positions)
    cho, jung, _ = decompose(word[i])
    return word[:i] + compose(cho, jung, 0) + word[i + 1 :]


# 띄어쓰기 오타 '단어 안에 공백': letter 조건을 만족하는 두 글자 사이 중 한 곳에 공백을 넣는다. 한국어
# 규칙에서 쓴다(letter = 한글 음절).
def _split_inside(word: str, rng: random.Random, letter: Callable[[str], bool]) -> str | None:
    cuts = [i for i in range(1, len(word)) if letter(word[i - 1]) and letter(word[i])]
    if not cuts:
        return None
    i = rng.choice(cuts)
    return word[:i] + " " + word[i:]


# '이웃한 두 글자 순서 바뀜': 서로 다른 두 이웃 글자(둘 다 letter)를 맞바꾼다. 같은 글자끼리 바꾸면 변화가
# 없어 후보에서 뺀다. 한국어(음절)·영어(알파벳) 모두 쓴다.
def _swap_neighbours(word: str, rng: random.Random, letter: Callable[[str], bool]) -> str | None:
    pairs = [
        i for i in range(len(word) - 1) if letter(word[i]) and letter(word[i + 1]) and word[i] != word[i + 1]
    ]
    if not pairs:
        return None
    i = rng.choice(pairs)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


# 영어 오타 '옆 키': 글자 하나를 QWERTY 같은 줄 좌우 키로 바꾼 경우 중 하나. 대문자면 바꾼 키도 대문자로 둔다.
# 숫자·문장부호는 자판 줄에 없어 후보가 되지 않는다.
def _english_neighbour_key(word: str, rng: random.Random) -> str | None:
    options = []
    for i, char in enumerate(word):
        for key in neighbours(char.lower()):
            options.append(word[:i] + (key.upper() if char.isupper() else key) + word[i + 1 :])
    return rng.choice(options) if options else None


# 영어 오타 '글자 빠짐'. 알파벳이 한 글자뿐인 단어는 사라지므로 두 글자 이상일 때만 적용한다.
def _english_dropped(word: str, rng: random.Random) -> str | None:
    positions = [i for i, char in enumerate(word) if char.isalpha()]
    if len(positions) < 2:
        return None
    i = rng.choice(positions)
    return word[:i] + word[i + 1 :]


# 영어 오타 '글자 두 번': 알파벳 하나를 한 번 더 쓴다.
def _english_doubled(word: str, rng: random.Random) -> str | None:
    positions = [i for i, char in enumerate(word) if char.isalpha()]
    if not positions:
        return None
    i = rng.choice(positions)
    return word[: i + 1] + word[i] + word[i + 1 :]


# MERGE: '앞 단어와 붙음'은 단어 하나만 보고 만들 수 없어(앞 단어가 필요) 함수 대신 표지 문자열로 두고
# add_typos가 처리한다.
# WORD_TYPOS: 언어별 오타 종류 목록(6절 평가 세트의 종류와 같음). 람다는 공통 함수에 언어별 글자 조건을 묶은
# 것이다.
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


# 문장의 단어(공백으로 나뉜 덩어리)마다 확률 rate로 오타 하나를 넣는다. 종류는 무작위 순서로 시도해 처음
# 적용되는 것을 쓴다(6절). 입력: 원문, 언어, 확률, 난수 생성기. 출력: 오타 문장.
def add_typos(text: str, language: str, rate: float, rng: random.Random) -> str:
    """Give each word one mistake with probability `rate`; types are tried in random order."""
    words = text.split(" ")
    out: list[str] = []
    for index, word in enumerate(words):
        if word and rng.random() < rate:
            # 종류 목록을 무작위로 섞어 차례로 시도한다. MERGE는 앞에 붙일 단어가 있을 때만 적용해 이 단어를
            # 앞 단어 끝에 이어 붙인다. 다른 종류는 None이거나 결과가 같으면 다음 종류로 넘어간다.
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
                # 어떤 종류도 적용되지 않았으면(for-else) 단어를 그대로 둔다.
                out.append(word)
            continue
        out.append(word)
    return " ".join(out)


# 분할의 FLEURS 병렬 문장(eval/mt_eval.py의 load_pairs, 번역 평가와 같은 문장 묶음)마다 두 언어 × 두 수준의
# 오타 판을 만든다.
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


# 명령줄 진입점. 세트를 JSONL로 쓰고, 언어·수준별로 오타가 실제로 들어간 문장 수와 예시 한 줄을 출력한다(6절
# 결과 머리의 '오타가 들어간 문장' 수가 이것이다).
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
