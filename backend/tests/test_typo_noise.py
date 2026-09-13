import random

import pytest

from eval.typo_noise import (
    CHO,
    JONG,
    JUNG,
    add_typos,
    compose,
    decompose,
    is_syllable,
    neighbours,
)

KOREAN = "겨울은 믿을 수 없을 만큼 쌀쌀해지기도 한다 영하 이하로 내려가는 날은 드물지만"
ENGLISH = "Winter can be surprisingly cold even though days below freezing are rare"


def test_hangul_syllables_decompose_and_compose_back():
    assert decompose("강") == (CHO.index("ㄱ"), JUNG.index("ㅏ"), JONG.index("ㅇ"))
    assert all(compose(*decompose(char)) == char for char in "가각힣쌀쌁뭐")


def test_neighbouring_keys_stay_on_the_same_row():
    assert neighbours("q") == ["w"]
    assert neighbours("g") == ["f", "h"]
    assert neighbours("1") == []


@pytest.mark.parametrize(("text", "language"), [(KOREAN, "ko"), (ENGLISH, "en")])
def test_the_same_seed_gives_the_same_typos(text, language):
    first = add_typos(text, language, 0.5, random.Random("seed"))
    second = add_typos(text, language, 0.5, random.Random("seed"))
    assert first == second
    assert first != text


@pytest.mark.parametrize(("text", "language"), [(KOREAN, "ko"), (ENGLISH, "en")])
def test_rate_zero_changes_nothing(text, language):
    assert add_typos(text, language, 0.0, random.Random(1)) == text


def test_korean_typos_keep_whole_syllables():
    # Mistakes are made by recomposing syllables, never by leaving loose jamo behind.
    for seed in range(50):
        noisy = add_typos(KOREAN, "ko", 1.0, random.Random(seed))
        assert all(is_syllable(char) or char == " " for char in noisy)


def test_typos_are_small_edits():
    # One mistake per word: the noisy sentence keeps close to the original length.
    for seed in range(50):
        noisy = add_typos(ENGLISH, "en", 1.0, random.Random(seed))
        assert abs(len(noisy) - len(ENGLISH)) <= len(ENGLISH.split())
