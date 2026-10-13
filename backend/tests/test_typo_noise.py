# [파일 역할] 오타 입력 번역 평가 세트를 만드는 eval/typo_noise.py의 오타 규칙을 검증한다(규칙은
# docs/experiments.md 6절).
# [왜 중요한가] 오타 교정 후보를 공정하게 비교하려면 누구나 같은 오타 세트를 다시 만들 수 있어야 한다. 그래서
# 같은 시드 → 같은 결과, 오타 확률 0 → 변화 없음, 한국어 오타는 온전한 음절로만 생김, 단어당 오타 하나(작은
# 편집)라는 성질을 고정한다.
# [방법] 규칙 함수는 eval 그룹(pyarrow 등) 없이 불러올 수 있게 분리돼 있어 CI에서도 돈다.

import random

import pytest

# CHO·JUNG·JONG은 유니코드 한글 음절 배열 순서의 초성·중성·종성 표다. decompose·compose는 음절을 자모 번호로
# 나누고 다시 합치며, neighbours는 QWERTY 자판에서 같은 줄의 좌우 키를 준다.
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

# 실제 평가 문장(FLEURS)과 비슷한 길이의 한국어·영어 시험 문장.
KOREAN = "겨울은 믿을 수 없을 만큼 쌀쌀해지기도 한다 영하 이하로 내려가는 날은 드물지만"
ENGLISH = "Winter can be surprisingly cold even though days below freezing are rare"


# '강'이 (ㄱ, ㅏ, ㅇ)의 번호로 나뉘고, 받침 없는 음절·겹받침(쌁)·이중 모음(뭐)·마지막 음절(힣)까지 나눴다
# 합치면 원래 글자로 돌아오는지 확인한다. 한국어 오타는 자모를 바꾼 뒤 다시 합쳐 만들므로 이 왕복이 정확해야
# 한다.
def test_hangul_syllables_decompose_and_compose_back():
    assert decompose("강") == (CHO.index("ㄱ"), JUNG.index("ㅏ"), JONG.index("ㅇ"))
    assert all(compose(*decompose(char)) == char for char in "가각힣쌀쌁뭐")


# 옆 키 오타는 같은 줄의 좌우 키만 쓴다. 줄 끝 키(q)는 한쪽만, 숫자처럼 표에 없는 키는 옆 키가 없다.
def test_neighbouring_keys_stay_on_the_same_row():
    assert neighbours("q") == ["w"]
    assert neighbours("g") == ["f", "h"]
    assert neighbours("1") == []


# 같은 시드의 난수 생성기로 두 번 만들면 오타 결과가 똑같아야 한다(평가 세트 재현성). 확률 0.5면 이 문장에서는
# 실제로 무엇인가 바뀌어야 한다.
@pytest.mark.parametrize(("text", "language"), [(KOREAN, "ko"), (ENGLISH, "en")])
def test_the_same_seed_gives_the_same_typos(text, language):
    first = add_typos(text, language, 0.5, random.Random("seed"))
    second = add_typos(text, language, 0.5, random.Random("seed"))
    assert first == second
    assert first != text


# 오타 확률 0이면 문장이 그대로다. 확률을 판정하기 전에 단어를 건드리는 회귀를 잡는다.
@pytest.mark.parametrize(("text", "language"), [(KOREAN, "ko"), (ENGLISH, "en")])
def test_rate_zero_changes_nothing(text, language):
    assert add_typos(text, language, 0.0, random.Random(1)) == text


# 모든 단어에 오타를 넣어도(확률 1.0) 결과에 낱자모(ㄱ, ㅏ 등)가 남지 않고 완성형 음절과 공백만 있는지 50개
# 시드로 확인한다. 오타는 자모를 바꾼 뒤 음절로 다시 합쳐 만든다.
def test_korean_typos_keep_whole_syllables():
    # Mistakes are made by recomposing syllables, never by leaving loose jamo behind.
    for seed in range(50):
        noisy = add_typos(KOREAN, "ko", 1.0, random.Random(seed))
        assert all(is_syllable(char) or char == " " for char in noisy)


# 오타 한 번은 글자를 하나 빼거나, 겹치거나, 공백을 빼는 정도라 길이가 많아야 1 바뀐다. 그래서 모든 단어에
# 오타를 넣어도 문장 길이 차이가 단어 수를 넘지 않아야 한다. 한 단어에 오타가 여러 번 들어가는 회귀를 잡는다.
def test_typos_are_small_edits():
    # One mistake per word: the noisy sentence keeps close to the original length.
    for seed in range(50):
        noisy = add_typos(ENGLISH, "en", 1.0, random.Random(seed))
        assert abs(len(noisy) - len(ENGLISH)) <= len(ENGLISH.split())
