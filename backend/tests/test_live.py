# [파일 역할] 동시통역 자막에서 '진하게 보일 앞부분'을 정하는 순수 함수(app/services/live.py의
# letters·common_start·stable_length)를 검증한다(docs/experiments.md 8절).
# [규칙] 서버는 1초마다 지금까지의 마디 전체를 다시 인식하고, 이웃한 두 결과가 글자 기준으로 같은 앞부분 중
# '완전한 단어'까지만 진하게 보인다. 대소문자·띄어쓰기(특히 한국어)·문장부호 차이는 말한 내용이 바뀐 것이
# 아니므로 무시한다.
# [왜 중요한가] 같은 함수를 측정 도구(eval/stream_eval.py)도 쓰므로, 이 규칙이 바뀌면 화면 동작과 기록된 자막
# 흔들림 수치가 함께 달라진다.

from app.services.live import common_start, letters, stable_length


# 비교용 정규화: 영어는 소문자로 바꾸고 공백·문장부호를 빼며, 한국어도 띄어쓰기와 느낌표가 빠지는지, 빈
# 문자열은 그대로인지 확인한다.
def test_letters_drop_case_spaces_and_punctuation():
    assert letters("Soon, officers  entered the yard.") == "soonofficersenteredtheyard"
    assert letters("지구 온난화의 영향을!") == "지구온난화의영향을"
    assert letters("") == ""


# 두 문자열의 같은 앞부분 길이: 중간에 다르면 그 앞까지, 같으면 전체, 한쪽이 비면 0.
def test_common_start():
    assert common_start("abcd", "abxd") == 2
    assert common_start("abc", "abc") == 3
    assert common_start("", "abc") == 0


# 첫 결과는 비교할 직전 결과가 없으므로 아무것도 진하게 보이지 않는다. 한 번 들은 것만으로는 확정된 것처럼
# 보이지 않게 하는 규칙이다.
def test_nothing_is_dark_without_a_previous_result():
    assert stable_length(None, "The UN also hopes") == 0


# 직전 결과가 'fin'까지만 들었으면 지금 결과의 'finalize'는 앞 세 글자가 같아도 진하게 하지 않고 'to'까지만
# 진하게 한다. 단어 중간에서 진한 부분이 끊기지 않게 하는 것이다. 두 결과가 같으면 전부 진하다.
def test_the_agreed_whole_words_are_dark():
    current = "The UN also hopes to finalize"
    assert current[: stable_length("The UN also hopes to fin", current)] == "The UN also hopes to"
    assert stable_length(current, current) == len(current)


# 'quipped'와 'equipped'처럼 두 결과가 다르게 들은 단어에서 진한 부분이 끝난다. 쉼표 차이("Soon,")는 무시된다.
def test_a_word_the_results_disagree_on_ends_the_dark_part():
    current = "Soon officers equipped with riot gear"
    assert current[: stable_length("Soon, officers quipped", current)] == "Soon officers"


# 대소문자·마침표·쉼표만 다른 영어는 전체가 진하고, 한국어는 '지구 온난화의'와 '지구온난화의'처럼 띄어쓰기만
# 다르면 같은 말로 본다. Whisper가 다시 인식할 때 한국어 띄어쓰기가 자주 바뀌어도 자막이 흔들려 보이지 않게
# 하는 규칙이다(services/live.py 모듈 설명).
def test_case_punctuation_and_korean_spacing_do_not_count_as_changes():
    assert stable_length("soon officers.", "Soon, officers") == len("Soon, officers")
    current = "지구온난화의 영향을 받는"
    assert current[: stable_length("지구 온난화의 영향", current)] == "지구온난화의"


# 첫 단어부터 다르게 들었으면(Argentina/Argentine) 진한 부분이 없고, 지금 결과가 비어 있으면 0이다.
def test_a_different_start_leaves_nothing_dark():
    assert stable_length("Argentina is", "Argentine was") == 0
    assert stable_length("anything", "") == 0
