from app.services.live import common_start, letters, stable_length


def test_letters_drop_case_spaces_and_punctuation():
    assert letters("Soon, officers  entered the yard.") == "soonofficersenteredtheyard"
    assert letters("지구 온난화의 영향을!") == "지구온난화의영향을"
    assert letters("") == ""


def test_common_start():
    assert common_start("abcd", "abxd") == 2
    assert common_start("abc", "abc") == 3
    assert common_start("", "abc") == 0


def test_nothing_is_dark_without_a_previous_result():
    assert stable_length(None, "The UN also hopes") == 0


def test_the_agreed_whole_words_are_dark():
    current = "The UN also hopes to finalize"
    assert current[: stable_length("The UN also hopes to fin", current)] == "The UN also hopes to"
    assert stable_length(current, current) == len(current)


def test_a_word_the_results_disagree_on_ends_the_dark_part():
    current = "Soon officers equipped with riot gear"
    assert current[: stable_length("Soon, officers quipped", current)] == "Soon officers"


def test_case_punctuation_and_korean_spacing_do_not_count_as_changes():
    assert stable_length("soon officers.", "Soon, officers") == len("Soon, officers")
    current = "지구온난화의 영향을 받는"
    assert current[: stable_length("지구 온난화의 영향", current)] == "지구온난화의"


def test_a_different_start_leaves_nothing_dark():
    assert stable_length("Argentina is", "Argentine was") == 0
    assert stable_length("anything", "") == 0
