# [파일 역할] app/services/errors.py의 describe를 검증한다. describe는 실패를 로그에 남길 때 예외 메시지 없이
# '예외 종류의 사슬 + 가장 안쪽 예외가 난 위치'만 한 줄로 만든다.
# [왜 중요한가] 모델 라이브러리의 오류 메시지에는 사용자가 넣은 문장이 그대로 들어갈 수 있다. 메시지가 로그에
# 새지 않는다는 것과, 그래도 어느 단계에서 왜 실패했는지는 알 수 있다는 것을 함께 고정한다.
# [방법] 모든 예외 메시지에 "secret"을 넣고, 결과 문자열에 그 단어가 없는지 본다.

from app.services.errors import describe


# KeyError → ValueError(처리 중 생긴 예외, __context__로 연결) → RuntimeError(raise ... from, __cause__로
# 연결)의 3단 사슬을 만든다. 두 가지 연결 방식을 모두 따라가는지 보려는 것이다. 가운데 줄의 B904 경고 끄기
# 표시는 암묵적 문맥(__context__)이 바로 시험 대상이라서 일부러 둔 것이다.
def raise_chain():
    try:
        try:
            raise KeyError("secret one")
        except KeyError:
            raise ValueError("secret two")  # noqa: B904 - the implicit context is the point
    except ValueError as exc:
        raise RuntimeError("secret three") from exc


# 바깥에서 안쪽 순서로 예외 이름이 " <- "로 이어지고, 위치는 가장 안쪽 예외(KeyError)가 난 파일
# 이름·줄·함수(raise_chain)를 가리키며, 메시지는 하나도 들어가지 않는지 확인한다.
def test_describe_names_the_chain_and_the_place_but_no_message():
    try:
        raise_chain()
    except RuntimeError as exc:
        text = describe(exc)
    assert text.startswith("RuntimeError <- ValueError <- KeyError (at test_errors.py:")
    assert text.endswith("in raise_chain)")
    assert "secret" not in text


# `raise ... from None`으로 숨긴 문맥(__suppress_context__)은 따라가지 않는지 확인한다. 개발자가 의도적으로
# 감춘 원인까지 로그에 끌어내지 않기 위해서다.
def test_describe_stops_at_a_suppressed_context():
    try:
        try:
            raise KeyError("secret")
        except KeyError:
            raise ValueError("secret") from None
    except ValueError as exc:
        assert describe(exc).startswith("ValueError (at")


# 서로를 원인으로 가리키는 순환 사슬에서도 무한 반복 없이 끝나고(describe의 seen 집합), raise하지 않고 만든
# 예외처럼 traceback이 없을 때는 위치 부분을 빼고 이름만 돌려주는지 확인한다.
def test_describe_survives_a_cycle_and_a_missing_traceback():
    first, second = ValueError("secret"), RuntimeError("secret")
    first.__cause__, second.__cause__ = second, first
    assert describe(first) == "ValueError <- RuntimeError"
