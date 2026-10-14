"""라우터 실패 로그 말고 나머지 로그 경로에서도 예외 메시지가 로그에 남지 않는지 검증한다."""

# [파일 역할] 라우터가 describe로 직접 남기는 실패 로그 말고도, 나머지 로그 경로에서 예외 메시지와 쿼리 값이
# 새지 않는지 확인한다(커밋 c49d841).
# [왜 필요한가] 모델 라이브러리나 DB 드라이버의 오류 메시지에는 사용자가 말하거나 입력한 원문·번역문이 그대로
# 들어갈 수 있다. 로그에는 예외 종류와 위치만 남긴다는 원칙을 uvicorn의 예상 밖 오류 로그, exc_info로 남긴 앱
# 로그, SQLAlchemy 오류 문자열까지 넓혀 고정한다.
# [방법] 예외 메시지와 쿼리 값에 표식 문자열(MARK)을 넣고, 로그·오류 문자열에 그 표식이 없는지 본다.

import logging

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

# app.main을 불러오는 것만으로 로그 필터(WithoutExceptionMessages)가 'app' 로거의 핸들러와 'uvicorn.error'
# 로거에 붙는다. ENGINE_OPTIONS·engine은 앱이 실제로 쓰는 DB 엔진 설정이다.
from app import main
from app.db.session import ENGINE_OPTIONS, engine

# 사용자 원문을 대신하는 표식. 로그나 오류 문자열에 이 글자가 보이면 원문이 샌 것이다.
MARK = "원문표식"


# 라이브러리가 입력 문장을 메시지에 인용하며 실패하는 상황을 흉내 낸다.
def raise_quoting_error():
    raise ValueError(f"cannot handle {MARK}")


# 어떤 핸들러도 답하지 않은 예외는 Starlette가 다시 던지고 uvicorn이 'uvicorn.error' 로거로 traceback과 함께
# 남긴다. 그 로그 줄이 'Exception in ASGI application: ValueError (at 파일:줄 ...)' 요약으로 바뀌고
# 메시지·Traceback이 없는지 확인한다.
def test_an_unexpected_error_logged_by_uvicorn_keeps_its_type_not_its_message(caplog):
    # 어떤 핸들러도 답하지 않은 오류를 Starlette가 다시 던질 때 uvicorn이 하는 일을 그대로 흉내 낸다.
    try:
        raise_quoting_error()
    except ValueError:
        logging.getLogger("uvicorn.error").exception("Exception in ASGI application")
    assert "Exception in ASGI application: ValueError (at test_log_privacy.py:" in caplog.text
    assert MARK not in caplog.text and "Traceback" not in caplog.text


# app.* 하위 로거에서 exc_info=True로 남긴 로그도 예외 종류만 남는지 확인한다. 먼저 'app' 로거의 핸들러에
# 필터가 붙어 있는지 본다. 필터를 로거가 아니라 핸들러에 두어야 app.services.somewhere 같은 하위 로거의
# 레코드까지 걸러지기 때문이다(main.py 주석).
# 핸들러 필터가 레코드 객체 자체를 고치므로, 그 뒤 루트로 전파된 레코드를 받는 caplog도 고쳐진 문장을 본다.
def test_an_app_record_with_exc_info_keeps_its_type_not_its_message(caplog):
    assert any(
        isinstance(f, main.WithoutExceptionMessages) for h in main._app_logger.handlers for f in h.filters
    )
    try:
        raise_quoting_error()
    except ValueError:
        logging.getLogger("app.services.somewhere").warning("Something failed", exc_info=True)
    assert "Something failed: ValueError" in caplog.text and MARK not in caplog.text


# 앱의 DB 엔진이 hide_parameters=True로 만들어졌는지 확인한다. 이 옵션이 없으면 SQLAlchemy 오류 문자열에
# 바인딩된 값(원문·번역문 등)이 그대로 찍힌다.
def test_the_engine_leaves_bound_values_out_of_errors():
    assert ENGINE_OPTIONS["hide_parameters"] is True
    assert engine.sync_engine.hide_parameters


# 실제 테스트 DB에 없는 테이블을 조회해 오류를 일으키고, 오류 문자열에 바인딩 값(MARK)이 없고 'parameters
# hidden' 표시만 있는지 확인한다. 옵션 값만 보는 위 테스트와 달리 실제 동작을 확인한다. db_engine fixture도
# 같은 ENGINE_OPTIONS로 만들어진다(conftest.py).
async def test_a_database_error_does_not_list_the_values(db_engine):
    async with db_engine.connect() as connection:
        with pytest.raises(DBAPIError) as caught:
            await connection.execute(text("select :value from no_such_table"), {"value": MARK})
    assert MARK not in str(caught.value)
    assert "parameters hidden" in str(caught.value)
