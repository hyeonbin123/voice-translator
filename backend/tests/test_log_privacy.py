"""T54: exception messages stay out of the log beyond the paths T49 covered."""

import logging

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app import main
from app.db.session import ENGINE_OPTIONS, engine

MARK = "원문표식"


def raise_quoting_error():
    raise ValueError(f"cannot handle {MARK}")


def test_an_unexpected_error_logged_by_uvicorn_keeps_its_type_not_its_message(caplog):
    # What uvicorn does when Starlette re-raises an error no handler answered.
    try:
        raise_quoting_error()
    except ValueError:
        logging.getLogger("uvicorn.error").exception("Exception in ASGI application")
    assert "Exception in ASGI application: ValueError (at test_log_privacy.py:" in caplog.text
    assert MARK not in caplog.text and "Traceback" not in caplog.text


def test_an_app_record_with_exc_info_keeps_its_type_not_its_message(caplog):
    assert any(
        isinstance(f, main.WithoutExceptionMessages) for h in main._app_logger.handlers for f in h.filters
    )
    try:
        raise_quoting_error()
    except ValueError:
        logging.getLogger("app.services.somewhere").warning("Something failed", exc_info=True)
    assert "Something failed: ValueError" in caplog.text and MARK not in caplog.text


def test_the_engine_leaves_bound_values_out_of_errors():
    assert ENGINE_OPTIONS["hide_parameters"] is True
    assert engine.sync_engine.hide_parameters


async def test_a_database_error_does_not_list_the_values(db_engine):
    # The test engine is built with ENGINE_OPTIONS, like the app's.
    async with db_engine.connect() as connection:
        with pytest.raises(DBAPIError) as caught:
            await connection.execute(text("select :value from no_such_table"), {"value": MARK})
    assert MARK not in str(caught.value)
    assert "parameters hidden" in str(caught.value)
