from app.services.errors import describe


def raise_chain():
    try:
        try:
            raise KeyError("secret one")
        except KeyError:
            raise ValueError("secret two")  # noqa: B904 - the implicit context is the point
    except ValueError as exc:
        raise RuntimeError("secret three") from exc


def test_describe_names_the_chain_and_the_place_but_no_message():
    try:
        raise_chain()
    except RuntimeError as exc:
        text = describe(exc)
    assert text.startswith("RuntimeError <- ValueError <- KeyError (at test_errors.py:")
    assert text.endswith("in raise_chain)")
    assert "secret" not in text


def test_describe_stops_at_a_suppressed_context():
    try:
        try:
            raise KeyError("secret")
        except KeyError:
            raise ValueError("secret") from None
    except ValueError as exc:
        assert describe(exc).startswith("ValueError (at")


def test_describe_survives_a_cycle_and_a_missing_traceback():
    first, second = ValueError("secret"), RuntimeError("secret")
    first.__cause__, second.__cause__ = second, first
    assert describe(first) == "ValueError <- RuntimeError"
