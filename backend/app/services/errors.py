"""Describe a failure for the log without the messages of the exceptions behind it (T49).

A model library's message can quote the text it was given, and a traceback prints the whole chain of
causes. Logs therefore name only the exception types along the chain and where the innermost one was
raised, which is enough to tell which stage failed and why.
"""

import traceback
from pathlib import Path


def describe(exc: BaseException) -> str:
    """For example "ModelError <- ValueError (at sentencepiece.py:123 in encode)". Never the messages."""
    names: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    innermost = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.append(type(current).__name__)
        innermost = current
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    frames = traceback.extract_tb(innermost.__traceback__)
    where = (
        f" (at {Path(frames[-1].filename).name}:{frames[-1].lineno} in {frames[-1].name})" if frames else ""
    )
    return " <- ".join(names) + where
