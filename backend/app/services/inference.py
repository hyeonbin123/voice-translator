"""Worker threads for the model calls (speech recognition, translation, synthesis).

A model call holds the CPU or GPU for hundreds of milliseconds. Made directly inside an
async handler, it would stall every other request the server is handling for that long
(measured in the earlier rag-doc-qa project), so model calls run on a bounded pool of
worker threads instead.

The pool size comes from MODEL_THREADS and defaults to 1: the models share one GPU, and
passes running side by side mostly compete for it. The real value is measured with the
chosen models in task T11.
"""

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import TypeVar

T = TypeVar("T")


@lru_cache
def _model_threads() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=int(os.environ.get("MODEL_THREADS", "1")), thread_name_prefix="model"
    )


async def _run(executor: ThreadPoolExecutor, fn: Callable[..., T], *args) -> T:
    # A cancelled call stays in the executor's queue until a thread reaches it and skips it. Let go of its
    # arguments now: a live pause/resume would otherwise pin a whole utterance's audio until then.
    job: list[Callable[[], T] | None] = [partial(fn, *args)]

    def call() -> T | None:
        queued = job[0]
        return None if queued is None else queued()

    try:
        return await asyncio.get_running_loop().run_in_executor(executor, call)
    except asyncio.CancelledError:
        job[0] = None
        raise


async def run_model(fn: Callable[..., T], *args) -> T:
    """Run `fn(*args)` on a model thread, queued if all of them are busy."""
    return await _run(_model_threads(), fn, *args)


@lru_cache
def _live_thread() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="live")


async def run_live_model(fn: Callable[..., T], *args) -> T:
    """Run a live subtitle update on a thread of its own, so it does not queue behind finals and other
    requests (LIVE_UPDATE_THREAD, docs/experiments.md 8, T61)."""
    return await _run(_live_thread(), fn, *args)
