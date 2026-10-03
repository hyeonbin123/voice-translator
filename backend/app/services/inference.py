"""Worker threads for the model calls (speech recognition, translation, synthesis).

A model call holds the CPU or GPU for hundreds of milliseconds. Made directly inside an
async handler, it would stall every other request the server is handling for that long
(measured in the earlier rag-doc-qa project), so model calls run on a bounded pool of
worker threads instead.

The pool size is the MODEL_THREADS setting (app/config.py), 1 by default: the models share
one GPU, and passes running side by side mostly compete for it. T11 measured 1 and 2 with
the chosen models and kept 1 (docs/experiments.md 4).
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import TypeVar

from app.config import get_settings

T = TypeVar("T")


@lru_cache
def _model_threads() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=get_settings().model_threads, thread_name_prefix="model")


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


@lru_cache
def _synthesis_thread() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="synthesis")


async def run_synthesis(fn: Callable[..., T], *args) -> T:
    """Run a speech synthesis that uses the CPU only (Supertonic, T78) on a thread of its own.

    Not on the model thread: the GPU models need not wait for it, nor it for them. One synthesis at a time,
    since one call already spreads over SUPERTONIC_THREADS cores.
    """
    return await _run(_synthesis_thread(), fn, *args)


@lru_cache
def _correction_thread() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="correction")


async def run_correction(fn: Callable[..., T], *args) -> T:
    """Run a typo correction (a blocking HTTP call to Ollama, T32) on a thread of its own.

    Not on a model thread, so a slow Ollama does not hold up speech, dialog and live work, and not on the
    default executor: one correction at a time, as when it shared the model thread, keeps concurrent typed
    requests from stacking Ollama's work on the GPU the models use. A request cancelled while its
    correction runs cannot stop the thread, so the next correction still waits for it to end.
    """
    return await _run(_correction_thread(), fn, *args)
