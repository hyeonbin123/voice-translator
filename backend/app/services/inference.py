"""Worker threads for the model calls (speech recognition, translation, synthesis).

A model call holds the CPU or GPU for hundreds of milliseconds. Made directly inside an
async handler, it would stall every other request the server is handling for that long
(measured in the earlier rag-doc-qa project), so model calls run on a bounded pool of
worker threads instead.

The pool size is the MODEL_THREADS setting (app/config.py), 1 by default: the models share
one GPU, and passes running side by side mostly compete for it. T11 measured 1 and 2 with
the chosen models and kept 1 (docs/experiments.md 4).
"""

# [파일 역할] 모델 호출을 이벤트 루프 밖의 전용 스레드 풀에서 돌리는 도우미. 용도별로 네 가지 실행기를 둔다:
# 모델 스레드(인식·번역·합성 공용), 동시통역 자막 전용, CPU 합성(Supertonic) 전용, 오타 교정(Ollama HTTP)
# 전용.
# [설계 이유] 위 docstring 그대로, 비동기 핸들러 안에서 모델을 직접 부르면 그동안 서버의 다른 요청이 모두
# 멈춘다(위 docstring: 앞선 rag-doc-qa 프로젝트에서 잰 문제). 용도별로 실행기를 나눈 것은 서로 다른 자원을
# 쓰는 작업이 한 대기열에서 서로를 기다리지 않게 하려는 것이다.
# [연결] services/pipeline.py, routers/translate.py, routers/live.py가 run_model 등을 await한다.

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import TypeVar

from app.config import get_settings

# T: 넘긴 함수의 반환 타입을 run_* 의 반환 타입으로 그대로 이어 주는 타입 변수.
T = TypeVar("T")


# 모델 스레드 풀. lru_cache로 처음 쓸 때 한 번만 만들어 프로세스 내내 공유한다(모듈을 불러올 때 만들지 않아
# 테스트가 설정을 바꿀 수 있다: tests/test_model_loading.py가 get_settings를 바꾼다). 크기는
# MODEL_THREADS(기본 1).
@lru_cache
def _model_threads() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=get_settings().model_threads, thread_name_prefix="model")


# 모든 run_* 의 공통 구현. fn(*args)를 실행기에서 돌리고 결과를 await한다.
# 취소 처리: 이미 대기열에 들어간 작업은 asyncio 쪽에서 취소해도 실행기 대기열에서 빠지지 않는다. 그래서
# 인수를 담은 partial을 리스트 한 칸(job)에 넣어 두고, 취소되면 그 칸을 비워 음성 같은 큰 인수를 바로
# 놓아준다. 나중에 스레드가 그 작업에 닿으면 빈 칸을 보고 아무것도 하지 않는다(위 영어 주석: 동시통역의
# pause/resume가 반복되면 취소된 작업이 마디 전체 음성을 대기열에 붙잡고 있었다).
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


# 인식·번역·GPU 합성·음성 검사(validate_audio)·언어 판별이 이 대기열을 함께 쓴다.
async def run_model(fn: Callable[..., T], *args) -> T:
    """Run `fn(*args)` on a model thread, queued if all of them are busy."""
    return await _run(_model_threads(), fn, *args)


# 동시통역 자막 갱신 전용 스레드 하나. LIVE_UPDATE_THREAD=true일 때만 쓰인다(routers/live.py).
@lru_cache
def _live_thread() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="live")


async def run_live_model(fn: Callable[..., T], *args) -> T:
    """Run a live subtitle update on a thread of its own, so it does not queue behind finals and other
    requests (LIVE_UPDATE_THREAD, docs/experiments.md 8, T61)."""
    return await _run(_live_thread(), fn, *args)


# CPU 합성(Supertonic) 전용 스레드 하나. KO_TTS=supertonic일 때 한국어 합성만 여기로
# 간다(services/pipeline.prepare가 synthesis_off_model_thread로 고름).
@lru_cache
def _synthesis_thread() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="synthesis")


async def run_synthesis(fn: Callable[..., T], *args) -> T:
    """Run a speech synthesis that uses the CPU only (Supertonic, T78) on a thread of its own.

    Not on the model thread: the GPU models need not wait for it, nor it for them. One synthesis at a time,
    since one call already spreads over SUPERTONIC_THREADS cores.
    """
    return await _run(_synthesis_thread(), fn, *args)


# 오타 교정 전용 스레드 하나. 기본 실행기(asyncio.to_thread)를 쓰지 않는 이유는 아래 docstring대로 교정을 한
# 번에 하나로 묶기 위해서다.
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
