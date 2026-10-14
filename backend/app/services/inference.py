"""모델 호출(음성 인식, 번역, 합성)을 위한 워커 스레드.

모델 호출 하나는 CPU나 GPU를 수백 밀리초 동안 쥔다. 비동기 핸들러 안에서 직접 부르면 서버가 처리 중인
다른 모든 요청이 그동안 멈추므로(앞선 rag-doc-qa 프로젝트에서 잰 문제), 모델 호출은 크기가 정해진 워커
스레드 풀에서 돈다.

풀 크기는 MODEL_THREADS 설정(app/config.py)이고 기본은 1이다. 모델들이 GPU 하나를 나눠 쓰므로 나란히 도는
처리는 대부분 GPU를 두고 다툴 뿐이다. 고른 모델로 1과 2를 재어 1을 유지했다(docs/experiments.md 4절).
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
# 취소 처리: 이미 대기열에 들어간 작업은 asyncio 쪽에서 취소해도, 스레드가 그 작업에 닿아 건너뛸 때까지
# 실행기 대기열에 남는다. 그래서 인수를 담은 partial을 리스트 한 칸(job)에 넣어 두고, 취소되면 그 칸을 바로
# 비워 음성 같은 큰 인수를 놓아준다. 나중에 스레드가 그 작업에 닿으면 빈 칸을 보고 아무것도 하지 않는다.
# 이렇게 하지 않으면 동시통역의 pause/resume가 반복될 때 취소된 작업이 마디 전체 음성을 그때까지 붙잡고 있게
# 된다.
async def _run(executor: ThreadPoolExecutor, fn: Callable[..., T], *args) -> T:
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
    """`fn(*args)`를 모델 스레드에서 실행한다. 스레드가 모두 바쁘면 대기열에서 기다린다."""
    return await _run(_model_threads(), fn, *args)


# 동시통역 자막 갱신 전용 스레드 하나. LIVE_UPDATE_THREAD=true일 때만 쓰인다(routers/live.py).
@lru_cache
def _live_thread() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="live")


async def run_live_model(fn: Callable[..., T], *args) -> T:
    """동시통역 자막 갱신을 전용 스레드에서 실행한다(LIVE_UPDATE_THREAD, docs/experiments.md 8절).

    최종 결과 처리와 다른 요청 뒤의 대기열에서 기다리지 않게 하려는 것이다.
    """
    return await _run(_live_thread(), fn, *args)


# CPU 합성(Supertonic) 전용 스레드 하나. KO_TTS=supertonic일 때 한국어 합성만 여기로
# 간다(services/pipeline.prepare가 synthesis_off_model_thread로 고름).
@lru_cache
def _synthesis_thread() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="synthesis")


async def run_synthesis(fn: Callable[..., T], *args) -> T:
    """CPU만 쓰는 음성 합성(Supertonic)을 전용 스레드에서 실행한다.

    모델 스레드에서 돌리지 않는다: GPU 모델이 이것을 기다릴 필요가 없고, 이것도 GPU 모델을 기다릴 필요가
    없다. 호출 하나가 이미 SUPERTONIC_THREADS개 코어에 퍼지므로 합성은 한 번에 하나씩 한다.
    """
    return await _run(_synthesis_thread(), fn, *args)


# 오타 교정 전용 스레드 하나. 기본 실행기(asyncio.to_thread)를 쓰지 않는 이유는 아래 docstring대로 교정을 한
# 번에 하나로 묶기 위해서다.
@lru_cache
def _correction_thread() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="correction")


async def run_correction(fn: Callable[..., T], *args) -> T:
    """오타 교정(Ollama에 보내는 막는 HTTP 호출)을 전용 스레드에서 실행한다.

    모델 스레드에서 돌리지 않으므로 Ollama가 느려도 음성·두 사람 대화·동시통역 작업이 밀리지 않는다. 기본
    실행기에서도 돌리지 않는다: 모델 스레드를 함께 쓰던 때처럼 교정을 한 번에 하나씩 해야, 동시에 들어온
    글자 요청들이 모델이 쓰는 GPU에 Ollama 작업을 겹겹이 쌓지 않는다. 교정이 도는 중에 요청이 취소돼도
    스레드를 멈출 수는 없으므로, 다음 교정은 그 교정이 끝나기를 기다린다.
    """
    return await _run(_correction_thread(), fn, *args)
