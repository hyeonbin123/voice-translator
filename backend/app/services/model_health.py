"""모델 호출이 연이어 실패하는지 세고, CUDA 문맥을 망가뜨리는 오류가 나면 프로세스를 끝낸다.

GPU 오류 하나(드라이버 재설정, 같은 GPU를 쓰는 다른 프로그램과의 메모리 다툼)로 모델 호출이 계속 실패해도,
상태 확인이 DB만 보면 서버는 정상으로 보이고 사람이 컨테이너를 다시 시작할 때까지 번역이 멈춘다. 그래서 두
가지를 한다.

1. 모델 실행기(services/inference.py의 run_model·run_live_model·run_synthesis)를 지나는 호출의 결과를 보고,
   모델 엔진 오류로 실패한 호출이 연이어 몇 번인지 센다. 모델 호출이 한 번 성공하면 0으로 돌아간다.
   /api/health(routers/health.py)가 이 수를 MODEL_FAILURE_LIMIT와 견줘 503을 낸다.
2. 원인 사슬에 CUDA 런타임 문서가 '프로세스를 끝내고 다시 띄워야 CUDA를 다시 쓸 수 있다'고 적은 오류가
   있으면, 그 프로세스의 GPU 호출은 모두 같은 오류로 실패하므로 종류만 로그에 남기고 종료 코드 1로 바로
   끝낸다. Docker compose의 api `restart: on-failure:3`이 새 프로세스를 띄운다(docs/api.md '상태 확인').
"""

# [파일 역할] 모델 호출 결과를 분류하는 함수 둘(is_model_failure, sticky_cuda_error)과, 프로세스 하나에 하나씩
# 두는 상태 객체(ModelHealth, MODEL_HEALTH)를 정의한다.
# [설계 이유] 회복을 사람 손 대신 '죽고 다시 뜨기'(crash-only)에 맡긴다. 망가진 CUDA 문맥은 같은 프로세스
# 안에서 되살릴 방법이 없고(아래 STICKY_CUDA_ERRORS의 근거), 비스웜 Docker는 상태 검사가 unhealthy인
# 컨테이너를 다시 시작하지 않으므로, 다시 띄우려면 프로세스가 0이 아닌 코드로 끝나야 restart 정책이 움직인다.
# 메모리 부족처럼 문맥이 살아 있는 실패는 끝내지 않고 세기만 한다: 다른 프로그램이 GPU 메모리를 놓으면 다음
# 호출이 성공해 저절로 0으로 돌아오기 때문이다(요청은 그동안에도 모델로 간다 — 호출을 막는 차단기는 없다).
# [연결] services/inference.py의 _run이 호출 하나가 끝날 때마다 record_success·record_failure를 부르고,
# routers/health.py가 consecutive_failures를 읽는다. 테스트는 MODEL_HEALTH를 종료 함수가 다른 새 객체로
# 바꿔 끼운다(tests/test_model_health.py, tests/test_health.py).

import contextlib
import logging
import os
import sys
import threading
from collections.abc import Callable, Iterator

from app.services.errors import describe
from app.services.interfaces import InvalidAudioError, ModelError

logger = logging.getLogger(__name__)

# 끈적이는(sticky) CUDA 오류: cudaGetErrorString이 내는 글 → cudaError 이름(로그에 남기는 값).
# 근거: CUDA Runtime API 문서의 cudaError 설명(https://docs.nvidia.com/cuda/cuda-runtime-api/
# group__CUDART__TYPES.html). 처음 여덟은 문서가 '프로세스를 일관되지 않은 상태로 남겨 이후의 모든
# CUDA 작업이 같은 오류를 내고, 계속 쓰려면 프로세스를 끝내고 다시 띄워야 한다'고 설명한 오류이고,
# cudaErrorAssert(커널 안의 assert)도 같은 문서가 프로세스를 다시 시작해야 CUDA를 다시 쓸 수 있다고
# 적는다. 글은 서버 이미지의 libcudart.so.12에 든 그대로다(2026-10-30 확인). torch는 "CUDA error:
# <글>"(torch 2.8부터는 RuntimeError의 하위 클래스 torch.AcceleratorError), CTranslate2는 "CUDA failed with
# error <글>"(RuntimeError) 모양으로 이 글을 담는다(서버 이미지의 libc10_cuda.so·libctranslate2에서 확인).
# 넣지 않은 것: 문서 목록 중 cudaErrorContained·cudaErrorTensorMemoryLeak·cudaErrorExternalDevice는 새
# 오류라 이 이미지에서 글을 확인하지 않았다. cudaErrorECCUncorrectable("uncorrectable ECC error
# encountered")은 문서가 프로세스 재시작을 요구하지 않는다. 메모리 부족("out of memory")은 문맥이 살아
# 있다. 이 셋은 세기만 한다.
# 주의: torch는 모든 CUDA 오류 끝에 "Compile with `TORCH_USE_CUDA_DSA` to enable device-side
# assertions."를 붙이므로, 'device-side assert'만 찾으면 메모리 부족까지 문맥 오류로 오판한다. 그래서
# cudaErrorAssert는 'device-side assert triggered' 전체로 찾는다.
STICKY_CUDA_ERRORS: dict[str, str] = {
    "an illegal memory access was encountered": "cudaErrorIllegalAddress",
    "the launch timed out and was terminated": "cudaErrorLaunchTimeout",
    "hardware stack error": "cudaErrorHardwareStackError",
    "an illegal instruction was encountered": "cudaErrorIllegalInstruction",
    "misaligned address": "cudaErrorMisalignedAddress",
    "operation not supported on global/shared address space": "cudaErrorInvalidAddressSpace",
    "invalid program counter": "cudaErrorInvalidPc",
    "unspecified launch failure": "cudaErrorLaunchFailure",
    "device-side assert triggered": "cudaErrorAssert",
}


# 원인 사슬을 바깥에서 안쪽으로 하나씩 돌려준다. 규칙은 services/errors.describe(로그의 종류 사슬)와 같다:
# `raise ... from`으로 단 원인(__cause__)을 먼저, 없으면 처리 중에 생긴 예외(__context__)를 따라가되 `from
# None`으로 숨긴 문맥은 따라가지 않는다. seen은 서로를 가리키는 사슬에서 무한 반복을 막는다.
def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)


def sticky_cuda_error(exc: BaseException) -> str | None:
    """원인 사슬에 프로세스를 다시 띄워야 하는 CUDA 오류가 있으면 그 cudaError 이름을, 없으면 None을 준다.

    사슬의 예외 중 RuntimeError(와 그 하위 클래스)이면서 메시지에 'cuda'와 STICKY_CUDA_ERRORS의 글이 함께
    있는 것만 본다(대소문자 무시). 두 조건은 오판을 줄이려는 것이다: 확인한 두 엔진(torch·CTranslate2)의 CUDA
    오류는 둘 다 'CUDA'로 시작하는 RuntimeError라 이 조건에 맞고, 입력 글을 인용하는 다른 종류의
    예외(ValueError·KeyError 등)에 사용자가 그 문구를 써 넣어 서버를 끝내는 일은 막는다. 완전한 방어는
    아니다: RuntimeError 메시지가 입력 글을 그대로 담고 그 글에 'CUDA'와 문구가 함께 있으면 끝난다(그래도
    다시 시작은 restart 정책의 3번으로 묶인다). 메시지를 읽다(str) 예외가 나면 그 고리는 건너뛴다.
    """
    for link in _chain(exc):
        if not isinstance(link, RuntimeError):
            continue
        try:
            message = str(link).lower()
        except Exception:  # noqa: BLE001 - 망가진 __str__ 때문에 분류가 실패하면 안 된다(그 고리만 건너뜀)
            continue
        if "cuda" not in message:
            continue
        for text, name in STICKY_CUDA_ERRORS.items():
            if text in message:
                return name
    return None


def is_model_failure(exc: BaseException) -> bool:
    """연속 실패로 세는 실패인지: 모델 엔진의 예외를 원인으로 단 ModelError만 센다.

    각 어댑터의 예외 경계(stt._recognize·detect_language, translation._guarded, tts._guarded)가 엔진
    예외를 `raise ModelError(...) from exc`로 바꾸므로 원인(__cause__)이 있다. 세지 않는 것: 입력 검사의
    ModelError(빈 글, 같은 언어 쌍, 지원하지 않는 언어 — 원인 없음), 빈 번역·빈 음성(원인 없음), 사용자 음성의
    문제인 InvalidAudioError 갈래(디코딩 실패는 FFmpeg 예외를 원인으로 달아도 입력 탓이다), ModelError가 아닌
    예외(30초 넘는 음성 등). 이것들은 모델이 고장 나지 않아도 입력에 따라 생긴다.
    """
    from_engine = exc.__cause__ is not None
    return isinstance(exc, ModelError) and not isinstance(exc, InvalidAudioError) and from_engine


# 종료 직전에 표준 출력·오류를 비운다. os._exit는 버퍼를 비우지 않고 끝나기 때문이다(로그 처리기는 줄마다
# 비우지만, 다른 출력이 남아 있을 수 있다). 스트림이 없거나(None) 닫혔으면 그냥 넘어간다: 끝내는 길에서 이
# 실패로 멈추면 안 된다.
def _flush_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, ValueError, AttributeError):
            stream.flush()


class ModelHealth:
    """프로세스의 모델 호출 상태: 엔진 오류로 연이어 실패한 호출 수와, 끈적이는 CUDA 오류 때의 종료.

    여러 실행기 스레드(모델·자막·합성)가 동시에 기록하므로 잠금으로 수를 지킨다. '연이어'는 호출이 끝난
    순서를 말한다. 수는 모든 모델(인식·번역·합성)에 하나다: 한 모델만 고장 나고 다른 모델 호출이 사이사이
    성공하면 쌓이지 않는다. 끈적이는 CUDA 오류는 프로세스의 모든 GPU 모델을 함께 멈추므로 이 수로 충분하다.
    `exit_process`는 종료 함수(기본 os._exit)이고, 테스트가 기록용 함수로 바꿔 넣는다.
    """

    def __init__(self, exit_process: Callable[[int], object] = os._exit) -> None:
        self.exit_process = exit_process
        self._lock = threading.Lock()
        self._failures = 0

    @property
    def consecutive_failures(self) -> int:
        """마지막 모델 성공 뒤 엔진 오류로 실패한 호출 수(서버 시작 때 0)."""
        with self._lock:
            return self._failures

    def record_success(self) -> None:
        """모델 호출 하나가 결과를 돌려줬다: 연속 실패 수를 0으로 되돌린다."""
        with self._lock:
            self._failures = 0

    def record_failure(self, exc: BaseException) -> None:
        """모델 실행기의 호출 하나가 `exc`로 실패했다.

        엔진 오류(is_model_failure)이거나 끈적이는 CUDA 오류면 연속 실패 수를 하나 늘리고, 그 밖의 실패(입력
        탓)는 무시한다. 끈적이는 CUDA 오류면 cudaError 이름과 예외 종류 사슬(describe — 메시지 없이 종류와
        가장 안쪽 위치)만 로그에 남기고 exit_process(1)을 부른다. 기본 os._exit는 정리 없이 곧바로 끝내므로
        처리 중이던 다른 요청·동시통역 마디도 버려진다(정상 종료를 기다리면 망가진 GPU 호출에 걸려 멈출 수
        있다). 바꿔 넣은 종료 함수가 돌아오면 호출한 쪽이 원래 예외를 그대로 올린다.
        """
        sticky = sticky_cuda_error(exc)
        if sticky is None and not is_model_failure(exc):
            return
        with self._lock:
            self._failures += 1
        if sticky is not None:
            logger.critical(
                "A model call hit %s, which leaves this process's CUDA context unusable; exiting with "
                "code 1 so the container restart policy starts a new process: %s",
                sticky,
                describe(exc),
            )
            _flush_output()
            self.exit_process(1)


# 프로세스에 하나뿐인 상태 객체. 실행기와 상태 확인은 부를 때마다 이 모듈 속성을 읽으므로(이름을 미리 가져가
# 두지 않음), 테스트가 monkeypatch로 통째로 바꿔 끼울 수 있다.
MODEL_HEALTH = ModelHealth()
