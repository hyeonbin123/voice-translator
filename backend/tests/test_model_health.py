# [파일 역할] 모델 호출의 연속 실패 세기와 CUDA 문맥 오류 때의 프로세스 종료(app/services/model_health.py)를
# 검증한다. 오류 분류 함수만 따로 보는 테스트, 모델 실행기 경계(app/services/inference.py의 run_*)를 거치는
# 테스트, 실제 음성 인식 어댑터(stt.WhisperSpeechToText)의 예외 경계와 번역 API까지 거치는 테스트로 나뉜다.
# [무엇을 확인하나] (1) 엔진 예외에서 온 ModelError만 실패로 세고, 입력 탓 오류(빈 글, 디코딩할 수 없는 음성,
# 말 없음)와 오타 교정 실행기의 실패는 세지 않는다 (2) 모델 호출이 한 번 성공하면 0으로 돌아가지만, 값을
# 돌려주지 않는 음성 검사(validate_audio)의 성공은 모델 성공으로 치지 않는다 (3) 원인 사슬에 프로세스를 다시
# 띄워야 하는 CUDA 오류가 있으면 종료 함수를 코드 1로 한 번 부르고, 메모리 부족은 세기만 한다 (4) 종료 로그에
# 예외 메시지가 들어가지 않는다.
# [왜 중요한가] 분류가 틀리면 멀쩡한 서버가 재시작을 되풀이하거나(종료 쪽 오판), 고장 난 서버가 상태 확인에서
# 정상으로 보인다(세기 쪽 오판). 실제 CUDA 오류는 GPU 없이 재현할 수 없어, torch·CTranslate2가 내는 메시지
# 모양을 그대로 흉내 낸 예외로 확인한다.
# [주의] 종료 함수의 기본값은 os._exit라 테스트 프로세스를 끝낸다. 아래 health 픽스처가 이 파일의 모든
# 테스트에서 프로세스 전체가 쓰는 상태 객체(model_health.MODEL_HEALTH)를, 종료 함수가 기록용 리스트(exits)인
# 새 객체로 바꾼다. 그래서 연속 실패 수도 테스트마다 0에서 시작한다.

import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.dependencies import get_audio_store, get_models
from app.main import app
from app.routers import health as health_router
from app.services import model_health, stt
from app.services.audio_store import AudioStore
from app.services.inference import run_correction, run_live_model, run_model, run_synthesis
from app.services.interfaces import ModelError, NoSpeechError, UndecodableAudioError
from app.services.model_health import ModelHealth, is_model_failure, sticky_cuda_error
from app.services.pipeline import PipelineModels, validate_audio
from tests.fakes import FakeTextToSpeech, FakeTranslator, silent_wav

# 예외 메시지 대신 쓰는 표식. 로그에 이 글자가 보이면 메시지가 샌 것이다(services/errors.py의 규칙).
SECRET = "secret-input-words"

# torch가 C10_CUDA_CHECK로 내는 CUDA 오류의 꼬리말. torch 2.11(서버 이미지)의 libc10_cuda.so에 이 글이
# 들어 있고, 끈적이지 않는 오류(아래 메모리 부족 등)에도 붙는다. 'device-side assertions'라는 말이 들어
# 있어, 'device-side assert'만 찾으면 모든 torch CUDA 오류를 문맥 오류로 오판한다.
TORCH_HINT = (
    "\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below "
    "might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\n"
    "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n"
)


# torch 2.8 이후 CUDA 오류의 종류(torch.AcceleratorError)를 흉내 낸다. 진짜와 같이 RuntimeError의 하위
# 클래스다(서버 이미지의 torch 2.11에서 MRO 확인). CI에는 torch가 없어 직접 만든다.
class AcceleratorError(RuntimeError):
    pass


# torch(C10_CUDA_CHECK)가 내는 모양: "CUDA error: <cudaGetErrorString 글>" + 꼬리말.
def torch_error(text: str, kind: type[RuntimeError] = RuntimeError) -> RuntimeError:
    return kind(f"CUDA error: {text}{TORCH_HINT}")


# CTranslate2(faster-whisper·opus 번역)가 내는 모양: "CUDA failed with error <cudaGetErrorString 글>".
# 서버 이미지의 libctranslate2(4.8.2)에 이 앞말이 들어 있다.
def ct2_error(text: str) -> RuntimeError:
    return RuntimeError(f"CUDA failed with error {text}")


# 엔진 예외를 원인으로 단 ModelError를 만든다. 실제 어댑터의 예외 경계(_guarded, _recognize)가 `raise
# ModelError(...) from exc`로 만드는 모양과 같다. 메시지에는 예외 종류 이름만 넣는다(어댑터와 같음).
def engine_failure(cause: BaseException) -> ModelError:
    try:
        raise cause
    except BaseException as exc:  # noqa: BLE001 - 어떤 예외든 원인으로 다는 테스트 도우미
        try:
            raise ModelError(f"the model failed: {type(exc).__name__}") from exc
        except ModelError as wrapped:
            return wrapped


# 넘긴 예외를 그대로 내는 '모델 호출'. 실행기에 넣어 경계에서 어떻게 세는지 본다.
def raising(error: BaseException):
    def call():
        raise error

    return call


# 종료 함수가 받은 종료 코드를 모은다. 비어 있으면 종료하려 하지 않은 것이다.
@pytest.fixture
def exits() -> list[int]:
    return []


@pytest.fixture(autouse=True)
def health(monkeypatch, exits) -> ModelHealth:
    fresh = ModelHealth(exit_process=exits.append)
    monkeypatch.setattr(model_health, "MODEL_HEALTH", fresh)
    return fresh


# --- 오류 분류 ---------------------------------------------------------------------------------------


# torch("CUDA error: ...")와 CTranslate2("CUDA failed with error ...")가 내는 끈적이는 오류를, ModelError의
# 원인으로 달렸든 그대로든, torch.AcceleratorError 모양이든 cudaError 이름으로 알아보는지 확인한다.
@pytest.mark.parametrize(
    ("error", "name"),
    [
        (torch_error("an illegal memory access was encountered"), "cudaErrorIllegalAddress"),
        (torch_error("unspecified launch failure", AcceleratorError), "cudaErrorLaunchFailure"),
        (ct2_error("an illegal memory access was encountered"), "cudaErrorIllegalAddress"),
        (ct2_error("the launch timed out and was terminated"), "cudaErrorLaunchTimeout"),
        (torch_error("device-side assert triggered"), "cudaErrorAssert"),
        (torch_error("misaligned address"), "cudaErrorMisalignedAddress"),
        (torch_error("an illegal instruction was encountered"), "cudaErrorIllegalInstruction"),
    ],
    ids=["torch-700", "torch-accelerator-719", "ct2-700", "ct2-702", "torch-710", "torch-716", "torch-715"],
)
@pytest.mark.parametrize("wrapped", [True, False], ids=["model-error", "raw"])
def test_sticky_cuda_errors_are_recognised(error, name, wrapped):
    assert sticky_cuda_error(engine_failure(error) if wrapped else error) == name


# 메모리 부족은 문맥이 살아 있어(다음 호출은 될 수 있다) 문맥 오류가 아니다. torch의 두 모양(할당기의
# OutOfMemoryError, C10_CUDA_CHECK의 'CUDA error: out of memory' + 'device-side assertions' 꼬리말)과
# CTranslate2 모양 모두. 엔진 예외가 아닌 것(ValueError에 같은 글), 'CUDA'가 없는 글, 원인 없는 ModelError도
# 문맥 오류로 보지 않는다(사용자 글이 메시지에 섞여 프로세스를 끝내는 일을 막는 조건).
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 11 GiB"),
        torch_error("out of memory"),
        ct2_error("out of memory"),
        torch_error("invalid argument"),
        ValueError("CUDA error: an illegal memory access was encountered"),
        RuntimeError("an illegal memory access was encountered"),
        ModelError("CUDA error: an illegal memory access was encountered"),
    ],
    ids=["torch-oom", "torch-c10-oom", "ct2-oom", "torch-invalid", "not-runtime", "no-cuda", "no-engine"],
)
def test_other_errors_are_not_sticky(error):
    assert sticky_cuda_error(engine_failure(error)) is None
    assert sticky_cuda_error(error) is None


# 원인 사슬은 services/errors.describe와 같은 규칙으로 따라간다: `from`으로 단 원인을 먼저, 없으면 처리 중에
# 생긴 예외(__context__)를. `from None`으로 숨긴 문맥은 따라가지 않는다.
def test_the_cause_chain_follows_the_context_unless_it_is_suppressed():
    def raise_while_handling(suppress: bool) -> ModelError:
        try:
            try:
                raise RuntimeError("CUDA error: an illegal memory access was encountered")
            except RuntimeError:
                if suppress:
                    raise ModelError("the model returned no audio") from None
                raise ModelError("the model returned no audio")  # noqa: B904 - 문맥만 남는 경우를 만든다
        except ModelError as exc:
            return exc

    assert sticky_cuda_error(raise_while_handling(suppress=False)) == "cudaErrorIllegalAddress"
    assert sticky_cuda_error(raise_while_handling(suppress=True)) is None


# 실패로 세는 것은 엔진 예외에서 온 ModelError(원인이 달린 것)뿐이다. 입력 검사의 ModelError(원인 없음), 음성
# 문제(InvalidAudioError 갈래 — 디코딩 실패는 FFmpeg 예외를 원인으로 달아도), 감싸지 않은 예외는 세지 않는다.
def test_only_model_errors_from_an_engine_are_model_failures():
    assert is_model_failure(engine_failure(RuntimeError("CUDA out of memory")))
    assert not is_model_failure(ModelError("text is empty"))
    assert not is_model_failure(NoSpeechError("no speech was recognized"))
    decoding = UndecodableAudioError("the audio could not be decoded")
    decoding.__cause__ = OSError("ffmpeg failed")
    assert not is_model_failure(decoding)
    assert not is_model_failure(RuntimeError("CUDA out of memory"))


# --- 모델 실행기 경계 ----------------------------------------------------------------------------------


# 모델 실행기 셋(모델 스레드, 자막 스레드, 합성 스레드)의 엔진 실패가 모두 이어서 세어지고, 값을 돌려준 호출
# 하나가 0으로 되돌린다. 실패한 호출의 예외는 그대로 호출한 쪽으로 올라간다.
async def test_model_executors_count_failures_in_a_row_and_a_success_resets(health):
    for run in (run_model, run_live_model, run_synthesis, run_model):
        with pytest.raises(ModelError):
            await run(raising(engine_failure(RuntimeError(f"CUDA out of memory {SECRET}"))))
    assert health.consecutive_failures == 4
    assert await run_live_model(lambda: "translated words") == "translated words"
    assert health.consecutive_failures == 0


# 오타 교정 실행기(Ollama에 보내는 HTTP 호출, 다른 프로세스)의 실패는 이 프로세스의 모델 상태가 아니라 세지
# 않고, 그 성공도 0으로 되돌리지 않는다.
async def test_the_correction_executor_is_not_counted(health):
    with pytest.raises(ModelError):
        await run_model(raising(engine_failure(RuntimeError("CUDA out of memory"))))
    with pytest.raises(ModelError):
        await run_correction(raising(engine_failure(OSError("connection refused"))))
    assert await run_correction(lambda: "corrected") == "corrected"
    assert health.consecutive_failures == 1


# 입력 탓 실패는 세지 않는다: 입력 검사의 ModelError(빈 글), 디코딩할 수 없는 업로드(실제 validate_audio가
# FFmpeg 예외를 원인으로 단 UndecodableAudioError), 말 없음. 또 값을 돌려주지 않는 실제 음성 검사의 성공은
# 모델 성공이 아니어서, 음성 요청마다 검사가 먼저 돌아도 앞의 연속 실패가 지워지지 않는다.
async def test_input_errors_and_the_audio_check_leave_the_count_alone(health):
    with pytest.raises(ModelError):
        await run_model(raising(engine_failure(RuntimeError("CUDA out of memory"))))
    with pytest.raises(ModelError, match="text is empty"):
        await run_model(raising(ModelError("text is empty")))
    with pytest.raises(UndecodableAudioError):
        await run_model(validate_audio, b"not an audio file")
    with pytest.raises(NoSpeechError):
        await run_model(raising(NoSpeechError("no speech was recognized")))
    assert await run_model(validate_audio, silent_wav(200)) is None
    assert health.consecutive_failures == 1


# 호출한 쪽이 취소돼도(동시통역의 pause 등) 이미 스레드에서 돌던 호출의 결과는 센다: 기록은 스레드 안에서
# 한다. 아직 시작하지 않은 채 취소된 호출은 돌지 않으므로 세지 않는다.
async def test_a_failure_is_counted_even_when_the_caller_was_cancelled(health):
    started, release = threading.Event(), threading.Event()

    def slow_failure():
        started.set()
        release.wait(5)
        raise engine_failure(RuntimeError("CUDA out of memory"))

    running = asyncio.create_task(run_model(slow_failure))
    await asyncio.to_thread(started.wait, 5)
    queued = asyncio.create_task(run_model(raising(engine_failure(RuntimeError("CUDA out of memory")))))
    await asyncio.sleep(0)
    running.cancel()
    queued.cancel()
    await asyncio.gather(running, queued, return_exceptions=True)
    release.set()
    await run_model(lambda: None)  # 같은 스레드의 대기열이 빌 때까지 기다린다(None은 세지 않는다)
    assert health.consecutive_failures == 1


# 끈적이는 CUDA 오류는 종료 함수를 코드 1로 한 번 부르고, 실패로도 센다. 로그에는 cudaError 이름과 예외 종류
# 사슬만 남고 메시지는 없다. 종료 함수를 바꿔 둔 테스트에서는 원래 예외가 호출한 쪽으로 그대로 올라간다.
async def test_a_sticky_cuda_error_exits_the_process_once(health, exits, caplog):
    error = engine_failure(RuntimeError(f"CUDA error: an illegal memory access was encountered {SECRET}"))
    with caplog.at_level(logging.CRITICAL), pytest.raises(ModelError):
        await run_model(raising(error))
    assert exits == [1]
    assert health.consecutive_failures == 1
    assert "cudaErrorIllegalAddress" in caplog.text and "ModelError <- RuntimeError" in caplog.text
    assert SECRET not in caplog.text


# 메모리 부족(torch의 'device-side assertions' 꼬리말이 붙은 모양 포함)은 종료하지 않고 세기만 한다.
async def test_out_of_memory_is_counted_without_exiting(health, exits):
    allocator_oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    for error in (allocator_oom, torch_error("out of memory")):
        with pytest.raises(ModelError):
            await run_model(raising(engine_failure(error)))
    assert exits == []
    assert health.consecutive_failures == 2


# --- 실제 음성 인식 어댑터와 번역 API를 거쳐 ----------------------------------------------------------


# 실제 WhisperSpeechToText 안에 끼우는 가짜 faster-whisper. error를 주면 CTranslate2처럼 호출 순간 그 예외를
# 내고, 없으면 segments를 인식 결과로 돌려준다(빈 리스트면 말 없음).
class ScriptedWhisper:
    error: Exception | None = None
    segments: list[str] = ["recognized words"]

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, language, **options):
        if self.error is not None:
            raise self.error
        return (SimpleNamespace(text=text) for text in self.segments), SimpleNamespace(duration=1.0)


# 모델을 올린 서버를 흉내 낸다: 인식은 실제 어댑터 + 가짜 엔진, 번역·합성은 가짜, 상태 확인은
# LOAD_MODELS=true 설정(기본 MODEL_FAILURE_LIMIT 5)을 읽는다.
@pytest.fixture
def whisper_server(client, tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "WhisperModel", ScriptedWhisper)
    monkeypatch.setattr(stt, "add_cuda_dll_dirs", lambda: None)
    monkeypatch.setattr(ScriptedWhisper, "error", None)
    monkeypatch.setattr(ScriptedWhisper, "segments", ["recognized words"])
    whisper = stt.WhisperSpeechToText("tiny", device="cpu", compute_type="int8")
    models = PipelineModels(whisper, FakeTranslator(), FakeTextToSpeech())
    store = AudioStore(tmp_path / "audio")
    app.dependency_overrides[get_models] = lambda: models
    app.dependency_overrides[get_audio_store] = lambda: store
    settings = Settings(load_models=True)
    monkeypatch.setattr(health_router, "get_settings", lambda: settings)
    return ScriptedWhisper


async def speech(client, headers, content: bytes | None = None):
    return await client.post(
        "/api/translate/speech",
        headers=headers,
        files={"audio": ("clip.wav", silent_wav(200) if content is None else content, "audio/wav")},
        data={"source_lang": "en", "target_lang": "ko"},
    )


# 인식 엔진이 CUDA 문맥 오류를 내면(torch 모양) 종료 함수가 한 번 불린다. 종료 함수를 바꿔 둔 테스트에서는
# 요청이 평소의 모델 실패처럼 503으로 끝난다(실제 서버라면 프로세스가 끝나 연결이 끊긴다).
async def test_a_cuda_context_error_in_recognition_exits_once(client, auth_headers, whisper_server, exits):
    whisper_server.error = torch_error("an illegal memory access was encountered")
    response = await speech(client, auth_headers)
    assert response.status_code == 503
    assert exits == [1]


# 메모리 부족이 4번 이어지면 아직 200(4 < 5)이고, 인식이 한 번 성공하면 0으로 돌아간다. 다시 5번 이어지면
# 503·models_ok false가 되고, 성공 한 번이면 200으로 돌아온다. 음성 요청마다 음성 검사가 먼저 성공하지만
# 연속 실패를 지우지 않아야 5에 닿는다.
async def test_failures_in_a_row_make_health_503_until_a_success(client, auth_headers, whisper_server, exits):
    async def fail(times: int) -> None:
        whisper_server.error = ct2_error("out of memory")
        for _ in range(times):
            assert (await speech(client, auth_headers)).status_code == 503
        whisper_server.error = None

    async def health() -> tuple[int, dict]:
        response = await client.get("/api/health")
        return response.status_code, response.json()

    await fail(4)
    status, body = await health()
    assert status == 200 and body["models_ok"] is True and body["consecutive_model_failures"] == 4
    assert (await speech(client, auth_headers)).status_code == 201
    assert (await health())[1]["consecutive_model_failures"] == 0
    await fail(5)
    status, body = await health()
    assert status == 503 and body["status"] == "unavailable" and body["database"] == "ok"
    assert body["models_ok"] is False and body["consecutive_model_failures"] == 5
    assert (await speech(client, auth_headers)).status_code == 201
    status, body = await health()
    assert status == 200 and body["models_ok"] is True and body["consecutive_model_failures"] == 0
    assert exits == []


# 사용자의 입력 탓인 실패는 세지 않는다: 말 없음(422), 디코딩할 수 없는 업로드(422), 빈 글(요청 검증 422).
async def test_input_errors_through_the_api_are_not_counted(client, auth_headers, whisper_server):
    whisper_server.segments = []
    assert (await speech(client, auth_headers)).status_code == 422
    assert (await speech(client, auth_headers, content=b"not an audio file")).status_code == 422
    empty = {"text": "   ", "source_lang": "en", "target_lang": "ko"}
    assert (await client.post("/api/translate/text", headers=auth_headers, json=empty)).status_code == 422
    body = (await client.get("/api/health")).json()
    assert body["consecutive_model_failures"] == 0 and body["models_ok"] is True
