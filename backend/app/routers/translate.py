# [파일 역할] 번역 HTTP API: 글자 번역(POST /api/translate/text), 음성 번역(POST /api/translate/speech), 두
# 사람 대화(POST /api/translate/dialog). 성공하면 기록을 저장하고 201과 Location 헤더(/api/history/{id})를
# 돌려준다.
# [설계 이유] 실패를 화면이 구분할 수 있게 상황마다 고정된 상태 코드·문구로 바꾼다(docs/api.md '실패했을 때'
# 표, 문구는 계약). 음성 업로드는 10MB를 넘는 순간 읽기를 멈추도록 multipart 파서를 바꿔 끼워, 큰 파일이
# 디스크·메모리에 다 쌓이기 전에 413으로 끊는다.
# [연결] 실제 흐름은 services/pipeline.py, 언어 결정은 services/dialog.py, 모델 호출 스레드는
# services/inference.py.

import logging
from contextlib import contextmanager
from typing import Annotated

# RequestValidationError: 폼 필드의 언어 쌍 검증 실패를 FastAPI 표준 422 형식으로 내려고. APIRoute: 경로
# 처리기를 감싸 multipart 파서를 바꿔 끼우려고. MultiPartParser·MultiPartException(Starlette)과
# MultipartParseError(python-multipart): 업로드 크기 제한과 깨진 multipart 처리.
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import ValidationError
from python_multipart.exceptions import MultipartParseError
from starlette.formparsers import MultiPartException, MultiPartParser

from app.config import get_settings
from app.dependencies import CurrentUser, DbSession, Models, StoredAudio
from app.schemas.translate import DialogResponse, LanguagePair, TextRequest, TranslationResponse
from app.services import pipeline
from app.services.dialog import OTHER, choose_language
from app.services.errors import describe
from app.services.inference import run_model
from app.services.interfaces import (
    Language,
    LanguageDetector,
    ModelError,
    NoSpeechError,
    UndecodableAudioError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/translate", tags=["translate"])


# with 블록 안에서 난 모델 예외를 API 계약의 HTTP 오류로 바꾼다. 순서가 중요하다:
# UndecodableAudioError·NoSpeechError는 ModelError의 하위 클래스라 먼저 잡아야 422가 되고, 그 밖의
# ModelError는 503이 된다. 로그에는 describe()로 종류와 위치만 남긴다.
@contextmanager
def failures_as_http():
    """The API's fixed answers for a translation that failed (docs/api.md)."""
    try:
        yield
    # The log names the failure but never its messages: a library's message can quote the input (T49).
    except UndecodableAudioError as exc:
        logger.warning("Undecodable audio: %s", describe(exc))
        raise HTTPException(422, "Audio could not be decoded") from exc
    except NoSpeechError as exc:
        logger.warning("No speech recognized: %s", describe(exc))
        raise HTTPException(422, "No speech was recognized") from exc
    except pipeline.AudioTooLongError as exc:
        raise HTTPException(422, "Audio is longer than 30 seconds") from exc
    except ModelError as exc:
        logger.error("Translation pipeline failed: %s", describe(exc))
        raise HTTPException(503, "Translation service is unavailable") from exc


# 번역(준비+저장)을 실행하고, 만들어진 기록의 주소를 Location 헤더에 넣는다(201 Created의 관례).
async def execute(response: Response, **kwargs) -> TranslationResponse:
    with failures_as_http():
        result = await pipeline.translate(**kwargs)
    response.headers["Location"] = f"/api/history/{result.id}"
    return result


# 글자 번역. TextRequest가 1~500자·NUL·빈 글자·언어 쌍을 먼저 검증한다(422). 영어 입력은 pipeline 안에서 오타
# 교정을 거칠 수 있다.
# 이 경로는 아래 router.route_class 지정보다 먼저 등록되므로 일반 APIRoute를 쓴다(JSON 요청이라 multipart
# 제한이 필요 없다).
@router.post("/text", status_code=201, response_model=TranslationResponse)
async def translate_text(
    body: TextRequest,
    response: Response,
    user: CurrentUser,
    db: DbSession,
    models: Models,
    store: StoredAudio,
) -> TranslationResponse:
    return await execute(
        response,
        models=models,
        store=store,
        db=db,
        user_id=user.id,
        source=body.source_lang,
        target=body.target_lang,
        text=body.text,
    )


# 업로드가 10MB를 넘었다는 표시. MultiPartException을 상속해 Starlette 파서 안에서 던질 수 있게 하고, 아래에서
# 다른 multipart 오류(400)와 구분해 413으로 바꾼다.
class AudioUploadTooLarge(MultiPartException):
    pass


# 파일 파트의 바이트를 받는 대로 세다가 10MB를 넘으면 바로 예외를 낸다. Content-Length 헤더만 보면 chunked
# 전송이나 거짓 헤더를 막지 못하므로, 실제로 들어온 바이트를 센다. 파트가 시작될 때마다 0으로 되돌리고, 파일이
# 아닌 폼 필드(source_lang 등)는 세지 않는다.
class LimitedMultipartParser(MultiPartParser):
    """Count file bytes before Starlette spools them, including chunked requests."""

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self.file_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            self.file_bytes += end - start
            if self.file_bytes > pipeline.MAX_AUDIO_BYTES:
                raise AudioUploadTooLarge("Audio file is larger than 10 MB")
        super().on_part_data(data, start, end)


# 이 라우터의 경로 처리기를 감싸, multipart 요청이면 FastAPI가 폼을 읽기 전에 제한된 파서로 먼저 읽는다. 파일
# 1개·필드 2개까지만 받는다(audio, 언어 필드). 파싱 오류는 400 "Invalid multipart request", 크기 초과는 413.
# 읽은 폼을 request._form에 넣어 두면 FastAPI의 File/Form 검증이 다시 읽지 않고 그 결과를 쓴다(공개 setter가
# 없어 내부 속성을 씀). 처리가 끝나면 임시 파일을 닫는다. multipart가 아닌 요청은 그대로 넘겨 FastAPI가 422
# 등을 내게 한다.
class LimitedUploadRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def limited(request: Request):
            media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type != "multipart/form-data":
                return await handler(request)
            try:
                form = await LimitedMultipartParser(
                    request.headers,
                    request.stream(),
                    max_files=1,
                    max_fields=2,
                ).parse()
            except AudioUploadTooLarge as exc:
                raise HTTPException(413, "Audio file is larger than 10 MB") from exc
            except (MultiPartException, MultipartParseError) as exc:
                raise HTTPException(400, "Invalid multipart request") from exc
            # Reuse the parsed form for FastAPI's normal File/Form validation and OpenAPI.
            # Starlette has no public setter for the request form cache.
            request._form = form
            try:
                return await handler(request)
            finally:
                await form.close()

        return limited


# 이 줄 뒤에 등록되는 경로(/speech, /dialog)만 위의 제한된 경로 클래스를 쓴다.
router.route_class = LimitedUploadRoute


# 음성 번역. 폼 필드: audio(녹음 파일), source_lang, target_lang. 인식 언어는 source_lang으로 고정한다.
# 언어 쌍은 폼 필드라 TextRequest처럼 본문 모델로 검증되지 않으므로, LanguagePair를 직접 만들어 검증하고 오류
# 위치(loc) 앞에 "body"를 붙여 다른 검증 오류와 같은 형식의 422로 낸다.
@router.post("/speech", status_code=201, response_model=TranslationResponse)
async def translate_speech(
    response: Response,
    user: CurrentUser,
    db: DbSession,
    models: Models,
    store: StoredAudio,
    audio: Annotated[UploadFile, File()],
    source_lang: Annotated[Language, Form()],
    target_lang: Annotated[Language, Form()],
) -> TranslationResponse:
    try:
        pair = LanguagePair(source_lang=source_lang, target_lang=target_lang)
    except ValidationError as exc:
        errors = [{**error, "loc": ("body", *error["loc"])} for error in exc.errors()]
        raise RequestValidationError(errors) from exc
    return await execute(
        response,
        models=models,
        store=store,
        db=db,
        user_id=user.id,
        source=pair.source_lang,
        target=pair.target_lang,
        audio=await read_audio(audio),
    )


# 업로드 파일을 바이트로 읽는다. 위의 파서가 이미 10MB를 막지만, 다른 경로로 만든 UploadFile이 들어와도
# 안전하도록 크기 확인을 한 번 더 한다(한도 + 1바이트만 읽어 넘었는지 판단).
async def read_audio(audio: UploadFile) -> bytes:
    # Defense in depth if another caller supplies an already parsed UploadFile.
    if audio.size is not None and audio.size > pipeline.MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file is larger than 10 MB")
    content = await audio.read(pipeline.MAX_AUDIO_BYTES + 1)
    if len(content) > pipeline.MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file is larger than 10 MB")
    return content


# 두 사람 대화. 순서: 업로드 읽기 → 언어 판별 기능 확인(없으면 503) → 음성 검사·언어 판별(모델 스레드) → 직전
# 마디 언어와 문턱으로 처리할 언어 결정 → 그 언어에서 상대 언어로 음성 번역과 같은 흐름 실행.
# 언어 판별과 번역 안의 인식은 같은 음성을 두 번 모델에 넣는다. 응답에는 확신(소수 셋째 자리까지)과 추측
# 여부를 더한다.
@router.post("/dialog", status_code=201, response_model=DialogResponse)
async def translate_dialog(
    response: Response,
    user: CurrentUser,
    db: DbSession,
    models: Models,
    store: StoredAudio,
    audio: Annotated[UploadFile, File()],
    previous_lang: Annotated[Language | None, Form()] = None,
) -> DialogResponse:
    """Two people, one screen (T35): the utterance's language is detected and it is translated into the
    other. When the detection is unsure, the conversation is taken to alternate (docs/experiments.md 10)."""
    content = await read_audio(audio)
    if not isinstance(models.stt, LanguageDetector):
        raise HTTPException(503, "Translation service is unavailable")
    with failures_as_http():
        await run_model(pipeline.validate_audio, content)
        detected, confidence = await run_model(models.stt.detect_language, content)
    language, guessed = choose_language(
        detected, confidence, previous_lang, get_settings().dialog_language_threshold
    )
    result = await execute(
        response,
        models=models,
        store=store,
        db=db,
        user_id=user.id,
        source=language,
        target=OTHER[language],
        audio=content,
    )
    return DialogResponse(
        **result.model_dump(), language_confidence=round(confidence, 3), language_guessed=guessed
    )
