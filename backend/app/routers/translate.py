import logging
from contextlib import contextmanager
from typing import Annotated

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


async def execute(response: Response, **kwargs) -> TranslationResponse:
    with failures_as_http():
        result = await pipeline.translate(**kwargs)
    response.headers["Location"] = f"/api/history/{result.id}"
    return result


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


class AudioUploadTooLarge(MultiPartException):
    pass


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


router.route_class = LimitedUploadRoute


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


async def read_audio(audio: UploadFile) -> bytes:
    # Defense in depth if another caller supplies an already parsed UploadFile.
    if audio.size is not None and audio.size > pipeline.MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file is larger than 10 MB")
    content = await audio.read(pipeline.MAX_AUDIO_BYTES + 1)
    if len(content) > pipeline.MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file is larger than 10 MB")
    return content


OTHER: dict[Language, Language] = {"ko": "en", "en": "ko"}


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
        language, confidence = await run_model(models.stt.detect_language, content)
    guessed = previous_lang is not None and confidence < get_settings().dialog_language_threshold
    if guessed:
        language = OTHER[previous_lang]
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
