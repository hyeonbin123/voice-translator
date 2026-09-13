import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import ValidationError
from python_multipart.exceptions import MultipartParseError
from starlette.formparsers import MultiPartException, MultiPartParser

from app.dependencies import CurrentUser, DbSession, Models, StoredAudio
from app.schemas.translate import LanguagePair, TextRequest, TranslationResponse
from app.services import pipeline
from app.services.errors import describe
from app.services.interfaces import Language, ModelError, NoSpeechError, UndecodableAudioError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/translate", tags=["translate"])


async def execute(response: Response, **kwargs) -> TranslationResponse:
    try:
        result = await pipeline.translate(**kwargs)
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
    # Defense in depth if another caller supplies an already parsed UploadFile.
    if audio.size is not None and audio.size > pipeline.MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file is larger than 10 MB")
    content = await audio.read(pipeline.MAX_AUDIO_BYTES + 1)
    if len(content) > pipeline.MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file is larger than 10 MB")
    return await execute(
        response,
        models=models,
        store=store,
        db=db,
        user_id=user.id,
        source=pair.source_lang,
        target=pair.target_lang,
        audio=content,
    )
