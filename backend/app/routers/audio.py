from uuid import UUID

from fastapi import APIRouter, HTTPException, Response
from sqlalchemy import select

from app.dependencies import CurrentUser, DbSession, StoredAudio
from app.models import AudioFile, Translation

router = APIRouter(prefix="/audio", tags=["audio"])


@router.get("/{id}")
async def get_audio(id: UUID, user: CurrentUser, db: DbSession, store: StoredAudio) -> Response:
    path = await db.scalar(
        select(AudioFile.path).join(Translation).where(AudioFile.id == id, Translation.user_id == user.id)
    )
    content = await store.read(path) if path is not None else None
    if content is None:
        raise HTTPException(404, "Audio not found")
    return Response(content, media_type="audio/wav", headers={"Cache-Control": "private, no-store"})
